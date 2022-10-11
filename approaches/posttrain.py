"""Continual post-training approach for CPT and baseline models."""
import logging
import math
import os
import shutil

import numpy as np
import torch
from accelerate import DistributedType
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    get_scheduler,
)

from networks.adapter_mask import save_roberta_adapter_model as save_cpt_adapter
from networks.adapter_mask.my_optimization import BertAdam as my_Adam
from networks.adapter_mask.roberta_adapter import get_view_for
from networks.adapter_mask.roberta_adapter import mask as MASK
from utils import utils

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def train(self, model, accelerator, train_dataset, tokenizer, train_dataloader):

        # ********************************* before tranining *********************************

        self.args.t = torch.LongTensor([self.args.task]).to(accelerator.device)
        if os.path.exists(os.path.join(self.args.prev_output, 'mask_pre')):
            mask_pre = torch.load(os.path.join(
                self.args.prev_output, 'mask_pre'), map_location=torch.device('cpu'))
            mask_back = torch.load(os.path.join(self.args.prev_output, 'mask_back'),
                                   map_location=torch.device('cpu'))

            for k, v in mask_pre.items():
                mask_pre[k] = mask_pre[k].cuda()

            for k, v in mask_back.items():
                mask_back[k] = mask_back[k].cuda()
        else:
            mask_pre = None
            mask_back = None

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                'name': [n for n, p in model.named_parameters()
                         if p.requires_grad and not any(nd in n for nd in no_decay)],
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                'name': [n for n, p in model.named_parameters()
                         if p.requires_grad and any(nd in n for nd in no_decay)],
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate

            },
        ]

        optimizer = my_Adam(optimizer_grouped_parameters)
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader = accelerator.my_prepare(
            model, optimizer, train_dataloader)

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps)

        if self.args.max_samples is not None:
            self.args.max_train_steps = self.args.max_samples // (
                self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps)

        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(
                self.args.max_train_steps / num_update_steps_per_epoch)

        self.args.num_warmup_steps = int(
            float(self.args.warmup_proportion) * float(self.args.max_train_steps))  # 0.1

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Train!
        total_batch_size = self.args.per_device_train_batch_size * \
            accelerator.num_processes * self.args.gradient_accumulation_steps

        if accelerator.is_main_process:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            logger.info(
                f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(
                f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(
                f"  Total optimization steps = {self.args.max_train_steps}")
            logger.info(
                f"  Total samples = {self.args.max_train_steps * total_batch_size}")
            logger.info(
                f"  Learning Rate = {self.args.learning_rate}, Warmup Num = {self.args.num_warmup_steps}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f"  Seq ID = {self.args.idrandom}, Task id = {self.args.task}, dataset name = {self.args.dataset_name}")
            logger.info(
                f"  Baseline = {self.args.baseline}, Smax = {self.args.smax}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps),
                            disable=not accelerator.is_local_main_process)
        completed_steps = 0
        global_step = 0

        if accelerator.is_main_process:
            tensorboard_file = os.path.join(
                self.args.output_dir, str(self.args.dataset_name) + '_log')
            print('tensorboard_file: ', tensorboard_file)
            if os.path.isdir(tensorboard_file):
                shutil.rmtree(tensorboard_file)
            writer = utils.setup_writer(tensorboard_file)

        try:

            for epoch in range(self.args.num_train_epochs):
                # break
                model.train()
                for step, inputs in enumerate(train_dataloader):
                    self.args.s = (self.args.smax - 1 / self.args.smax) * step / len(
                        train_dataloader) + 1 / self.args.smax

                    outputs = model(**inputs, return_dict=True,
                                    t=self.args.t, s=self.args.s)

                    loss = outputs.loss

                    loss = loss / self.args.gradient_accumulation_steps
                    accelerator.backward(loss)  # sync

                    # Restrict layer gradients in backprop
                    if self.args.task > 0:
                        for n, p in model.named_parameters():
                            if n in mask_back and p.grad is not None:
                                p.grad.data *= mask_back[n]

                    for n, p in model.named_parameters():
                        if 'adapter_mask.e' in n or n.startswith('e'):
                            num = torch.cosh(torch.clamp(
                                self.args.s * p.data, -self.args.thres_cosh, self.args.thres_cosh)) + 1
                            den = torch.cosh(p.data) + 1
                            p.grad.data *= self.args.smax / self.args.s * num / den

                    global_step += 1

                    if step % self.args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                        optimizer.my_step(
                            type='mask', t=self.args.t, mask_back=mask_back)
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                        for n, p in model.named_parameters():
                            if 'adapter_mask.e' in n or n.startswith('e'):
                                p.data = torch.clamp(
                                    p.data, -self.args.thres_emb, self.args.thres_emb)

                        progress_bar.set_description(
                            'Train Iter (loss=%5.3f)' % loss.item())  # show the loss, mean while

                        if accelerator.is_main_process:
                            utils.log_loss(
                                writer, scalar_value=loss.item(), global_step=global_step)
                            utils.log_loss(writer, loss_name=' MLM loss', scalar_value=outputs.loss.item(),
                                           global_step=global_step)

                    # break
                    if completed_steps >= self.args.max_train_steps:
                        break

            self.after_training_op(
                accelerator, model, tokenizer, mask_pre=mask_pre)

        except KeyboardInterrupt:  # Even if control-C, still want to save model
            return

    def after_training_op(self, accelerator, model, tokenizer, mask_pre=None):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            try:
                unwrapped_model.model.save_pretrained(self.args.output_dir)
            except AttributeError:
                unwrapped_model.save_pretrained(self.args.output_dir)
            tokenizer.save_pretrained(self.args.output_dir)

        self.args.s = self.args.smax
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            mask_pre_path = os.path.join(self.args.output_dir, 'mask_pre')
            mask_back_path = os.path.join(self.args.output_dir, 'mask_back')
            unwrapped_model = accelerator.unwrap_model(model)
            save_cpt_adapter(unwrapped_model, os.path.join(
                self.args.output_dir, str(self.args.seed) + '.model'), accelerator)
            mask = MASK(unwrapped_model, self.args.t, s=self.args.smax,
                        adapter_type="parallel" if "parallel" in self.args.baseline else "sequential")
            for key, value in mask.items():
                mask[key] = torch.autograd.Variable(
                    value.data.clone(), requires_grad=False)
            if self.args.t == 0:
                mask_pre = mask
            else:
                for key, value in mask_pre.items():
                    mask_pre[key] = torch.max(mask_pre[key].to(
                        unwrapped_model.device), mask[key].to(unwrapped_model.device))

            # Weights mask
            mask_back = {}
            for n, p in unwrapped_model.named_parameters():
                vals = get_view_for(unwrapped_model, n, p, mask_pre)
                if vals is not None:
                    mask_back[n] = 1 - vals

            accelerator.save(mask_pre, mask_pre_path)  # not in state_dict
            accelerator.save(mask_back, mask_back_path)

    def eval(self, model, eval_dataloader, eval_dataset, accelerator):
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        model.eval()
        losses = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'], labels=batch['labels'], return_dict=True, t=self.args.t,
                    s=self.args.s)

            loss = outputs.loss
            losses.append(accelerator.gather(
                loss.repeat(self.args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        return perplexity

    def save_ppl(self, t, u, perplexity):
        logger.info(f"Pre-trained: {t}, Test: {u}, perplexity: {perplexity}")

        if self.args.pt_task == -1:  # base, no posst-train
            progressive_ppl_path = os.path.join(
                self.args.output_dir, 'progressive_ppl_' + str(self.args.seed))
        else:
            progressive_ppl_path = os.path.join(
                self.args.output_dir + '../', 'progressive_ppl_' + str(self.args.seed))
        print('progressive_ppl_path: ', progressive_ppl_path)

        if os.path.exists(progressive_ppl_path):
            print('loading *************')
            ppls = np.loadtxt(progressive_ppl_path)
        else:
            ppls = np.zeros(
                (self.args.ntasks, self.args.ntasks), dtype=np.float32)

        ppls[t][u] = perplexity
        np.savetxt(progressive_ppl_path, ppls, '%.4f', delimiter='\t')

        if u == self.args.ntasks - 1:
            if self.args.pt_task == -1:
                final_f1 = os.path.join(
                    self.args.output_dir, 'ppl_' + str(self.args.seed))
                forward_f1 = os.path.join(
                    self.args.output_dir, 'forward_ppl_' + str(self.args.seed))
            else:
                final_f1 = os.path.join(
                    self.args.output_dir + '../', 'ppl_' + str(self.args.seed))
                forward_f1 = os.path.join(
                    self.args.output_dir + '../', 'forward_ppl_' + str(self.args.seed))
            print('final_f1: ', final_f1)

            with open(final_f1, 'w') as f1_file:
                for j in range(ppls.shape[1]):
                    f1_file.writelines(str(ppls[-1][j]) + '\n')

            with open(forward_f1, 'w') as f1_file:
                for j in range(ppls.shape[1]):
                    f1_file.writelines(str(ppls[j][j]) + '\n')
