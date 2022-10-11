"""Individual fine-tuning approach for CPT and baseline models."""
import logging
import math
import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        return

    def train(self, model, accelerator, train_loader, test_loader):
        special_lr = ['adapter']
        try:
            adapter_lr = self.args.adapter_lr
        except Exception as e:
            print(
                "Notice: You are using the same learning rate for adapter and backbone model.")
            adapter_lr = self.args.lr

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in special_lr) and p.requires_grad],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.lr,
            },
            # Set different learning rate for adapter.
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' in n],
                'weight_decay': self.args.weight_decay,
                'lr': adapter_lr,
            }
        ]
        # Set the optimizer
        optimizer = AdamW(optimizer_grouped_parameters)

        num_update_steps_per_epoch = math.ceil(
            len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(
                self.args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Prepare everything with the accelerator
        model, optimizer, train_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader)

        # Train!
        logger.info("***** Running training *****")
        logger.info(
            f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.dataset_name}, seed = {self.args.seed}")

        for epoch in range(self.args.epoch):
            print("Epoch {} started".format(epoch))
            train_acc, training_loss = self.train_epoch(
                model, optimizer, train_loader, accelerator, lr_scheduler)
            print("train acc = {:.4f}, training loss = {:.4f}".format(
                train_acc, training_loss))

            micro_f1, macro_f1, acc, test_loss = self.eval(
                model, test_loader, accelerator)

            logger.info(
                "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(self.args.model_name_or_path,
                                                                                        self.args.dataset_name,
                                                                                        macro_f1,
                                                                                        acc, self.args.seed))

        if accelerator.is_main_process:
            if self.args.few_shot:
                progressive_f1_path = os.path.join(self.args.output_dir + '/../',
                                                   'few_shot_progressive_f1_' + str(self.args.seed))
                progressive_acc_path = os.path.join(self.args.output_dir + '/../',
                                                    'few_shot_progressive_acc_' + str(self.args.seed))
            else:
                progressive_f1_path = os.path.join(self.args.output_dir + '/../',
                                                   'progressive_f1_' + str(self.args.seed))
                progressive_acc_path = os.path.join(self.args.output_dir + '/../',
                                                    'progressive_acc_' + str(self.args.seed))
            print('progressive_f1_path: ', progressive_f1_path)
            print('progressive_acc_path: ', progressive_acc_path)

            if os.path.exists(progressive_f1_path):
                f1s = np.loadtxt(progressive_f1_path)
                accs = np.loadtxt(progressive_acc_path)

            else:
                f1s = np.zeros(
                    (self.args.ntasks, self.args.ntasks), dtype=np.float32)
                accs = np.zeros(
                    (self.args.ntasks, self.args.ntasks), dtype=np.float32)

            f1s[self.args.pt_task][self.args.ft_task] = macro_f1
            np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

            accs[self.args.pt_task][self.args.ft_task] = acc
            np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

            if self.args.ft_task == self.args.ntasks - 1:  # last ft task, we need a final one
                if self.args.few_shot:
                    final_f1 = os.path.join(
                        self.args.output_dir + '/../', 'few_shot_f1_' + str(self.args.seed))
                    final_acc = os.path.join(
                        self.args.output_dir + '/../', 'few_shot_acc_' + str(self.args.seed))

                    forward_f1 = os.path.join(self.args.output_dir + '/../',
                                              'few_shot_forward_f1_' + str(self.args.seed))
                    forward_acc = os.path.join(self.args.output_dir + '/../',
                                               'few_shot_forward_acc_' + str(self.args.seed))
                else:
                    final_f1 = os.path.join(
                        self.args.output_dir + '/../', 'f1_' + str(self.args.seed))
                    final_acc = os.path.join(
                        self.args.output_dir + '/../', 'acc_' + str(self.args.seed))

                    forward = os.path.join(
                        self.args.output_dir + '/../', 'forward_f1_' + str(self.args.seed))
                    forward_acc = os.path.join(
                        self.args.output_dir + '/../', 'forward_acc_' + str(self.args.seed))

                print('final_f1: ', final_f1)
                print('final_acc: ', final_acc)

                if self.args.baseline == 'one':
                    with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                        for j in range(accs.shape[1]):
                            file.writelines(str(accs[j][j]) + '\n')
                            f1_file.writelines(str(f1s[j][j]) + '\n')

                else:
                    with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                        for j in range(accs.shape[1]):
                            file.writelines(str(accs[-1][j]) + '\n')
                            f1_file.writelines(str(f1s[-1][j]) + '\n')

                    with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                        for j in range(accs.shape[1]):
                            file.writelines(str(accs[j][j]) + '\n')
                            f1_file.writelines(str(f1s[j][j]) + '\n')

    def train_epoch(self, model, optimizer, dataloader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)),
                            disable=not accelerator.is_local_main_process)
        model.train()
        train_acc = 0.0
        training_loss = 0.0
        total_num = 0.0
        t = torch.LongTensor([self.args.ft_task]).to(model.device)
        s = self.args.smax
        for batch, inputs in enumerate(dataloader):

            res = model(**inputs, return_dict=True, t=t, s=s)
            outp = res.logits
            loss = res.loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            pred = outp.max(1)[1]
            train_acc += (inputs['labels'] == pred).sum().item()
            training_loss += loss.item()
            total_num += inputs['labels'].size(0)

            progress_bar.update(1)

        return train_acc / total_num, training_loss / total_num

    def eval(self, model, dataloader, accelerator):
        model.eval()
        label_list = []
        prediction_list = []
        total_loss = 0
        total_num = 0
        t = torch.LongTensor([self.args.ft_task]).to(accelerator.device)
        s = self.args.smax
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)),
                            disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for _, inputs in enumerate(dataloader):
                input_ids = inputs['input_ids']
                res = model(**inputs, return_dict=True, t=t, s=s)

                real_b = input_ids.size(0)
                loss = res.loss
                outp = res.logits
                pred = outp.max(1)[1]

                total_loss += loss.data.cpu().numpy().item() * real_b
                total_num += real_b
                label_list += inputs['labels'].cpu().numpy().tolist()
                prediction_list += pred.cpu().numpy().tolist()
                progress_bar.update(1)

        micro_f1 = f1_score(label_list, prediction_list, average='micro')
        macro_f1 = f1_score(label_list, prediction_list, average='macro')
        accuracy = sum([float(label_list[i] == prediction_list[i]) for i in range(len(label_list))]) * 1.0 / len(
            prediction_list)

        return micro_f1, macro_f1, accuracy, total_loss / total_num
