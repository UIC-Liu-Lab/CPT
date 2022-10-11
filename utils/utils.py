import os
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here


def setup_writer(name):
    writer = SummaryWriter(name)
    return writer


def log_loss(writer, loss_name='training loss', scalar_value=None, global_step=None):
    # ...log the running loss
    writer.add_scalar(loss_name, scalar_value, global_step=global_step)


def prepare_sequence_finetune(args):
    with open(os.path.join('sequences', args.ft_sequence_file.replace('_reduce', '')), 'r') as f:
        datas = f.readlines()[args.idrandom]
        ft_data = datas.split()

    args.task_name = ft_data
    args.current_dataset_name = ft_data[args.ft_task]

    with open(os.path.join('sequences', args.pt_sequence_file), 'r') as f:
        datas = f.readlines()[args.idrandom]
        pt_data = datas.split()

    if "cpt_datasets" in args.pt_sequence_file:
        args.dataset_name = 'pt'
    else:
        raise NotImplementedError(
            f"The current sequence file {args.sequence_file} is not supported yet!")

    output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(ft_data[args.ft_task]) + "_roberta/"
    if args.pt_task == -1:  # Get the PLM results before any post-training.
        ckpt = "roberta-base"
    else:
        ckpt = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.pt_seed) + "/" + str(
            args.baseline) + '/' + str(args.dataset_name) + '/' + str(pt_data[args.pt_task]) + "_roberta/"

    args.task = args.ft_task

    args.output_dir = output

    if 'base' in ckpt:
        args.adapter_path = "None"
    else:
        args.adapter_path = ckpt + str(args.pt_seed) + ".model"

    args.saved_output_dir = [args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(ft_data[t]) + "_roberta/" for t in
        range(args.ft_task + 1)]

    args.model_name_or_path = ckpt
    if 'cpt' in args.baseline:
        args.model_name_or_path = 'roberta-base'

    print('saved_output_dir: ', args.saved_output_dir)
    print('output_dir: ', args.output_dir)
    print('args.dataset_name: ', args.dataset_name)
    print('current_dataset_name: ', args.current_dataset_name)
    print('args.model_name_or_path: ', args.model_name_or_path)
    print("adapter_path", args.adapter_path)
    return args


def prepare_sequence_posttrain(args):
    with open(os.path.join('sequences', args.sequence_file), 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    args.task_name = data
    args.current_dataset_name = data[args.pt_task]

    if "cpt_datasets" in args.sequence_file:
        args.dataset_name = 'pt'
    else:
        raise NotImplementedError(
            f"The current sequence file {args.sequence_file} is not supported yet!")

    output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.pt_task]) + "_roberta/"
    ckpt = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.pt_task - 1]) + "_roberta/"

    if args.pt_task > 0:
        args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
            args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.pt_task - 1]) + "_roberta/"
    else:
        args.prev_output = ''
    args.task = args.pt_task

    args.output_dir = output

    args.saved_output_dir = [args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[t]) + "_roberta/" for t in
        range(args.pt_task + 1)]

    if args.task == 0:  # no pre-trained for the first
        args.model_name_or_path = "roberta-base"
        args.adapter_path = "None"
    else:
        args.model_name_or_path = ckpt
        args.adapter_path = ckpt + str(args.seed) + ".model"

    if 'cpt' in args.baseline:
        args.model_name_or_path = 'roberta-base'

    print('saved_output_dir: ', args.saved_output_dir)
    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('dataset_name: ', args.dataset_name)
    print('current_dataset_name: ', args.current_dataset_name)
    print('model_name_or_path: ', args.model_name_or_path)
    print('adapter_path: ', args.adapter_path)

    return args


def lookfor_model_finetune(args):

    if 'cpt_parallel' in args.baseline:
        from networks.adapter_mask import load_roberta_adapter_model, RobertaMaskForSequenceClassification
        model = RobertaMaskForSequenceClassification.from_pretrained("roberta-base", num_labels=args.class_num,
                                                                     problem_type=args.problem_type)
        load_roberta_adapter_model(
            model,
            checkpoint=args.adapter_path,
            mode="parallel",
            attn_adapter_size=args.attn_adapter_size,
            ffn_adapter_size=args.ffn_adapter_size,
            ntasks=args.ntasks,
        )

        if args.unfreeze_lm:
            for p in model.parameters():
                p.requires_grad = True
        model.classifier.requires_grad = True

        adapter_masks = \
            [model.roberta.encoder.layer[layer_id].attention.self.adapter_mask for layer_id in range(12)] + \
            [model.roberta.encoder.layer[layer_id].output.adapter_mask for layer_id in range(
                12)]

        for mask in adapter_masks:
            for param in mask.efc1.parameters():
                param.requires_grad = False
            for param in mask.efc2.parameters():
                param.requires_grad = False

    return model


def lookfor_model_posttrain(args):
    if 'cpt_parallel' in args.baseline:
        from networks.adapter_mask import (
            RobertaMaskForMaskedLM,
            load_roberta_adapter_model,
        )
        model = RobertaMaskForMaskedLM.from_pretrained('roberta-base')
        load_roberta_adapter_model(
            model,
            checkpoint=args.adapter_path,
            mode="parallel",
            attn_adapter_size=args.attn_adapter_size,
            ffn_adapter_size=args.ffn_adapter_size,
            ntasks=args.ntasks,
        )
    else:
        raise NotImplementedError()

    return model
