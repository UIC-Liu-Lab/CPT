"""Training and model configuration for continual post-training and end tasks fine-tuning."""
import argparse
import logging

from transformers import MODEL_MAPPING

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parsing_posttrain():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='roberta-base',
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='./ckpt', help="Where to store the final model.")
    parser.add_argument("--saved_output_dir", type=str, default='./ckpt', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=111, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=16,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--baseline", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument('--share_weight', action='store_true')
    parser.add_argument(
        "--max_train_samples",
        type=float,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
             "value if set.",
    )
    parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--pt_task", type=int, help="task id")
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument("--base_dir", default='./outputs', type=str, help="task id")
    parser.add_argument("--s", type=int, help="smax")
    parser.add_argument("--smax", default=400, type=int, help="smax")
    parser.add_argument('--thres_cosh', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--thres_emb', default=6, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument("--sequence_file", type=str, help="continual post-training sequence")
    parser.add_argument("--ffn_adapter_size", type=int, default=512, help="adapter size for ffn part.")
    parser.add_argument("--attn_adapter_size", type=int, default=256, help="adapter size for attn part.")
    parser.add_argument("--adapter_size", type=int, default=2000, help="general adapter size.")
    parser.add_argument("--idrandom", type=int, help="which sequence to use")
    args = parser.parse_args()

    return args


def parsing_finetune():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--adapter_lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--params', type=str, default=None)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--least_epoch', type=int)
    parser.add_argument('--patient', default=3, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--problem_type', type=str, default='single_label_classification')
    # For recording results only ==============================================
    parser.add_argument('--round', type=int, help='the round number in repeated experiments.')
    parser.add_argument("--baseline", type=str, help="one,ncl")
    parser.add_argument('--saved_model', type=str)
    parser.add_argument("--saved_output_dir", type=str, default='./ckpt', help="Where to store the final model.")
    parser.add_argument("--finetune_type", type=str, help="Where to store the final model.")

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument("--generator", default='', type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--train_sample_ratio", default=None, type=float,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--n_tokens", type=int, default=100, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides epoch.",
    )
    parser.add_argument("--how_to_block", type=str, help="['grad','head_mask']")
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--pt_task", type=int, help="task id")
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--base_dir", default='./outputs', type=str, help="task id")
    parser.add_argument("--smax", default=400, type=int, help="smax")
    parser.add_argument("--idrandom", type=int, help="which sequence to use")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--ft_sequence_file", type=str, help="corresponding fine-tuning sequence")
    parser.add_argument("--pt_sequence_file", type=str, help="corresponding post-training sequence")
    parser.add_argument("--pt_seed", type=int, help="seed for post-training, used for ckpt loading.")
    parser.add_argument('--unfreeze_lm', action='store_true')
    parser.add_argument("--adapter_size", type=int, default=2000, help="general adapter size.")
    parser.add_argument("--ffn_adapter_size", type=int, default=512, help="adapter size for ffn part.")
    parser.add_argument("--attn_adapter_size", type=int, default=200, help="adapter size for attn part.")
    parser.add_argument('--few_shot', action="store_true", help="few shot version.")
    args = parser.parse_args()

    return args
