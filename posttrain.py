#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Continually post-train models with a sequence of domain-specific data using masked language model.
"""

import logging
import os
import random

import datasets
import torch
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import (
    MODEL_MAPPING,
    RobertaTokenizer,
    set_seed, DataCollatorForLanguageModeling,
)

from approaches.posttrain import Appr
from config import parsing_posttrain
from dataloader.data import get_dataset
from networks.adapter_mask.MyAccelerator import myAccelerator
from utils import utils

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    args = parsing_posttrain()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args = utils.prepare_sequence_posttrain(args)

    appr = Appr(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if 'cpt' not in args.baseline:
        accelerator = Accelerator(fp16=args.fp16, kwargs_handlers=[ddp_kwargs])
    else:
        print(args.fp16)
        accelerator = myAccelerator(fp16=args.fp16)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            pass
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = utils.lookfor_model_posttrain(args)
    accelerator.wait_for_everyone()

    # Get the dataset
    if args.current_dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = get_dataset(
            args.current_dataset_name, tokenizer=None, args=args)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (
                    total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    train_dataset = tokenized_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    print('train_dataset: ', len(train_dataset))
    if args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        train_dataset = train_dataset.select(
            range(int(args.max_train_samples)))

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,
        num_workers=0
    )

    appr.train(model, accelerator, train_dataset, tokenizer, train_dataloader)


if __name__ == "__main__":
    main()
