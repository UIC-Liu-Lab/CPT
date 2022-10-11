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
Fine-tuning models on domain-specific downstream tasks.
"""
from approaches.finetune import Appr
from config import parsing_finetune
import logging
import os
import random

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from transformers import (
    set_seed,
)

from dataloader.data import get_dataset, dataset_class_num
from utils import utils

# Set up logger
logger = logging.getLogger(__name__)


def main():
    args = parsing_finetune()
    args = utils.prepare_sequence_finetune(args)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    appr = Appr(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if args.log_dir is not None:
        handler = logging.FileHandler(args.log_dir)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # if use multilabel-classification datasets:
    if 'multi' in args.dataset_name:
        args.problem_type = 'multi_label_classification'

    accelerator.wait_for_everyone()

    # Get the datasets and process the data.
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    max_length = args.max_seq_length

    logger.info('==> Preparing data..')

    datasets = get_dataset(args.current_dataset_name,
                           tokenizer=tokenizer, args=args)
    print('dataset_name: ', args.current_dataset_name)

    print('train_loader: ', len(datasets['train']))
    print('test_loader: ', len(datasets['test']))

    train_dataset = datasets['train']
    test_dataset = datasets['test']

    test_dataset = test_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
    test_dataset.set_format(type='torch', columns=[
                            'input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=8)

    train_dataset = train_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
    train_dataset.set_format(type='torch', columns=[
                             'input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=8)  # consider batch size

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    if args.class_num is None:
        args.class_num = dataset_class_num[args.current_dataset_name]

    # Declare the model and set the training parameters.
    logger.info('==> Building model..')

    model = utils.lookfor_model_finetune(args)
    
    appr.train(model, accelerator, train_loader, test_loader)


if __name__ == '__main__':
    main()
