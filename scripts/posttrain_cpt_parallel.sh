#!/bin/bash

for idrandom in 0;
do
  for task in 0 1 2 3;
  do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m torch.distributed.launch --nproc_per_node 2 --master_port ${port} --use_env posttrain.py \
    --per_device_train_batch_size 25 \
    --fp16 \
    --max_seq_length 164 \
    --idrandom ${idrandom} \
    --ntasks 4 \
    --sequence_file 'cpt_datasets_pt' \
    --pt_task ${task} \
    --ffn_adapter_size 768 \
    --attn_adapter_size 512 \
    --baseline 'cpt_parallel_unfreeze'
  done
done
