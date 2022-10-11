#!/bin/bash

seed=(2021 111 222 333 444 555 666 777 888 999)

for round in 0 1 2;
do
  for idrandom in 0;
  do
  for pt_task in 0 1 2 3
    do
      for ft_task in $(seq 0 ${pt_task});
        do
          CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune.py \
          --max_seq_length 164 \
          --batch_size 20 \
          --pt_task ${pt_task} \
          --ft_task ${ft_task} \
          --idrandom ${idrandom} \
          --ntasks 4 \
          --epoch 20 \
          --ft_sequence_file 'cpt_datasets_ft' \
          --pt_sequence_file 'cpt_datasets_pt' \
          --seed ${seed[$round]} \
          --pt_seed 111 \
          --ffn_adapter_size 768 \
          --attn_adapter_size 512 \
          --baseline 'cpt_parallel_unfreeze' \
          --few_shot \
          --unfreeze_lm
      done
    done
  done
done
