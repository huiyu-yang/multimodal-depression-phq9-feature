#!/bin/bash
set -e

python train_lora_pf9.py \
  --model_dir /path/to/Llama-3.1-8B-Instruct \
  --train_jsonl /path/to/CMDC_5fold_jsonl_loss_of_interest/fold1_train.jsonl \
  --output_dir /path/to/CMDC_lora_llama8b_loss_of_interest/fold1 \
  --max_length 2048 \
  --epochs 5 \
  --lr 1e-4 \
  --batch_size 2 \
  --grad_accum 4 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.10 \
  --logging_steps 5 \
  --save_steps 50 \
  --num_workers 4 \
  --dataloader_workers 0