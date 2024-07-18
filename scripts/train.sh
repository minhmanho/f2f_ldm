#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_dir="images" \
  --train_data_dir="username/f2f_tcga_kidney_1024/train/metadata.csv" --caption_column="text" \
  --resolution=512 --random_flip --center_crop\
  --train_batch_size=1 \
  --rank=8 \
  --max_train_steps=150000 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=1337 \
  --output_dir="checkpoints" \
  --train_text_encoder \
  --disable_crop_top_lefts \
  --unet_addition_embed_type="text_latent_addembeddingft"
