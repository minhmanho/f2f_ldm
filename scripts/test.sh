#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py \
    --input_path="images/testA" \
    --output_path="images/testA_out" \
    --sd_model_id="stabilityai/stable-diffusion-xl-base-1.0" \
    --lora_model_path="checkpoints" \
    --high_noise_frac=0.5 \
    --target_resolution=1024 \
    --unet_addition_embed_type="text_latent_addembeddingft" \
    --unet_add_embedding_input=4096 \
    --unet_add_embedding_output=1280 \
    --guidance_scale=12.0 \
    --feat_model_name="DINO" \
    --et_model_name="cyclegan" \
    --et_model_path="embedding_translation/checkpoints/et_cyclegan.ckpt" \
    --reg="l0" \
    --et_weight=0.25
