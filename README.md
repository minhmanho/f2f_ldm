# F2FLDM

#### [[Page]](https://minhmanho.github.io/f2f_ldm/) [[Paper]](https://arxiv.org/abs/2404.12650)

[F2FLDM: Latent Diffusion Models with Histopathology Pre-Trained Embeddings for Unpaired Frozen Section to FFPE Translation](https://arxiv.org/abs/2404.12650)<br>
[Man M. Ho](https://minhmanho.github.io/), Shikha Dubey, Yosep Chong, Beatrice Knudsen, Tolga Tasdizen<br>
In ArXiv, 2024.

# Environment Setup

## Create Anaconda Environment

To create the Anaconda environment, run the following command:

```shell
conda create -n f2fldm python numpy pillow
conda activate f2fldm
```

## Install Dependencies

To install the required dependencies, execute the following commands:

PyTorch:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Other packages:

```shell
pip install huggingface transformers diffusers pytorch-lightning==2.1.2 timm accelerate datasets
```

Replace official diffusers with our customized diffusers:

```shell
cd diffusers
pip install -e .
```

# Run Frozen Section to FFPE Image Translation
To run the Frozen Section to FFPE Image Translation, use the following command:

```shell
CUDA_VISIBLE_DEVICES=0 python run.py \
    --input_path="images/testA" \
    --output_path="images/testA_out" \
    --sd_model_id="stabilityai/stable-diffusion-xl-base-1.0" \
    --lora_model_path="checkpoints" \
    --high_noise_frac=0.5 \ # Strength S
    --target_resolution=1024 \
    --unet_addition_embed_type="text_latent_addembeddingft" \
    --unet_add_embedding_input=4096 \
    --unet_add_embedding_output=1280 \
    --guidance_scale=12.0 \ # Guidance Scale GS
    --feat_model_name="DINO" \
    --et_model_name="cyclegan" \
    --et_model_path="embedding_translation/checkpoints/et_cyclegan.ckpt" \
    --reg="l0" \ # Regularization on Noise Difference, comment this line for not using it.
    --et_weight=0.25 # Embedding Translation Weight alpha
```

or

```shell
scripts/test.sh
```

# Finetune pre-trained SDXL with LoRA for Frozen Section to FFPE Image Translation

To finetune the pre-trained SDXL model for Frozen Section to FFPE Image Translation, use the following command:

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_dir="images" \
  --train_data_dir="username/f2f_tcga_kidney_1024/train/metadata.csv" --caption_column="text" \
  --resolution=512 --random_flip --center_crop\
  --train_batch_size=1 \
  --rank=8 \ # LoRA rank
  --max_train_steps=150000 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=1337 \
  --output_dir="checkpoints" \
  --train_text_encoder \
  --disable_crop_top_lefts \
  --unet_addition_embed_type="text_latent_addembeddingft"
```

or 

```shell
scripts/train.sh
```

## Citation
If you find this work useful, please consider citing:
```
@article{ho2024f2fldm,
  title={F2FLDM: Latent Diffusion Models with Histopathology Pre-Trained Embeddings for Unpaired Frozen Section to FFPE Translation},
  author={Ho, Man M and Dubey, Shikha and Chong, Yosep and Knudsen, Beatrice and Tasdizen, Tolga},
  journal={arXiv preprint arXiv:2404.12650},
  year={2024}
}
```

## Contact
If you have any questions, feel free to contact me at [manminhho.cs@gmail.com](mailto:manminhho.cs@gmail.com)


