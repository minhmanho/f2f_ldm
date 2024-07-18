import argparse
import os
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from pipeline import SDXLDDIMPipeline, SDXLImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModelDev, DDIMScheduler
from embedding_translation import CycleGAN
from feature_extractor import get_feat_model, get_transform

def list_int_arg(raw_value: str) -> List[int]:
    return [int(item) for item in raw_value.split(',')]

def list_float_arg(raw_value: str) -> List[float]:
    return [float(item) for item in raw_value.split(',')]

def make_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="A delightful script for image processing with Stable Diffusion and more.")
    argparser.add_argument("--input_path", type=str, required=True, help="Path to the input images.")
    argparser.add_argument("--output_path", type=str, required=True, help="Path to save the output images.")
    argparser.add_argument("--sd_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Model ID for Stable Diffusion.")
    argparser.add_argument("--lora_model_path", type=str, required=True, help="Path to the LoRA model.")
    argparser.add_argument("--unet_latent_proj_ckpt_path", type=str, default=None, help="Path to the UNet latent projection checkpoint.")
    argparser.add_argument("--unet_add_embedding_ckpt_path", type=str, default=None, help="Path to the UNet additional embedding checkpoint.")
    argparser.add_argument("--unet_add_embedding_input", type=int, default=None, help="Input dimension for additional embedding.")
    argparser.add_argument("--unet_add_embedding_output", type=int, default=None, help="Output dimension for additional embedding.")
    argparser.add_argument("--unet_addition_embed_type", type=str, default="text_latent", help="Type of additional embedding.")
    argparser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale for the model.")
    argparser.add_argument("--feat_model_name", type=str, default=None, help="Feature model name.")
    argparser.add_argument("--et_model_name", type=str, default=None, help="Embedding translation model name.")
    argparser.add_argument("--et_model_path", type=str, default=None, help="Path to the embedding translation model.")
    argparser.add_argument("--target_resolution", type=int, default=1024, help="Resolution of the target images.")
    argparser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process.")
    argparser.add_argument("--reg", default=None, choices=["l0", "l1"], help="Regularize changes with l0 or l1 norm.")
    argparser.add_argument("--high_noise_frac", type=float, default=None, help="Fraction of high noise tiles for mixed prompt.")
    argparser.add_argument("--et_weight", type=float, default=1.0, help="Weight for embedding translation.")
    argparser.add_argument("--skip_existing", action="store_true", help="Skip existing files.")
    return argparser.parse_args()

def setup_output_path(args: argparse.Namespace) -> str:
    out_folder = []
    if args.guidance_scale:
        out_folder.append(f"guidance_scale={args.guidance_scale}")
    if args.reg:
        out_folder.append(f"reg={args.reg}")
    if args.high_noise_frac:
        out_folder.append(f"high_noise_frac={args.high_noise_frac}")
    if args.et_weight != 1.0:
        out_folder.append(f"et_weight={args.et_weight}")
    output_path = os.path.join(args.output_path, "__".join(out_folder))
    os.makedirs(output_path, exist_ok=True)
    return output_path

def load_unet(args: argparse.Namespace, weight_dtype: torch.dtype) -> UNet2DConditionModelDev:
    unet = UNet2DConditionModelDev.from_pretrained(
        args.sd_model_id, subfolder="unet", revision=None, addition_embed_type=args.unet_addition_embed_type, 
        low_cpu_mem_usage=False, torch_dtype=weight_dtype
    )

    if args.unet_add_embedding_ckpt_path:
        print("Loading add_embedding from", args.unet_add_embedding_ckpt_path)
        unet.reset_add_embedding(args.unet_add_embedding_input, args.unet_add_embedding_output)
        unet.add_embedding.load_state_dict(torch.load(args.unet_add_embedding_ckpt_path))
    elif os.path.exists(os.path.join(args.lora_model_path, "add_embedding.pt")):
        print("Loading add_embedding from", os.path.join(args.lora_model_path, "add_embedding.pt"))
        unet.reset_add_embedding(args.unet_add_embedding_input, args.unet_add_embedding_output)
        unet.add_embedding.load_state_dict(torch.load(os.path.join(args.lora_model_path, "add_embedding.pt")))
    else:
        print("No add_embedding.pt found, using default add_embedding")

    if args.unet_latent_proj_ckpt_path:
        print("Loading latent projection from", args.unet_latent_proj_ckpt_path)
        unet.latent_proj.load_state_dict(torch.load(args.unet_latent_proj_ckpt_path))
    elif os.path.exists(os.path.join(args.lora_model_path, "latent_proj.pt")):
        print("Loading latent projection from", os.path.join(args.lora_model_path, "latent_proj.pt"))
        unet.latent_proj.load_state_dict(torch.load(os.path.join(args.lora_model_path, "latent_proj.pt")))
    else:
        print("No latent_proj.pt found, using default latent_proj")
    
    return unet

def load_pipelines(args: argparse.Namespace, unet: UNet2DConditionModelDev, weight_dtype: torch.dtype) -> Tuple[SDXLDDIMPipeline, Union[SDXLImg2ImgPipeline, StableDiffusionXLPipeline]]:
    ddim_pipe = SDXLDDIMPipeline.from_pretrained(args.sd_model_id, unet=unet, torch_dtype=weight_dtype, safety_checker=None)
    ddim_pipe.to("cuda")
    ddim_pipe.load_lora_weights(args.lora_model_path)

    if args.high_noise_frac is not None:
        ddpm_pipe = SDXLImg2ImgPipeline.from_pretrained(args.sd_model_id, unet=unet, torch_dtype=weight_dtype, safety_checker=None)
    else:
        ddpm_pipe = StableDiffusionXLPipeline.from_pretrained(args.sd_model_id, unet=unet, torch_dtype=weight_dtype, safety_checker=None)
    ddpm_pipe.scheduler = DDIMScheduler.from_config(args.sd_model_id, subfolder="scheduler")
    ddpm_pipe.to("cuda")
    ddpm_pipe.load_lora_weights(args.lora_model_path)

    return ddim_pipe, ddpm_pipe

def load_feature_extractor(args: argparse.Namespace, device: torch.device) -> Tuple[Optional[torch.nn.Module], Optional[T.Compose]]:
    if args.feat_model_name:
        feat_model, feat_model_output_size = get_feat_model(args.feat_model_name)
        feat_model = feat_model.to(device)
        feat_model.eval()
        feat_model_transform = get_transform(args.feat_model_name)
        return feat_model, feat_model_transform
    return None, None

def load_embedding_translation(args: argparse.Namespace) -> Optional[CycleGAN]:
    if args.et_model_path:
        params = {'input_nc': 384, 'output_nc': 384}
        f2f_cyclegan = CycleGAN(params)
        f2f_cyclegan.load_state_dict(torch.load(args.et_model_path)["state_dict"])
        f2f_cyclegan.eval().cuda()
        return f2f_cyclegan
    return None

def process_image(image_path: str, args: argparse.Namespace, ddim_pipe: SDXLDDIMPipeline, ddpm_pipe: Union[SDXLImg2ImgPipeline, StableDiffusionXLPipeline], 
                  feat_model: Optional[torch.nn.Module], feat_model_transform: Optional[T.Compose], f2f_cyclegan: Optional[CycleGAN], 
                  device: torch.device, weight_dtype: torch.dtype, output_path: str, tensor_to_pil: T.ToPILImage) -> None:
    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        with open("error_images.txt", "a") as f:
            f.write(image_path + "\n")
        return

    resized_img = original_img.resize((args.target_resolution, args.target_resolution), resample=Image.BILINEAR)
    output_dir = os.path.join(output_path, str(args.target_resolution))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(image_path))
    
    if args.skip_existing and os.path.exists(output_file):
        print("Already exists", output_file)
        return

    x0_frozen_latent = torch.empty(0)
    x0_ffpe_latent = torch.empty(0)
    
    if feat_model and feat_model_transform:
        x0_frozen_latent = feat_model(feat_model_transform(resized_img).unsqueeze(0).to(device)).squeeze()
        if args.et_model_name == "cyclegan" and f2f_cyclegan:
            x0_ffpe_latent = f2f_cyclegan(x0_frozen_latent.unsqueeze(0)).squeeze()
        else:
            x0_ffpe_latent = torch.zeros_like(x0_frozen_latent)
    
    if args.et_weight != 1.0:
        x0_ffpe_latent = x0_ffpe_latent * args.et_weight + x0_frozen_latent * (1.0 - args.et_weight)
    
    # DDIM Inversion
    ddim_output = ddim_pipe(
        prompt="frozen tissue tile", image=resized_img, original_size=(args.target_resolution, args.target_resolution), target_size=(args.target_resolution, args.target_resolution),
        guided_latent=x0_frozen_latent.to(device, dtype=weight_dtype), denoising_end=(1 - args.high_noise_frac) if args.high_noise_frac is not None else None
    )

    # Denoising
    if args.high_noise_frac:
        out_image = ddpm_pipe(
            prompt="ffpe tissue tile", image=ddim_output[0].clone(), guidance_scale=args.guidance_scale, original_size=(args.target_resolution, args.target_resolution), 
            target_size=(args.target_resolution, args.target_resolution), guided_latent=x0_ffpe_latent.to(device, dtype=weight_dtype), 
            denoising_start=args.high_noise_frac, regularization=args.reg
        ).images[0]
    else:
        out_image = ddpm_pipe(
            prompt="ffpe tissue tile", latents=ddim_output[0].clone(), guidance_scale=args.guidance_scale, original_size=(args.target_resolution, args.target_resolution), 
            target_size=(args.target_resolution, args.target_resolution), guided_latent=x0_ffpe_latent.to(device, dtype=weight_dtype)
        ).images[0]

    out_image.save(output_file)

def main() -> None:
    args = make_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    tensor_to_pil = T.ToPILImage()
    
    output_path = setup_output_path(args)
    unet = load_unet(args, weight_dtype)
    ddim_pipe, ddpm_pipe = load_pipelines(args, unet, weight_dtype)
    feat_model, feat_model_transform = load_feature_extractor(args, device)
    f2f_cyclegan = load_embedding_translation(args)

    torch.manual_seed(42)
    image_paths = sorted(glob.glob(os.path.join(args.input_path, "*.png")))
    if args.max_samples:
        image_paths = image_paths[:args.max_samples]

    for image_path in tqdm(image_paths):
        process_image(image_path, args, ddim_pipe, ddpm_pipe, feat_model, 
                      feat_model_transform, f2f_cyclegan, device, weight_dtype, 
                      output_path, tensor_to_pil)

if __name__ == "__main__":
    main()
