# Reference: https://github.com/lunit-io/benchmark-ssl-pathology/tree/main

import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms
from typing import Tuple

def get_pretrained_url(key: str) -> str:
    """
    Get the URL for the pretrained weights based on the given key.

    Args:
        key (str): Key to identify the model in the registry.

    Returns:
        str: URL for the pretrained weights.
    """
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch"
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    if model_zoo_registry.get(key) is None:
        raise ValueError(f"Invalid key: {key}. Available keys: {list(model_zoo_registry.keys())}")
    return pretrained_url

def vit_small(pretrained: bool, progress: bool, key: str, **kwargs) -> VisionTransformer:
    """
    Create a VisionTransformer model with the option to load pretrained weights.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        progress (bool): Whether to display a progress bar.
        key (str): Key to identify the model in the registry.
        **kwargs: Additional keyword arguments for the VisionTransformer.

    Returns:
        VisionTransformer: VisionTransformer model.
    """
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        model.load_state_dict(state_dict)
    return model

def get_transform(model_name: str) -> transforms.Compose:
    """
    Get the appropriate torchvision transform for the given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        transforms.Compose: Transformations to apply to the input images.
    """
    if model_name.lower() == "dino":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        raise NotImplementedError(f"model_name: {model_name} is not supported")

def get_feat_model(model_name: str) -> Tuple[VisionTransformer, int]:
    """
    Get the feature extraction model and its output feature size based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        Tuple[VisionTransformer, int]: The feature extraction model and its output feature size.
    """
    if model_name.lower() == "dino":
        model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        return model, 384
    else:
        raise NotImplementedError(f"model_name: {model_name} is not supported")
