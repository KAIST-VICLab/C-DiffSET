"""Standalone inference script for C-DiffSET.

Loads a fine-tuned U-Net checkpoint, translates every SAR ``.png`` under
``--sar-dir`` into an EO image, and (optionally) saves the confidence map
captured at the midpoint denoising step.

Example::

    python test.py \
        --sar-dir /path/to/test/SAR \
        --output-dir ./results/eo \
        --conf-dir  ./results/confidence \
        --checkpoint /path/to/model.safetensors
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


# ---------------------------------------------------------------------- #
# Architecture modifications (must match training)                       #
# ---------------------------------------------------------------------- #
def replace_unet_conv_in(unet):
    """Expand ``conv_in`` from 4 to 8 channels (duplicated weights x 0.5)."""
    _weight = unet.conv_in.weight.clone()               # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()                   # [320]
    _weight = _weight.repeat((1, 2, 1, 1)) * 0.5        # [320, 8, 3, 3]

    out_channels = unet.conv_in.out_channels
    new_conv_in = nn.Conv2d(8, out_channels, kernel_size=(3, 3),
                            stride=(1, 1), padding=(1, 1))
    new_conv_in.weight = nn.Parameter(_weight)
    new_conv_in.bias = nn.Parameter(_bias)
    unet.conv_in = new_conv_in
    if hasattr(unet, 'config'):
        unet.register_to_config(in_channels=8)
    return unet


def replace_unet_conv_out(unet, device, dtype):
    """Expand ``conv_out`` from 4 to 5 channels (noise 4 + variance 1)."""
    _weight = unet.conv_out.weight.clone()              # [4, 320, 3, 3]
    _bias = unet.conv_out.bias.clone()                  # [4]
    # Zero-initialize the extra variance channel.
    _weight = torch.cat(
        [_weight, torch.zeros((1, 320, 3, 3), device=device, dtype=dtype)], dim=0)  # [5, 320, 3, 3]
    _bias = torch.cat([_bias, torch.zeros((1,), device=device, dtype=dtype)], dim=0)  # [5]

    in_channels = unet.conv_out.in_channels
    new_conv_out = nn.Conv2d(in_channels, 5, kernel_size=(3, 3),
                             stride=(1, 1), padding=(1, 1))
    new_conv_out.weight = nn.Parameter(_weight)
    new_conv_out.bias = nn.Parameter(_bias)
    unet.conv_out = new_conv_out
    if hasattr(unet, 'config'):
        unet.register_to_config(out_channels=5)
    return unet


def load_custom_weights(model, checkpoint_path):
    """Load fine-tuned U-Net weights, stripping ``module.`` / ``unet.`` prefixes."""
    print(f"Loading custom UNet weights from {checkpoint_path}...")
    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

    filtered_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_key = new_key.replace('unet.', '') if new_key.startswith('unet.') else new_key
        filtered_dict[new_key] = v

    model.load_state_dict(filtered_dict, strict=False)
    print("UNet loaded successfully.")
    return model


def softmax_inverse(x):
    return math.log(math.exp(x) - 1)


def get_parser():
    parser = argparse.ArgumentParser(description='C-DiffSET inference')
    parser.add_argument('--sar-dir', required=True,
                        help='directory of input SAR .png images (searched recursively)')
    parser.add_argument('--output-dir', required=True,
                        help='directory to save generated EO images')
    parser.add_argument('--conf-dir', default=None,
                        help='optional directory to save confidence maps')
    parser.add_argument('--checkpoint', required=True,
                        help='path to the fine-tuned U-Net (.safetensors) or a folder with a "unet" subdir')
    parser.add_argument('--pretrained-model-name-or-path', type=str,
                        default='Manojb/stable-diffusion-2-1-base')
    parser.add_argument('--num-inference-steps', type=int, default=50)
    return parser


def main():
    args = get_parser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    if args.conf_dir:
        os.makedirs(args.conf_dir, exist_ok=True)

    # Variance parameters (identical to training).
    min_var, max_var, initial_var = 1e-6, 10, 1
    init_var_offset = softmax_inverse(initial_var - min_var)

    # ---- Base models ------------------------------------------------
    print("Loading base models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae").to(device, dtype=weight_dtype)

    inference_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    inference_scheduler.set_timesteps(args.num_inference_steps, device=device)

    # ---- U-Net: expand channels then load fine-tuned weights --------
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet").to(device, dtype=weight_dtype)
    unet = replace_unet_conv_in(unet)
    unet = replace_unet_conv_out(unet, device, weight_dtype)

    if os.path.isfile(args.checkpoint):
        unet = load_custom_weights(unet, args.checkpoint)
    else:
        unet_path = os.path.join(os.path.dirname(args.checkpoint), 'unet')
        if os.path.exists(unet_path):
            unet = UNet2DConditionModel.from_pretrained(unet_path).to(device, dtype=weight_dtype)

    text_encoder.eval()
    vae.eval()
    unet.eval()

    # Fixed text prompt embedding ("electro-optical image"), matching the paper.
    text_inputs = tokenizer(
        "electro-optical image", padding="do_not_pad", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embed = text_encoder(text_inputs.input_ids)[0].detach().clone()

    to_pil = transforms.ToPILImage()

    # ---- Collect input files ----------------------------------------
    file_list = []
    for root, _, files in os.walk(args.sar_dir):
        for fname in files:
            if fname.lower().endswith('.png'):
                file_list.append(os.path.join(root, fname))
    print(f"Total images to process: {len(file_list)}")

    # ---- Inference loop ---------------------------------------------
    with torch.no_grad():
        for file_path in tqdm(file_list, desc="Inference (Eps+Conf)"):
            rel_path = os.path.relpath(file_path, start=args.sar_dir)
            save_path_eo = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(save_path_eo), exist_ok=True)

            # (1) Load and normalize the SAR image to [-1, 1].
            img = np.array(Image.open(file_path).convert('RGB'))
            H, W, _ = img.shape
            img = img.reshape(*img.shape[:2], -1).transpose(2, 0, 1)  # (C, H, W)
            sar_tensor = torch.from_numpy(img).float() / 255.0 * 2 - 1
            sar_imgs = sar_tensor.unsqueeze(0).to(device, dtype=weight_dtype)

            # (2) Encode SAR to the VAE latent space.
            sar_latents = vae.encode(sar_imgs).latent_dist.mean * vae.config.scaling_factor

            # (3) Initialize the EO latent with Gaussian noise.
            eo_latents = torch.randn(sar_latents.shape, device=device, dtype=weight_dtype)
            timesteps = inference_scheduler.timesteps
            var_map = None

            # (4) DDIM denoising loop.
            for i, t in enumerate(timesteps):
                unet_input = torch.cat([sar_latents, eo_latents], dim=1)  # 8-channel input
                noise_pred = unet(unet_input, t, encoder_hidden_states=text_embed).sample
                eo_latents = inference_scheduler.step(noise_pred[:, :4], t, eo_latents).prev_sample
                if i == len(timesteps) // 2:
                    var_map = noise_pred[:, -1].unsqueeze(1)

            # (5) Decode and save the generated EO image.
            eo_generated = vae.decode(eo_latents / vae.config.scaling_factor).sample
            eo_generated = (eo_generated * 0.5 + 0.5).squeeze(0).cpu().clamp(0, 1)
            to_pil(eo_generated).save(save_path_eo)

            # (6) Optional: post-process and save the confidence map.
            if args.conf_dir and var_map is not None:
                save_path_conf = os.path.join(args.conf_dir, rel_path)
                os.makedirs(os.path.dirname(save_path_conf), exist_ok=True)

                # Recover the variance, then confidence = 1 / variance.
                var = F.softplus(var_map + init_var_offset) + min_var
                var = torch.clamp(var, min_var, max_var)
                conf = 1.0 / var

                # Min-max normalize to [-1, 1] for visualization.
                conf_min = conf.amin(dim=(1, 2, 3), keepdim=True)
                conf_max = conf.amax(dim=(1, 2, 3), keepdim=True)
                conf = (conf - conf_min) / (conf_max - conf_min + 1e-8) * 2.0 - 1.0

                conf = F.interpolate(conf, size=(H, W), mode='bicubic').repeat(1, 3, 1, 1)
                conf_image = (conf * 0.5 + 0.5).squeeze(0).cpu().clamp(0, 1)
                to_pil(conf_image).save(save_path_conf)

    print(f"Inference complete. EO images saved in: {args.output_dir}")
    if args.conf_dir:
        print(f"Confidence maps saved in: {args.conf_dir}")


if __name__ == '__main__':
    main()
