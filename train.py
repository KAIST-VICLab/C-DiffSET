"""Training / validation loop for C-DiffSET.

The :class:`Trainer` fine-tunes a pretrained Stable Diffusion U-Net for
SAR-to-EO translation. Two architectural modifications are made to the
pretrained U-Net (see paper, Sec. III-C):

    * ``conv_in``  : 4 -> 8 channels, to concatenate the SAR latent with the
      noisy EO latent (duplicated weights scaled by 0.5).
    * ``conv_out`` : 4 -> 5 channels, where the extra channel predicts the raw
      spatial variance used by the confidence-guided diffusion (C-Diff) loss
      (its weights are zero-initialized for stable optimization).
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from utils import ImageQualityMetrics, Train_Report, Test_Report


def softmax_inverse(x):
    """Inverse of SoftPlus at ``x``; used to offset the initial variance."""
    return math.log(math.exp(x) - 1)


class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.test_data_loader = data_loader['test']

        # ---- Accelerator ------------------------------------------------
        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            project_config=self.accelerator_project_config,
        )
        if self.accelerator.is_main_process and args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)

        # Resolve weight dtype from the mixed-precision setting.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # ---- Model ------------------------------------------------------
        # 1) Load frozen VAE / text encoder + the pretrained U-Net.
        self.load_checkpoint()

        # 2) Expand conv_in to 8 channels (SAR latent | noisy EO latent).
        if self.unet.config["in_channels"] == 4:
            self.replace_unet_conv_in()

        # 3) Load pretrained (SAR-conditioned) U-Net weights.
        state_dict = load_file(self.args.accelerator_path)
        self.unet.load_state_dict(state_dict)

        # 4) Prepare text conditioning and confidence hyper-parameters.
        self.init_text_prompt()
        self.init_confidence()

        # 5) Expand conv_out to 5 channels (noise prediction | raw variance).
        if self.unet.config["out_channels"] == 4:
            self.replace_unet_conv_out()

        # ---- Optimizer / LR scheduler ----------------------------------
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup,
            num_training_steps=self.args.num_iter,
        )

        # ---- Metrics / inference scheduler -----------------------------
        self.metrics = ImageQualityMetrics(device=self.accelerator.device)
        self.num_inference_steps = self.args.num_inference_steps
        self.inference_scheduler.set_timesteps(self.num_inference_steps,
                                               device=self.accelerator.device)

        (self.unet, self.optimizer, self.lr_scheduler,
         self.train_data_loader, self.test_data_loader) = self.accelerator.prepare(
            self.unet, self.optimizer, self.lr_scheduler,
            self.train_data_loader, self.test_data_loader,
        )

    # ------------------------------------------------------------------ #
    # Model construction                                                 #
    # ------------------------------------------------------------------ #
    def load_checkpoint(self):
        """Load the frozen VAE / CLIP text encoder and the pretrained U-Net."""
        noise_scheduler_config = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler").config
        noise_scheduler_config['prediction_type'] = self.args.prediction_type
        self.noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)
        self.inference_scheduler = DDIMScheduler.from_config(noise_scheduler_config)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="unet")

        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)

        # Only the U-Net is trained.
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

    def init_text_prompt(self):
        """Pre-compute the fixed CLIP embedding used as text conditioning.

        Following the paper (Sec. III-C), the generic prompt
        "electro-optical image" is used as a stable semantic anchor, rather
        than a null prompt.
        """
        prompt = "electro-optical image"
        text_inputs = self.tokenizer(
            prompt, padding="do_not_pad", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt").to(self.accelerator.device)
        self.text_embed = self.text_encoder(
            text_inputs.input_ids)[0].detach().clone().to(self.accelerator.device)

    def replace_unet_conv_in(self):
        """Expand ``conv_in`` from 4 to 8 input channels.

        The original weights are duplicated across the two latent groups and
        scaled by 0.5 so that the activation magnitude is preserved.
        """
        _weight = self.unet.conv_in.weight.clone()      # [320, 4, 3, 3]
        _bias = self.unet.conv_in.bias.clone()          # [320]
        _weight = _weight.repeat((1, 2, 1, 1)) * 0.5    # [320, 8, 3, 3]

        out_channels = self.unet.conv_in.out_channels
        new_conv_in = nn.Conv2d(8, out_channels, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1))
        new_conv_in.weight = nn.Parameter(_weight)
        new_conv_in.bias = nn.Parameter(_bias)
        self.unet.conv_in = new_conv_in
        self.unet.config["in_channels"] = 8

    def replace_unet_conv_out(self):
        """Expand ``conv_out`` from 4 to 5 output channels.

        The 5th channel predicts the raw spatial variance; its weights/bias are
        zero-initialized so training starts from the standard L2 objective.
        """
        _weight = self.unet.conv_out.weight.clone()     # [4, 320, 3, 3]
        _bias = self.unet.conv_out.bias.clone()         # [4]
        _weight = torch.cat(
            [_weight, torch.zeros((1, 320, 3, 3), device=self.accelerator.device,
                                  dtype=_weight.dtype)], dim=0)   # [5, 320, 3, 3]
        _bias = torch.cat(
            [_bias, torch.zeros((1,), device=self.accelerator.device,
                                dtype=_weight.dtype)], dim=0)     # [5]

        in_channels = self.unet.conv_out.in_channels
        new_conv_out = nn.Conv2d(in_channels, 5, kernel_size=(3, 3),
                                 stride=(1, 1), padding=(1, 1))
        new_conv_out.weight = nn.Parameter(_weight)
        new_conv_out.bias = nn.Parameter(_bias)
        self.unet.conv_out = new_conv_out
        self.unet.config["out_channels"] = 5

    def init_confidence(self):
        """Initialize variance bounds and the SoftPlus offset (initial var = 1)."""
        self.min_var = 1e-6
        self.max_var = 10
        initial_var = 1
        self.init_var_offset = softmax_inverse(initial_var - self.min_var)

    # ------------------------------------------------------------------ #
    # Checkpoint / visualization helpers                                 #
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch):
        self.accelerator.save_state(os.path.join(self.args.work_dir, f"epoch-{epoch}"))

    def save_best_model(self):
        self.accelerator.save_state(os.path.join(self.args.work_dir, 'best'))

    def save_image(self, sar, gen_eo, eo, var, idx):
        """Save a 2x2 grid: (SAR | EO) over (variance | generated EO)."""
        path = os.path.join(self.args.work_dir, 'results')
        os.makedirs(path, exist_ok=True)

        gt_image = torch.cat((sar, eo), dim=-1)
        generated_image = torch.cat((var, gen_eo), dim=-1)
        generated_image = torch.cat((gt_image, generated_image), dim=-2) * 0.5 + 0.5
        generated_image = generated_image.squeeze(0).detach().cpu().clamp(0, 1)
        transforms.ToPILImage()(generated_image).save(f'{path}/{idx:04}.tif')

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #
    def train(self, train_log, global_step):
        self.unet.requires_grad_(True)
        report = Train_Report()
        start = time.time()

        for idx, (sar_imgs, eo_imgs) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.unet):
                # ---- Encode images to the shared latent space -----------
                with torch.no_grad():
                    sar_imgs = sar_imgs.to(self.accelerator.device, dtype=self.weight_dtype)
                    eo_imgs = eo_imgs.to(self.accelerator.device, dtype=self.weight_dtype)
                    sar_latents = self.vae.encode(sar_imgs).latent_dist.mean * self.vae.config.scaling_factor
                    gt_eo_latents = self.vae.encode(eo_imgs).latent_dist.mean * self.vae.config.scaling_factor
                    text_embed = self.text_embed.repeat((self.args.batch_size, 1, 1))

                    # Forward diffusion: add noise to the EO latent.
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (self.args.batch_size,), device=gt_eo_latents.device).long()
                    noise = torch.randn(gt_eo_latents.shape, device=gt_eo_latents.device)
                    noisy_latents = self.noise_scheduler.add_noise(gt_eo_latents, noise, timesteps)

                    alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
                        self.accelerator.device, dtype=self.weight_dtype)
                    sqrt_alpha_prod = (alphas_cumprod[timesteps] ** 0.5).flatten()

                # Condition on the SAR latent via channel concatenation.
                cat_latents = torch.cat([sar_latents, noisy_latents], dim=1)  # [B, 8, h, w]

                if self.args.prediction_type == "sample":
                    target = gt_eo_latents
                elif self.args.prediction_type == "epsilon":
                    target = noise
                else:
                    raise ValueError(f"Unknown prediction type {self.args.prediction_type}")

                # ---- Predict noise + raw variance -----------------------
                model_pred = self.unet(cat_latents, timesteps, text_embed).sample
                noise_pred = model_pred[:, :-1]
                var = model_pred[:, -1].unsqueeze(1)
                var = F.softplus(var + self.init_var_offset) + self.min_var
                var = torch.clamp(var, self.min_var, self.max_var)

                # ---- Confidence-guided diffusion (C-Diff) loss ----------
                # Gaussian NLL with a timestep-dependent beta-NLL weight
                # (var.detach() ** sqrt(alpha_bar_t)); see paper, Sec. III-C.
                var = var.expand_as(target)
                l1 = -0.5 * ((target - noise_pred) ** 2 / var
                             + torch.log(var) + math.log(2 * math.pi))
                weight = var.detach() ** sqrt_alpha_prod.view(-1, 1, 1, 1)
                l1 = l1 * weight
                loss = -torch.mean(l1)
                reduced_loss = self.accelerator.gather(loss).mean()

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size, reduced_loss.item())

            global_step += 1

            # ---- Logging / checkpointing --------------------------------
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t'
                                + report.result_str(lr, period_time))
                start = time.time()
                report.__init__()

            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(
                    os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))

            if global_step >= self.args.num_iter:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t'
                                + report.result_str(lr, period_time))
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'lastest'))
                self.accelerator.end_training()
                return global_step

        return global_step

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    def val(self, test_log, epoch, val_len=100):
        self.unet.requires_grad_(False)
        report = Test_Report()

        for idx, (sar_imgs, eo_imgs) in tqdm(enumerate(self.test_data_loader)):
            if idx >= val_len:
                break
            with torch.no_grad():
                B, C, H, W = sar_imgs.shape
                sar_imgs = sar_imgs.to(self.accelerator.device, dtype=self.weight_dtype)
                sar_latents = self.vae.encode(sar_imgs).latent_dist.mean * self.vae.config.scaling_factor
                eo_imgs = eo_imgs.to(self.accelerator.device, dtype=self.weight_dtype)
                eo_latents = torch.randn(sar_latents.shape, device=sar_latents.device)
                text_embed = self.text_embed.repeat((self.args.test_batch_size, 1, 1))

                # ---- DDIM reverse process ------------------------------
                timesteps = self.inference_scheduler.timesteps
                for i, t in enumerate(timesteps):
                    unet_input = torch.cat([sar_latents, eo_latents], dim=1)
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=text_embed).sample
                    eo_latents = self.inference_scheduler.step(
                        noise_pred[:, :4], t, eo_latents).prev_sample

                    # Capture the confidence/variance map at the midpoint step.
                    if i == len(timesteps) // 2:
                        var = noise_pred[:, -1].unsqueeze(1)

                # ---- Post-process the variance map for visualization ----
                var = F.softplus(var + self.init_var_offset) + self.min_var
                var = torch.clamp(var, self.min_var, self.max_var)
                var = (var - var.amin(dim=(1, 2, 3))) / \
                      (var.amax(dim=(1, 2, 3)) - var.amin(dim=(1, 2, 3))) * 2.0 - 1.0
                var = F.interpolate(var, size=(H, W), mode='bicubic')
                var = var.repeat(1, 3, 1, 1)

                eo_generated = self.vae.decode(eo_latents / self.vae.config.scaling_factor).sample
                metrics = self.metrics.calculate_metrics(eo_imgs, eo_generated)
                report.update(self.args.test_batch_size, metrics)
                self.save_image(sar_imgs, eo_generated, eo_imgs, var, idx)

        test_log.write(f'Epoch[{epoch}]\t' + report.result_str())
        return report.psnr
