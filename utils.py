"""Evaluation metrics and lightweight logging/reporting utilities."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

import lpips
from pytorch_fid import fid_score
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    SpatialCorrelationCoefficient,
)
from torchmetrics.image.inception import InceptionScore


def write(log, string):
    """Flush stdout and append a line to a log file handle."""
    sys.stdout.flush()
    log.write(string + '\n')
    log.flush()


class ImageQualityMetrics:
    """Wraps the image-quality metrics reported in the paper.

    Note:
        ``denormalize`` maps model outputs to [0, 1] with ``img + 0.5``. This is
        the exact transform used to produce the reported numbers; change it only
        if you also re-generate all baselines for a fair comparison.
    """

    def __init__(self, device='cuda', fid_real_images_path=None):
        self.device = device

        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.scc = SpatialCorrelationCoefficient().to(device)
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.inception_score_fn = InceptionScore().to(device)

        # FID requires a directory of real images.
        self.fid_real_images_path = fid_real_images_path

    def denormalize(self, img):
        """Map an image to the [0, 1] range used for metric computation."""
        return (img + 0.5).clamp(0, 1)

    def denormalize_and_convert(self, img):
        """Denormalize and convert to uint8 (required by Inception Score)."""
        img = self.denormalize(img)
        return (img * 255).to(torch.uint8)

    def calculate_psnr(self, real_images, generated_images):
        return self.psnr(self.denormalize(generated_images), self.denormalize(real_images))

    def calculate_ssim(self, real_images, generated_images):
        return self.ssim(self.denormalize(generated_images), self.denormalize(real_images))

    def calculate_scc(self, real_images, generated_images):
        return self.scc(self.denormalize(generated_images), self.denormalize(real_images))

    def calculate_lpips(self, real_images, generated_images):
        return self.lpips_fn(
            self.denormalize(generated_images), self.denormalize(real_images)
        ).mean()

    def calculate_fid(self, generated_images_path):
        if not self.fid_real_images_path:
            raise ValueError("Path to real images for FID calculation not provided.")
        return fid_score.calculate_fid_given_paths(
            [self.fid_real_images_path, generated_images_path],
            batch_size=50, device=self.device, dims=2048,
        )

    def calculate_inception_score(self, generated_images):
        gen_images_uint8 = self.denormalize_and_convert(generated_images)
        return self.inception_score_fn(gen_images_uint8)

    def calculate_metrics(self, real_images, generated_images, generated_images_path=None):
        """Compute PSNR/SSIM/SCC/LPIPS (+ optional IS and FID) for a batch."""
        metrics = {
            'psnr': self.calculate_psnr(real_images, generated_images).item(),
            'ssim': self.calculate_ssim(real_images, generated_images).item(),
            'scc': self.calculate_scc(real_images, generated_images).item(),
            'lpips': self.calculate_lpips(real_images, generated_images).item(),
        }

        # Inception Score is only meaningful for a sufficiently large batch.
        if generated_images.shape[0] > 10:
            is_mean, is_std = self.calculate_inception_score(generated_images)
            metrics['is_mean'] = is_mean.item()
            metrics['is_std'] = is_std.item()
        else:
            metrics['is_mean'] = 0
            metrics['is_std'] = 0

        if generated_images_path:
            metrics['fid'] = self.calculate_fid(generated_images_path).item()
        else:
            metrics['fid'] = 0

        return metrics


class Report:
    """Append-mode text logger that also echoes to stdout."""

    def __init__(self, save_dir, type):
        filename = os.path.join(save_dir, f'{type}_log.txt')
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        mode = 'a' if os.path.exists(filename) else 'w'
        self.logFile = open(filename, mode)

    def write(self, string):
        print(string)
        write(self.logFile, string)

    def __del__(self):
        self.logFile.close()


class Train_Report:
    """Accumulates the running training loss over a logging interval."""

    def __init__(self):
        self.total_loss = []
        self.num_examples = 0

    def update(self, batch_size, total_loss):
        self.num_examples += batch_size
        self.total_loss.append(total_loss * batch_size)

    def compute_mean(self):
        self.total_loss = np.sum(self.total_loss) / self.num_examples

    def result_str(self, lr, period_time):
        self.compute_mean()
        return (f'Total Loss: {self.total_loss:.6f}\t'
                f'learning rate: {lr:.7f}\tTime: {period_time:.4f}')


class Test_Report:
    """Accumulates per-batch metrics and reports their means."""

    def __init__(self):
        self.psnr = []
        self.ssim = []
        self.scc = []
        self.lpips = []
        self.fid = []
        self.is_mean = []
        self.is_std = []
        self.num_examples = 0

    def update(self, batch_size, metrics):
        self.num_examples += batch_size
        self.psnr.append(metrics['psnr'])
        self.ssim.append(metrics['ssim'])
        self.scc.append(metrics['scc'])
        self.lpips.append(metrics['lpips'])
        self.fid.append(metrics['fid'])
        self.is_mean.append(metrics['is_mean'])
        self.is_std.append(metrics['is_std'])

    def compute_mean(self):
        self.psnr = np.sum(self.psnr) / self.num_examples
        self.ssim = np.sum(self.ssim) / self.num_examples
        self.scc = np.sum(self.scc) / self.num_examples
        self.lpips = np.sum(self.lpips) / self.num_examples
        self.fid = np.sum(self.fid) / self.num_examples
        self.is_mean = np.sum(self.is_mean) / self.num_examples
        self.is_std = np.sum(self.is_std) / self.num_examples

    def result_str(self):
        self.compute_mean()
        return (f'PSNR: {self.psnr:.6f}\tSSIM: {self.ssim:.6f}\t'
                f'SCC: {self.scc:.6f}\tLPIPS: {self.lpips:.6f}\t'
                f'FID: {self.fid:.6f}\tIS-mean: {self.is_mean:.6f}\t'
                f'IS-std: {self.is_std:.6f}')
