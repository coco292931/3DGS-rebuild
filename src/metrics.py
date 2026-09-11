"""评估指标：PSNR 与 SSIM（纯 PyTorch，不引入额外依赖）。"""

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """[3,H,W] 或 [B,3,H,W] 均可。"""
    mse = F.mse_loss(pred, target)
    if mse.item() <= 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(max_val * max_val) / mse))


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device="cpu", dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """标准 SSIM（高斯窗，逐通道平均）。输入 [3,H,W]。"""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    c = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, c, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, window, groups=c)
    mu2 = F.conv2d(target, window, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, groups=c) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def d_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """官方 loss 里的 (1 - SSIM) 项。"""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    c = pred.shape[1]
    window = _gaussian_window(11, 1.5, c, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, groups=c)
    mu2 = F.conv2d(target, window, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, groups=c) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1.0 - ssim_map.mean()
