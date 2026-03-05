from typing import Tuple, Union

import torch

from otter.models.distributions import Delta, Gaussian

DistributionWithMean = Union[Delta, Gaussian]


def _power_spectrum_density_mse(
    pred: torch.Tensor, trg: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 1D PSD for a batch of 2D tensors.
    """
    assert pred.ndim == 3, "Input must be a batch of 2D images: [B, H, W]"
    assert trg.ndim == 3, "Input must be a batch of 2D images: [B, H, W]"
    B, H, W = pred.shape
    device = pred.device

    # 2D FFT and power spectrum
    fft2_pred = torch.fft.fft2(pred)
    fft2_shifted_pred = torch.fft.fftshift(fft2_pred, dim=(-2, -1))
    psd_pred = torch.abs(fft2_shifted_pred) ** 2 / (H * W)

    # 2D FFT and power spectrum
    fft2_trg = torch.fft.fft2(trg)
    fft2_shifted_trg = torch.fft.fftshift(fft2_trg, dim=(-2, -1))
    psd_trg = torch.abs(fft2_shifted_trg) ** 2 / (H * W)

    psd_mse = (psd_pred - psd_trg) ** 2

    # Create frequency radius map
    y = torch.arange(H, device=device).reshape(-1, 1)
    x = torch.arange(W, device=device).reshape(1, -1)
    cy, cx = H // 2, W // 2
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.round().long()
    r_max = int(r.max().item())

    # Flatten the radius map
    r_flat = r.flatten().unsqueeze(0).repeat(B, 1)
    psd_mse_flat = psd_mse.reshape(B, -1)
    psd_trg_flat = psd_trg.reshape(B, -1)

    psd_1d = torch.zeros(B, r_max + 1, dtype=torch.float32, device=device)
    psd_trg_1d = torch.zeros(B, r_max + 1, dtype=torch.float32, device=device)

    counts = torch.zeros(B, r_max + 1, dtype=torch.float32, device=device)

    counts.scatter_add_(
        dim=1,
        index=r_flat,
        src=torch.ones_like(psd_mse_flat, dtype=torch.float32, device=device),
    )
    counts = counts.clamp(min=1)

    psd_1d.scatter_add_(dim=1, index=r_flat, src=psd_mse_flat)

    psd_1d /= counts

    psd_trg_1d.scatter_add_(
        dim=1,
        index=r_flat,
        src=psd_trg_flat,
    )
    psd_trg_1d /= counts

    return psd_1d, psd_trg_1d


def _batch_tensor(input: torch.Tensor) -> torch.Tensor:
    """
    Batch a tensor along the first dimension.
    """

    assert input.ndim == 5
    B, lat, lon, T, V = input.shape

    # Step 1: move lat/lon to end temporarily
    input = input.permute(0, 3, 4, 1, 2)  # [B, T, V, lat, lon]

    # Step 2: reshape everything except lat/lon into one batch dimension
    new_batch = B * T * V
    input = input.reshape(new_batch, lat, lon)

    return input


def power_spectrum_density_mse(
    trg: torch.Tensor,  # (batch, lat, lon, time_idx, var_and_level)
    prd_dist: DistributionWithMean,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = prd_dist.mean  # (batch, lat, lon, time_idx, var_and_level)

    B, T, V = pred.shape[0], pred.shape[3], pred.shape[4]

    pred_batched = _batch_tensor(pred)  # (batch * time * var, lat, lon)
    trg_batched = _batch_tensor(trg)  # (batch * time * var, lat, lon)

    psd_mse, psd_trg = _power_spectrum_density_mse(pred_batched, trg_batched)

    psd_mse = psd_mse.reshape(B, T, V, -1).mean(dim=0)
    psd_trg = psd_trg.reshape(B, T, V, -1).mean(dim=0)

    return psd_mse, psd_trg
