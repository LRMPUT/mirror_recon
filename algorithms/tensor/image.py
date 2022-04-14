import torch
import torch.nn.functional as F
import numpy as np


def kernel_custom_noise(size, kernels=(np.ones((1, 1)), np.ones((1, 3)) / 3, np.ones((1, 5)) / 55),
                        rates=(0.01, 0.02, 0.01, 0.0001), device='cpu'):
    masks = []

    for i, (k, r) in enumerate(zip(kernels, rates)):
        noise = torch.rand(size, device=device)
        bound = noise < r
        h, w = np.shape(k)
        weights = torch.tensor(k, dtype=torch.float32, device=device).view(1, 1, h, w)

        mask = torch.where(bound, torch.ones_like(noise, device=device),
                           torch.zeros_like(noise, device=device))

        dialted_mask = F.conv2d(mask, weights, padding='same')

        masks.append(dialted_mask.sgn())
    masks = torch.cat(masks, 0)
    masks = torch.sum(masks, dim=0, keepdim=True).clamp(0, 1)

    return masks


def apply_gradient_mask(x, mask, threshold=0.5):
    gradient = (torch.sin(torch.linspace(0, np.pi / 2, x.shape[2], device=x.device)) * torch.rand(x.shape[2],
                                                                                                  device=x.device)).view(
        -1, 1)

    filtered_mask = mask * gradient
    filtered_mask = 1 - (filtered_mask - threshold).clamp(0, 1).sgn()
    filtered_mask = filtered_mask.squeeze(dim=0)

    return x * filtered_mask


def drop(x: torch.Tensor, noise: torch.Tensor, rate):
    sparse = torch.where(noise <= rate, torch.ones(x.shape, device=x.device),
                         torch.zeros(x.shape, device=x.device))
    return x * sparse


def select_mask(x: torch.Tensor, index: torch.Tensor):
    index = index.view(-1, 1, 1, 1)
    return torch.where(x == index, torch.ones(x.shape, device=x.device),
                       torch.zeros(x.shape, device=x.device))


def scale_and_clip(x: torch.Tensor, s, min_val=0.0, max_val=1.0):
    return torch.clip(x * s, min_val, max_val)
