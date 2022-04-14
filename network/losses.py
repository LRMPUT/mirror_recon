import torch
import numpy as np


def point_distance_loss(x_distance_map, y_distance_map, criterion=torch.nn.L1Loss()):
    min_val = torch.amin(y_distance_map, dim=(1, 2)).view(-1, 1, 1)
    max_val = torch.amax(y_distance_map, dim=(1, 2)).view(-1, 1, 1)
    weights = (y_distance_map - min_val) / (
            max_val - min_val)

    weights = torch.clip(weights, 0.0, 2 * np.arccos(0.4) / np.pi)
    weights = torch.cos(np.pi / 2 * weights)

    loss = criterion(x_distance_map * weights, y_distance_map * weights)
    return loss


def mask_loss(x_distance_map, y_distance_map, mask, l1=0.99, l2=0.01, criterion=torch.nn.L1Loss()):
    loss = criterion(x_distance_map * mask, y_distance_map * mask) * l1 \
           + criterion(x_distance_map * (1 - mask), y_distance_map * (1 - mask)) * l2
    return loss
