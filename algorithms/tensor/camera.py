import torch
import numpy as np


def get_point_map(depth, model):
    m = torch.tensor(model, dtype=depth.dtype, device=depth.device)
    grid_x, grid_y = torch.meshgrid(torch.arange(0, depth.shape[2], dtype=depth.dtype),
                                    torch.arange(0, depth.shape[3], dtype=depth.dtype))

    uv = torch.stack([grid_y, grid_x, torch.ones_like(grid_x)])
    uv = uv.permute([1, 2, 0]).unsqueeze(0)
    rays = torch.einsum('ij, klmj->klmi', m, uv)
    return depth * rays.permute([0, 3, 1, 2])


def get_point_map_with_offcenter(depth, model, center=(0, 0)):
    m = torch.tensor(model, dtype=depth.dtype, device=depth.device)

    offset = center - torch.tensor([depth.shape[3], depth.shape[2]], device=depth.device) / 2
    offset = offset.view(-1, 1, 1, 2)
    offset = torch.cat([offset, torch.zeros(size=(depth.shape[0], 1, 1, 1), device=depth.device)], dim=-1)

    grid_x, grid_y = torch.meshgrid(torch.arange(0, depth.shape[2], dtype=depth.dtype),
                                    torch.arange(0, depth.shape[3], dtype=depth.dtype))

    uv = torch.stack([grid_y.to(depth.device), grid_x.to(depth.device), torch.ones_like(grid_x, device=depth.device)])
    uv = uv.permute([1, 2, 0]).unsqueeze(0).repeat(depth.shape[0], 1, 1, 1)
    uv_offset = uv + offset
    rays = torch.einsum('ij, klmj->klmi', m, uv_offset)
    return depth * rays.permute([0, 3, 1, 2])

def transfer_point_map(x, point_map):
    ### [n, 4, 4] [n, 3, h, w]
    shape = list(point_map.shape)
    shape[1] = 1
    point_map = torch.cat([point_map, torch.ones(shape, device=point_map.device)], dim=1)
    out = torch.einsum('kij, kjmn->kimn', x, point_map)
    return out[:, :3, ...]

# f_x = 525
# f_y = 525
# c_x = 960 / 2 - 0.5
# c_y = 540 / 2 - 0.5
# model = np.array([[1 / f_x, 0, -c_x / f_x], [0, 1 / f_y, -c_y / f_y], [0, 0, 1]])
# model_tensor = torch.tensor(model)
#
# depths = np.random.uniform(size=(10, 1, 5, 6))
# depth_tensor = torch.tensor(depths)
# steps = torch.tensor(np.linspace(0, depth_tensor.shape[2] - 1, depth_tensor.shape[2]))
# grid_x, grid_y = torch.meshgrid(torch.arange(0, depth_tensor.shape[2], dtype=depth_tensor.dtype), torch.arange(0, depth_tensor.shape[3], dtype=depth_tensor.dtype))
#
#
# uv = torch.stack([grid_y, grid_x, torch.ones_like(grid_x)])
# uv = uv.permute([1, 2, 0]).unsqueeze(0)
#
#
# pass

# depth = torch.einsum('ij,klm->...jm',model_tensor, uv)
