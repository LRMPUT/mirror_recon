import torch
import numpy as np
import numpy.linalg as la
from algorithms.camera import transfer_point_cloud, get_depth_with_offcenter, get_pcl_with_offcenter
from algorithms.tensor.camera import get_point_map_with_offcenter, transfer_point_map


def remove_closest_points(p1, p2, d=0.3, device='cpu'):
    p1_t = torch.tensor(p1).to(device)
    p2_t = torch.tensor(p2).to(device)

    vec_t = torch.unsqueeze(p1_t, 0) - torch.unsqueeze(p2_t, 1)
    dist = torch.sqrt(torch.sum(vec_t * vec_t, dim=2))
    min_dist, _ = torch.min(dist, dim=1)

    valid_idx = torch.squeeze(torch.nonzero(min_dist > d))
    p2_t_reduced = p2_t[valid_idx]
    return p2_t_reduced.cpu().numpy()


def remove_points_overlapping_depth(pcl, depth, camera_model, viewpoint, center, excerpt_resolution):
    pcl_in_viewpoint = transfer_point_cloud(pcl, la.inv(viewpoint))

    depth_from_viewpoint = get_depth_with_offcenter(pcl_in_viewpoint, camera_model,
                                                    excerpt_resolution,
                                                    center)

    depth_filtered = np.zeros(excerpt_resolution)

    h, w = np.shape(depth)[:2]

    for i in range(h):
        for j in range(w):
            if depth[i, j] == 0:
                depth_filtered[i, j] = depth_from_viewpoint[i, j]
            elif depth[i, j] > depth_from_viewpoint[i, j]:
                depth_filtered[i, j] = 0
            else:
                depth_filtered[i, j] = depth_from_viewpoint[i, j]

    points = get_pcl_with_offcenter(depth_filtered, camera_model, center)
    points = transfer_point_cloud(points, viewpoint)

    return points


def distance_map(x, camera_model, obj_tf, cam_tf, center):
    point_map = get_point_map_with_offcenter(x, camera_model, center)
    point_map = transfer_point_map(cam_tf, point_map)
    point_map = point_map.permute([0, 2, 3, 1])
    return torch.norm(point_map - obj_tf[:, :3, 3].view(-1, 1, 1, 3), dim=3)
