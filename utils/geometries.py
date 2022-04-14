import open3d as o3d
import numpy as np


def coordinates_lineset(transform, length=1.0):
    line_points, line_idx = [], []
    x = transform[:3, 0]
    y = transform[:3, 1]
    z = transform[:3, 2]
    t = transform[:3, 3]
    line_points.append(t)
    line_points.append(t + length * x)
    line_points.append(t + length * y)
    line_points.append(t + length * z)
    line_idx.append([0, 1])
    line_idx.append([0, 2])
    line_idx.append([0, 3])

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_idx))

    return lineset


def coordinates_colorized_lineset(transform, length=1.0):
    line_points, line_idx = [], []
    x = transform[:3, 0]
    y = transform[:3, 1]
    z = transform[:3, 2]
    t = transform[:3, 3]
    line_points.append(t)
    line_points.append(t + length * x)
    line_points.append(t + length * y)
    line_points.append(t + length * z)
    line_idx.append([0, 1])
    line_idx.append([0, 2])
    line_idx.append([0, 3])

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_idx))

    lineset.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    return lineset


def points_to_lineset(points_start, points_end):
    line_points, line_idx = [], []
    for i, (p1, p2) in enumerate(zip(points_start, points_end)):
        line_points.append(p1)
        line_points.append(p2)
        line_idx.append([2 * i, 2 * i + 1])

    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_idx))


def points_to_colorized_pcl(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def points_to_pcl(points, color=(1.0, 0.0, 0.0)):
    colors = [color for _ in points]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
