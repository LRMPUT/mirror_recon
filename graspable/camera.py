import numpy as np


def kinect_model():
    f_x = 525
    f_y = 525
    c_x = 960 / 2 - 0.5
    c_y = 540 / 2 - 0.5
    model = np.array([[1 / f_x, 0, -c_x / f_x], [0, 1 / f_y, -c_y / f_y], [0, 0, 1]])

    return model


def graspable_model():
    f_x = 525
    f_y = 525
    c_x = 640 / 2 - 0.5
    c_y = 480 / 2 - 0.5
    model = np.array([[1 / f_x, 0, -c_x / f_x], [0, 1 / f_y, -c_y / f_y], [0, 0, 1]])

    return model


def get_point(u, v, z, model):
    px = np.array([u, v, 1])
    point = z * model @ px
    return point


def get_pcl(image, depth, model):
    points = []
    colors = []

    h, w, _ = np.shape(image)
    for i in range(h):
        for j in range(w):
            points.append(get_point(j, i, depth[i, j], model))
            colors.append(image[i, j])

    return points, colors


def get_point_map(depth, model):
    h, w = np.shape(depth)[:2]
    out = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            out[i, j] = get_point(j, i, depth[i, j], model)

    return out


def get_point_map_with_offcenter(depth, model, center=(0, 0)):
    h, w = np.shape(depth)[:2]
    out = np.zeros((h, w, 3))

    x_offset = center[1] - w // 2
    y_offset = center[0] - h // 2

    for i in range(h):
        for j in range(w):
            out[i, j] = get_point(j + x_offset, i + y_offset, depth[i, j], model)

    return out


def pcl_from_camera(points, camera=np.eye(4)):
    def _point_4v(point):
        out = np.ones(4)
        out[:3] = point
        return out

    points_in_camera = []

    for p in points:
        points_in_camera.append(np.matmul(camera, _point_4v(p))[:3])

    return points_in_camera
