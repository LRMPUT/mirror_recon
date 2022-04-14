import numpy as np
import numpy.linalg as la


## fx = 1062.3 fy=1062.3
## res 1280 960
##focal axis 639.5 479.5

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


def get_pixel(point, model):
    p = point / point[2]
    pixel = np.matmul(la.inv(model), p)
    return pixel[:2]


def get_pcl(image, depth, model):
    points = []
    colors = []

    h, w, _ = np.shape(image)
    for i in range(h):
        for j in range(w):
            if depth[i, j] != 0:
                points.append(get_point(j, i, depth[i, j], model))
                colors.append(image[i, j])

    return points, colors


def get_pcl_with_offcenter(depth, model, center=(0, 0)):
    h, w = np.shape(depth)[:2]
    out = []
    x_offset = center[0] - w // 2
    y_offset = center[1] - h // 2

    for i in range(h):
        for j in range(w):
            out.append(get_point(j + x_offset, i + y_offset, depth[i, j], model))

    return out


def get_depth(pcl, model, resolution):
    image = np.zeros(resolution)
    for p in pcl:
        pixel = get_pixel(p, model)
        if 0 <= pixel[0] < resolution[0] and 0 <= pixel[1] < resolution[1]:
            image[pixel[0], pixel[1]] = p[2]

    return image


def get_depth_with_offcenter(pcl, model, excerpt_resolution, center):
    image = np.zeros(excerpt_resolution)
    x_offset = center[0] - excerpt_resolution[1] // 2
    y_offset = center[1] - excerpt_resolution[0] // 2

    for p in pcl:
        pixel = get_pixel(p, model)
        shifted_pixel = np.pad(pixel, [0, 1]) - np.pad([x_offset, y_offset], [0, 1])
        if 0 <= int(round(shifted_pixel[1])) < excerpt_resolution[0] and 0 <= int(round(shifted_pixel[0])) < \
                excerpt_resolution[1]:
            image[int(round(shifted_pixel[1])), int(round(shifted_pixel[0]))] = p[2]

    return image


# def get_depth_from_pcl(points, model, resolution):
#     out = np.zeros(resolution)
#     for (x, y, z) in points:
#         p = np.array([x/z, y/z, 1])
#         pixel = np.linalg.inv(model)*p
#         if 0 <= int(pixel[0]) < resolution[0] and 0 <= int(pixel[1]) < resolution[1]:
#             out[int(pixel[0]), int(pixel[1])] = pixel[2]
#
#     return out


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

    x_offset = center[0] - w // 2
    y_offset = center[1] - h // 2

    for i in range(h):
        for j in range(w):
            out[i, j] = get_point(j + x_offset, i + y_offset, depth[i, j], model)

    return out


def transfer_point_cloud(points, camera=np.eye(4)):
    def _point_4v(point):
        out = np.ones(4)
        out[:3] = point
        return out

    points_in_camera = []

    for p in points:
        points_in_camera.append(np.matmul(camera, _point_4v(p))[:3])

    return points_in_camera
