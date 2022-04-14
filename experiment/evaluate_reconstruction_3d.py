import torch
import numpy as np
import os
import re


def get_last_epoch(path):
    files = os.listdir(path)
    files.sort(key=natural_keys)

    if len(files) == 0:
        return None

    pattern = re.compile(r"model_(\d+)\.pt")

    files = list(filter(pattern.match, files))

    epoch = int(pattern.match(files[-1]).group(1))
    return epoch


### Util functions ####


def tensor_from_nch_image(image: np.array) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1)


def image_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).numpy()
    return img.astype(np.float32)


def natural_keys(text):
    def _atoi(txt):
        return int(txt) if txt.isdigit() else txt

    return [_atoi(c) for c in re.split(r'(\d+)', text)]


### 3d reconstruction ###

def get_pcl_with_offcenter(depth, model, center=(0, 0)):
    h, w = np.shape(depth)[:2]
    out = []
    x_offset = center[0] - w // 2
    y_offset = center[1] - h // 2

    for i in range(h):
        for j in range(w):
            out.append(get_point(j + x_offset, i + y_offset, depth[i, j], model))

    return out


def get_point(u, v, z, model):
    px = np.array([u, v, 1])
    point = z * model @ px
    return point


def transfer_point_cloud(points, camera=np.eye(4)):
    def _point_4v(point):
        out = np.ones(4)
        out[:3] = point
        return out

    points_in_camera = []

    for p in points:
        points_in_camera.append(np.matmul(camera, _point_4v(p))[:3])

    return points_in_camera


def graspable_model():
    f_x = 525
    f_y = 525
    c_x = 640 / 2 - 0.5
    c_y = 480 / 2 - 0.5
    model = np.array([[1 / f_x, 0, -c_x / f_x], [0, 1 / f_y, -c_y / f_y], [0, 0, 1]])

    return model


load_path = '/home/rafal/Models/ral_3d_gen/mug_mask_dropout_0.5_data_0.8'

last_epoch = get_last_epoch(load_path)
start_epoch = last_epoch if last_epoch is not None else 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load(os.path.join(load_path, 'model_{}.pth'.format(last_epoch)))
model.eval()
model = model.to(device)

#### Experiment #####

camera_model = graspable_model()
max_depth = 32000

# image input | every image has to have at least one channel
rgb = np.ones((96, 128, 3))
depth = np.ones((96, 128, 1))
proj = np.ones((96, 128, 1))

# convertion to tensor
rgb_tensor = tensor_from_nch_image(rgb) / 255.0
depth_tensor = tensor_from_nch_image(depth) / max_depth
proj_tensor = tensor_from_nch_image(proj) / max_depth

rgb_tensor = rgb_tensor.unsqueeze(0)
depth_tensor = depth_tensor.unsqueeze(0)
proj_tensor = proj_tensor.unsqueeze(0)

network_input = torch.cat([torch.mean(rgb_tensor, dim=1, keepdim=True), depth_tensor, proj_tensor], dim=1)
network_input = network_input.to(device)
network_input = network_input.float()

pred = model(network_input)

### point cloud ###
center = (200, 200)  # x,y center coordinaes
cam_t = np.eye(4)

pred = pred.detach().cpu()
output_depth = image_from_tensor(pred[0])  # depth image as numpy array - (h,w,1) format
pcl = get_pcl_with_offcenter(output_depth * max_depth / 5000, camera_model,
                             center)

global_pcl = transfer_point_cloud(pcl, cam_t) # pcl in global coordinate system given the camera transform
