from piqa import ssim
from torch import nn
from tqdm import tqdm

from algorithms.tensor.image import scale_and_clip
from network.data import *
from utils.misc import *
from utils.geometries import *
from algorithms.tensor.point_cloud import remove_points_overlapping_depth
from imageio import imwrite


def dictify_input(x):
    data = {}
    data['rgb_obj'] = x[:, :3, ...]
    data['depth_obj'] = x[:, 3:4, ...]
    data['depth_obj_proj'] = x[:, 4:5, ...]
    return data


def dictify_output(x):
    data = {}
    data['depth_obj_back'] = x[:, :1, ...]
    return data


def dictify_prediction(x):
    data = {}
    data['depth_obj_pred'] = x[:, :1, ...]
    return data


def get_last_epoch(path):
    files = os.listdir(path)
    files.sort(key=natural_keys)

    if len(files) == 0:
        return None

    pattern = re.compile(r"model_(\d+)\.pt")

    files = list(filter(pattern.match, files))

    epoch = int(pattern.match(files[-1]).group(1))
    return epoch


def input_type_to_idx(x):
    def _chr_to_idx(c):
        selector = {'r': 0,
                    'g': 1,
                    'b': 2,
                    'd': 3,
                    'p': 4}
        return selector[c]

    return [_chr_to_idx(c) for c in x]


def append_to_dict(x, k, v):
    if x.get(k):
        x[k].append(v)
    else:
        x[k] = [v]


path = '/home/rafal/Datasets/shapeNetGraspable_30instances/shapeNetGraspable_30instances_bottle'
load_path = '/home/rafal/Models/OtherSide/shapeNetGraspable_30instances_unet_dp'

last_epoch = get_last_epoch(load_path)
start_epoch = last_epoch if last_epoch is not None else 0

_, _, test = get_sparse_multi_segmentation_multi_class_dataset(path, 0.8, 0.1, 0.1, seed=42)
log_step = lambda x: x % 20 == 0

test_laoder = DataLoader(dataset=test, batch_size=1, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load(os.path.join(load_path, 'model_{}.pth'.format(last_epoch)))
model.eval()
model = model.to(device)

criterion = nn.L1Loss()
ssim_criterion = ssim.MS_SSIM(n_channels=1, window_size=4).cuda() if torch.cuda.is_available() else ssim.MS_SSIM(
    n_channels=1,
    window_size=4)

test_step = 0

#### Experiment #####
input_type = 'dp'

camera_model = graspable_model()

resolution = (480, 640)
excerpt_resolution = (96, 128)

visualize = []
data = {}

### VIsualization #####
vis = o3d.visualization.Visualizer()

for i, d in enumerate(tqdm(test_laoder)):
    x, y, seg, obj, obj_back, cam, cam_back = d

    inputs = dictify_input(x)
    targets = dictify_output(y)

    network_input = x[:, input_type_to_idx(input_type), ...]
    network_output = y

    transforms = []
    pcls = []

    obj_t = obj[3][0].numpy()
    obj_back_t = obj_back[3][0].numpy()
    cam_t = cam[0].numpy()
    cam_back_t = cam_back[0].numpy()

    network_input = network_input.to(device)
    network_output = network_output.to(device)

    pred = model(network_input)
    predictions = dictify_prediction(pred.cpu().detach())

    append_to_dict(data, 'rgb', image_from_tensor(inputs['rgb_obj'][0]))
    append_to_dict(data, 'depth', image_from_tensor(inputs['depth_obj'][0]))
    append_to_dict(data, 'depth_proj', image_from_tensor(inputs['depth_obj_proj'][0]))
    append_to_dict(data, 'depth_back', image_from_tensor(targets['depth_obj_back'][0]))
    append_to_dict(data, 'depth_pred', image_from_tensor(predictions['depth_obj_pred'][0]))

    center_front, center_back = obj[2][0], obj_back[2][0]

    pcl_front_in_cam = get_pcl_with_offcenter(data['depth'][i] * test.max_depth / 5000, camera_model,
                                              center_front)
    pcl_back_in_cam = get_pcl_with_offcenter(data['depth_pred'][i] * test.max_depth / 5000, camera_model,
                                             center_back)

    #########################

    pcl_front = transfer_point_cloud(pcl_front_in_cam, cam_t)
    pcl_back = transfer_point_cloud(pcl_back_in_cam, cam_back_t)

    pcl_back = remove_points_overlapping_depth(pcl_back, data['depth'][i] * test.max_depth / 5000, camera_model,
                                               cam_t, center_front, excerpt_resolution)

    pcls.append(points_to_colorized_pcl(pcl_front,
                                        np.reshape(cv2.cvtColor(data['rgb'][i], cv2.COLOR_BGR2RGB), (-1, 3))))
    pcls.append(points_to_pcl(pcl_back, (0.0, 1.0, 0.0)))

    transforms.append(coordinates_colorized_lineset(cam_t, length=1))
    transforms.append(coordinates_colorized_lineset(cam_back_t, length=1))
    transforms.append(coordinates_colorized_lineset(obj_t, length=0.1))

    # visualize.extend(transforms)
    visualize.extend(pcls)

    o3d.visualization.draw_geometries(visualize)

    ### Visualize Point Cloud

    for x in visualize:
        vis.add_geometry(x)

    #######################
    vis.clear_geometries()
    visualize.clear()
    pcls.clear()
    transforms.clear()

vis.destroy_window()
