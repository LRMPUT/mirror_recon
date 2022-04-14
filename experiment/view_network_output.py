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
    data['rgb_obj'] = x[:, 0:3, ...]
    data['rgb_obj_proj'] = x[:, 3:6, ...]
    data['depth_obj'] = x[:, 6:7, ...]
    data['depth_obj_proj'] = x[:, 7:8, ...]
    return data


def dictify_output(x):
    data = {}
    data['rgb_obj_back'] = x[:, :3, ...]
    data['depth_obj_back'] = x[:, 3:4, ...]
    return data


def dictify_scene(x):
    data = {}
    data['rgb'] = x[:, 0:3, ...]
    data['rgb_back'] = x[:, 3:6, ...]
    data['depth'] = x[:, 6:7, ...]
    data['depth_back'] = x[:, 7:8, ...]
    return data


def dictify_prediction(x):
    data = {}
    data['depth_obj_pred'] = x[:, :1, ...]
    return data


def dictify_metrics(loss, l1_loss, ssim_loss):
    data = {}
    data['l1_loss'] = l1_loss
    data['ssim_loss'] = ssim_loss
    data['loss'] = loss
    return data


def get_last_epoch(path):
    files = os.listdir(path)
    files.sort(key=natural_keys)

    if len(files) == 0:
        return None

    pattern = re.compile(r"model_(\d+)\.pt")
    epoch = int(pattern.match(files[-1]).group(1))
    return epoch


def process_input(x, y, scale_factor):
    x[:, 3:5, ...] = scale_and_clip(x[:, 3:5, ...], scale_factor)
    y = scale_and_clip(y, scale_factor)
    return x, y


path = '/home/rafal/Datasets/graspable_filtered/shapeNetGraspable_1instance_cube'
load_path = '/home/rafal/Models/ral_3d_gen/cube_mask_dropout_0.5_data_0.8_scale_3.0'
images_save_path = '/home/rafal/Test/ral_3d_gen/graspable/cube_mask_dropout_0.5_data_0.8_scale_3.0'

makedirs(images_save_path)
cleardirs(images_save_path)

last_epoch = get_last_epoch(load_path)
start_epoch = last_epoch if last_epoch is not None else 0

_, _, test = get_full_segmentation_dataset(path, 0.8, 0.1, 0.1, seed=42)
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

camera_model = graspable_model()
labels = ['rgb', 'rgb_proj', 'rgb_back', 'depth', 'depth_proj', 'depth_back', 'depth_pred', 'rgb_scene',
          'rgb_scene_back', 'depth_scene',
          'depth_scene_back']

results_data = dict(zip(labels, [[] for _ in range(len(labels))]))
resolution = (480, 640)
excerpt_resolution = (96, 128)
scale_factor = 3.0

for i, d in enumerate(tqdm(test_laoder)):
    x, y, scene, obj, obj_back, cam, cam_back = d

    inputs = dictify_input(x)
    targets = dictify_output(y)
    scenes = dictify_scene(scene)

    network_input = torch.cat([torch.mean(x[:, :3, ...], dim=1, keepdim=True), x[:, 6:7, ...], x[:, 7:8, ...]], dim=1)

    network_input = network_input.to(device)

    network_input, y = process_input(network_input, y, scale_factor)

    x = x.to(device)
    y = y.to(device)
    network_input = network_input.to(device)

    pred = model(network_input) / scale_factor/1000
    predictions = dictify_prediction(pred.cpu().detach())

    results_data['rgb'].append(image_from_tensor(inputs['rgb_obj'][0]))
    results_data['rgb_proj'].append(image_from_tensor(inputs['rgb_obj_proj'][0]))
    results_data['rgb_back'].append(image_from_tensor(targets['rgb_obj_back'][0]))
    results_data['depth'].append(image_from_tensor(inputs['depth_obj'][0]))
    results_data['depth_proj'].append(image_from_tensor(inputs['depth_obj_proj'][0]))
    results_data['depth_back'].append(image_from_tensor(targets['depth_obj_back'][0]))
    results_data['depth_pred'].append(image_from_tensor(predictions['depth_obj_pred'][0]))
    results_data['rgb_scene'].append(image_from_tensor(scenes['rgb'][0]))
    results_data['rgb_scene_back'].append(image_from_tensor(scenes['rgb_back'][0]))
    results_data['depth_scene'].append(image_from_tensor(scenes['depth'][0]))
    results_data['depth_scene_back'].append(image_from_tensor(scenes['depth_back'][0]))

    ##### Examples Save #####

    imwrite(os.path.join(images_save_path, 'rgb_{:04d}.png'.format(i)), results_data['rgb'][i] * 255.0)
    imwrite(os.path.join(images_save_path, 'rgb_back_{:04d}.png'.format(i)), results_data['rgb_back'][i] * 255.0)
    imwrite(os.path.join(images_save_path, 'depth_{:04d}.png'.format(i)), results_data['depth'][i])
    imwrite(os.path.join(images_save_path, 'depth_back_{:04d}.png'.format(i)), results_data['depth_back'][i])
    imwrite(os.path.join(images_save_path, 'depth_pred_{:04d}.png'.format(i)), results_data['depth_pred'][i])

    # save_txt('Center Front: {}, Center Back: {}'.format(center_front.numpy(), center_back.numpy()), 'centers.txt')
    #########################
