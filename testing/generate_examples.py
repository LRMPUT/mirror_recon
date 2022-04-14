from piqa import ssim
from torch import nn
from tqdm import tqdm

from algorithms.tensor.image import scale_and_clip
from network.data import *
from utils.misc import *
from utils.geometries import *
from algorithms.tensor.point_cloud import remove_points_overlapping_depth
from imageio import imwrite
from utils.misc import save_txt


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

    files = list(filter(pattern.match, files))

    epoch = int(pattern.match(files[-1]).group(1))
    return epoch


path = '/mnt/m2/datasets/RS/shapeNetGraspable_30instances/shapeNetGraspable_30instances_mug'
load_path = '/home/rafal/Models/OtherSide/shapeNetGraspable_30instances_dp_mae_mask'
images_save_path = '/home/rafal/Test/OtherSide/gdp_no_drop_mug'

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
max_examples = 20

for i, d in enumerate(tqdm(test_laoder)):
    if i == max_examples:
        break
    x, y, scene, obj, obj_back, cam, cam_back = d

    obj_t = obj[3][0].numpy()
    obj_back_t = obj_back[3][0].numpy()
    cam_t = cam[0].numpy()
    cam_back_t = cam_back[0].numpy()

    inputs = dictify_input(x)
    targets = dictify_output(y)
    scenes = dictify_scene(scene)

    network_input = torch.cat([torch.mean(x[:, :3, ...], dim=1, keepdim=True), x[:, 6:7, ...], x[:, 7:8, ...]], dim=1)

    network_input = network_input.to(device)

    x = x.to(device)
    y = y.to(device)

    pred = model(network_input)
    predictions = dictify_prediction(pred.cpu().detach())

    results_data['rgb'].append(image_from_tensor(inputs['rgb_obj'][0]) * 255.0)
    results_data['rgb_proj'].append(image_from_tensor(inputs['rgb_obj_proj'][0]) * 255.0)
    results_data['rgb_back'].append(image_from_tensor(targets['rgb_obj_back'][0]) * 255.0)
    results_data['depth'].append(image_from_tensor(inputs['depth_obj'][0]) * test.max_depth)
    results_data['depth_proj'].append(image_from_tensor(inputs['depth_obj_proj'][0]) * test.max_depth)
    results_data['depth_back'].append(image_from_tensor(targets['depth_obj_back'][0]) * test.max_depth)
    results_data['depth_pred'].append(image_from_tensor(predictions['depth_obj_pred'][0]) * test.max_depth)
    results_data['rgb_scene'].append(image_from_tensor(scenes['rgb'][0]) * 255.0)
    results_data['rgb_scene_back'].append(image_from_tensor(scenes['rgb_back'][0]) * 255.0)
    results_data['depth_scene'].append(image_from_tensor(scenes['depth'][0]) * test.max_depth)
    results_data['depth_scene_back'].append(image_from_tensor(scenes['depth_back'][0]) * test.max_depth)

    center_front, center_back = obj[2][0], obj_back[2][0]

    pcl_front_in_cam = get_pcl_with_offcenter(results_data['depth'][i] / 5000, camera_model,
                                              center_front)
    pcl_back_in_cam = get_pcl_with_offcenter(results_data['depth_pred'][i] / 5000, camera_model,
                                             center_back)

    # save_txt('Center Front: {}, Center Back: {}'.format(center_front.numpy(), center_back.numpy()), 'centers.txt')
    #########################

    pcl_front = transfer_point_cloud(pcl_front_in_cam, cam_t)
    pcl_back = transfer_point_cloud(pcl_back_in_cam, cam_back_t)

    pcl_back = remove_points_overlapping_depth(pcl_back, results_data['depth'][i] / 5000, camera_model,
                                               cam_t, center_front, excerpt_resolution)

    ##### Examples Save #####

    imwrite(os.path.join(images_save_path, 'rgb_{:04d}.png'.format(i)),
            cv2.cvtColor(results_data['rgb'][i], cv2.COLOR_BGR2RGB))
    imwrite(os.path.join(images_save_path, 'rgb_back_{:04d}.png'.format(i)),
            cv2.cvtColor(results_data['rgb_back'][i], cv2.COLOR_BGR2RGB))
    imwrite(os.path.join(images_save_path, 'depth_{:04d}.png'.format(i)), results_data['depth'][i].astype(np.uint16))
    imwrite(os.path.join(images_save_path, 'depth_back_{:04d}.png'.format(i)),
            results_data['depth_back'][i].astype(np.uint16))
    imwrite(os.path.join(images_save_path, 'depth_pred_{:04d}.png'.format(i)),
            results_data['depth_pred'][i].astype(np.uint16))
    imwrite(os.path.join(images_save_path, 'depth_proj_{:04d}.png'.format(i)),
            results_data['depth_proj'][i].astype(np.uint16))


    o3d.io.write_point_cloud(os.path.join(images_save_path, 'points_{:04d}.ply'.format(i)),
                             points_to_colorized_pcl(pcl_front,
                                                     np.reshape(cv2.cvtColor(results_data['rgb'][i] / 255.0,
                                                                             cv2.COLOR_BGR2RGB),
                                                                (-1, 3))), write_ascii=True)

    o3d.io.write_point_cloud(os.path.join(images_save_path, 'points_pred_{:04d}.ply'.format(i)),
                             points_to_pcl(pcl_back, (0.0, 1.0, 0.0)), write_ascii=True)

    save_txt(' '.join(map(str, np.reshape(cam_t, -1))), os.path.join(images_save_path,'cam_{:04d}.txt'.format(i)))
    save_txt(' '.join(map(str, np.reshape(cam_back_t, -1))), os.path.join(images_save_path,'cam_back_{:04d}.txt'.format(i)))
    save_txt(' '.join(map(str, np.reshape(center_front.numpy(), -1))), os.path.join(images_save_path,'center_{:04d}.txt'.format(i)))
    save_txt(' '.join(map(str, np.reshape(center_back.numpy(), -1))), os.path.join(images_save_path,'center_back_{:04d}.txt'.format(i)))
