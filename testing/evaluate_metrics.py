import torch
from piqa import ssim
from utils.summaries import *

from network.data import *

from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from utils.plot import save_images_cascade
from algorithms.tensor.image import select_mask
from algorithms.tensor.losses import L1Loss_masked
import matplotlib.pyplot as plt
import re


class TestLogs(Logs):

    def update(self, inputs, targets, predictions, metrics, step):
        self._writer.add_images('train/rgb_obj', inputs['rgb_obj'], step)
        self._writer.add_images('train/depth_obj', inputs['depth_obj'], step)
        self._writer.add_images('train/depth_obj_proj', inputs['depth_obj_proj'], step)
        self._writer.add_images('train/depth_obj_back', targets['depth_obj_back'], step)
        self._writer.add_images('train/depth_obj_back_pred', predictions['depth_obj_back'], step)
        self._writer.add_scalar("train/l1_loss", metrics['l1_loss'], step)
        self._writer.add_scalar("train/ssim_loss", metrics['ssim_loss'], step)
        self._writer.add_scalar("train/loss", metrics['loss'], step)


def dictify_input(x):
    data = {}
    data['rgb_obj'] = x[:, :1, ...]
    data['depth_obj'] = x[:, 1:2, ...]
    data['depth_obj_proj'] = x[:, 2:3, ...]
    return data


def dictify_output(x):
    data = {}
    data['depth_obj_back'] = x[:, :1, ...]
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


path = '/home/rafal/Datasets/shapeNetGraspable_30instances/*'

load_path = '/home/rafal/Models/OtherSide/shapeNetGraspable_30instances_dp_mae_mask'

last_epoch = get_last_epoch(load_path)
start_epoch = last_epoch if last_epoch is not None else 0

test_logs_path = './logs/testing'

_, _, test = get_sparse_multi_segmentation_multi_class_dataset(path, 0.8, 0.1, 0.1, seed=42)
log_step = lambda x: x % 20 == 0

test_logs = TestLogs(test_logs_path)

test_laoder = DataLoader(dataset=test, batch_size=1, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load(os.path.join(load_path, 'model_{}.pth'.format(last_epoch)))
model.eval()
model = model.to(device)

l1_criterion = nn.L1Loss()

ssim_criterion = ssim.MS_SSIM(n_channels=1, window_size=4).cuda() if torch.cuda.is_available() else ssim.MS_SSIM(
    n_channels=1,
    window_size=4)

test_step = 0

#### Experiment #####
to_cam_dist = test.max_depth / 5000
image_rows = 6
image_count = len(test_laoder)
labels = ['rgb', 'depth', 'depth_proj', 'depth_back', 'depth_pred']

l1_losses = []
l1_mask_losses = []
ssim_losses = []
l1_losses_per_class = {}
l1_mask_losses_per_class = {}

for i, d in enumerate(tqdm(test_laoder)):
    x, y, seg, obj, obj_back, cam, cam_back = d
    cls = obj[0]
    # x = torch.cat([torch.mean(x[:, :3, ...], dim=1, keepdim=True), x[:, 3:4, ...], x[:, 4:5, ...]], dim=1) #gdp
    # x = x[:, 3:4, ...] # d
    x = torch.cat([x[:, 3:4, ...], x[:, 4:5, ...]], dim=1) # dp
    # x = torch.cat([x[:, :3, ...], x[:, 3:4, ...]], dim=1) # rgbd
    # x = torch.cat([x[:, :3, ...], x[:, 3:4, ...], x[:, 4:5, ...]], dim=1) # rgbdp

    seg_idx = obj[1].float()

    x = x.to(device)
    y = y.to(device)
    seg = seg.to(device)

    mask = select_mask(seg, seg_idx.to(device))
    mask = mask[:, 1:2, ...]

    foreground_count = torch.count_nonzero(mask, dim=(1, 2, 3))

    pred = model(x)

    predictions = dictify_prediction(pred.cpu().detach())

    l1_loss = l1_criterion(pred, y).cpu().detach().numpy()
    l1_loss_mask = L1Loss_masked(pred, y, mask).cpu().detach().numpy()
    ssim_loss = 1 - ssim_criterion(torch.clamp(torch.abs(pred), 0.0, 1.0),
                                   torch.clamp(torch.abs(y), 0.0, 1.0)).cpu().detach().numpy()

    l1_losses.append(l1_loss)
    l1_mask_losses.append(l1_loss_mask)
    ssim_losses.append(ssim_loss)

    if not l1_losses_per_class.get(cls[0]):
        l1_losses_per_class[cls[0]] = []

    if not l1_mask_losses_per_class.get(cls[0]):
        l1_mask_losses_per_class[cls[0]] = []

    l1_losses_per_class[cls[0]].append(l1_loss)
    l1_mask_losses_per_class[cls[0]].append(l1_loss_mask)

print("L1 loss: ", np.mean(l1_losses))
print("L1 loss [m]: ", np.mean(l1_losses) * to_cam_dist)
print("L1 mask loss: ", np.mean(l1_mask_losses))
print("L1 mask loss [m]: ", np.mean(l1_mask_losses) * to_cam_dist)
print("SSIM loss: ", np.mean(ssim_losses))

for k in l1_losses_per_class.keys():
    print(
        'Object: {}, Mean loss: {}, Mean loss [m]: {}, Mean mask loss: {}, Mean mask loss [m]: {},  Count: {}'.format(k,
                                                                                                                      np.mean(
                                                                                                                          l1_losses_per_class[
                                                                                                                              k]),
                                                                                                                      np.mean(
                                                                                                                          l1_losses_per_class[
                                                                                                                              k]) * to_cam_dist,
                                                                                                                      np.mean(
                                                                                                                          l1_mask_losses_per_class[
                                                                                                                              k]),
                                                                                                                      np.mean(
                                                                                                                          l1_mask_losses_per_class[
                                                                                                                              k]) * to_cam_dist,
                                                                                                                      len(
                                                                                                                          l1_losses_per_class[
                                                                                                                              k])))
