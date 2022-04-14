from piqa import ssim

from algorithms.tensor.image import drop, select_mask
from algorithms.tensor.point_cloud import distance_map
from network.losses import mask_loss
from utils.summaries import *

from network.data import *
from algorithms.tensor.camera import *

from torch.utils.data import DataLoader
from fastai.vision.learner import create_unet_model
from fastai.vision import models
from algorithms.tensor.losses import L1Loss_masked
from torch import nn
from tqdm import tqdm
import re


class TrainLogs(Logs):

    def update(self, inputs, targets, predictions, metrics, step):
        self._writer.add_images('train/rgb_obj', inputs['rgb_obj'], step)
        self._writer.add_images('train/depth_obj', inputs['depth_obj'], step)
        self._writer.add_images('train/depth_obj_proj', inputs['depth_obj_proj'], step)
        self._writer.add_images('train/depth_obj_back', targets['depth_obj_back'], step)
        self._writer.add_images('train/depth_obj_back_pred', predictions['depth_obj_back'], step)
        self._writer.add_scalar("train/l1_image_loss", metrics['l1_image_loss'], step)
        self._writer.add_scalar("train/l1_image_mask_loss", metrics['l1_image_mask_loss'], step)
        self._writer.add_scalar("train/ssim_loss", metrics['ssim_loss'], step)
        self._writer.add_scalar("train/pd_loss", metrics['pd_loss'], step)
        self._writer.add_scalar("train/loss", metrics['loss'], step)


class ValidLogs(Logs):

    def update(self, inputs, targets, predictions, metrics, step):
        self._writer.add_images('valid/rgb_obj', inputs['rgb_obj'], step)
        self._writer.add_images('valid/depth_obj', inputs['depth_obj'], step)
        self._writer.add_images('valid/depth_obj_proj', inputs['depth_obj_proj'], step)
        self._writer.add_images('valid/depth_obj_back', targets['depth_obj_back'], step)
        self._writer.add_images('valid/depth_obj_back_pred', predictions['depth_obj_back'], step)
        self._writer.add_scalar("valid/l1_image_loss", metrics['l1_image_loss'], step)
        self._writer.add_scalar("valid/l1_image_mask_loss", metrics['l1_image_mask_loss'], step)
        self._writer.add_scalar("valid/ssim_loss", metrics['ssim_loss'], step)
        self._writer.add_scalar("valid/pd_loss", metrics['pd_loss'], step)
        self._writer.add_scalar("valid/loss", metrics['loss'], step)


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


def dictify_metrics(loss, l1_image_loss, ssim_loss, pd_loss, l1_image_mask_loss):
    data = {}
    data['l1_image_loss'] = l1_image_loss
    data['l1_image_mask_loss'] = l1_image_mask_loss
    data['ssim_loss'] = ssim_loss
    data['pd_loss'] = pd_loss
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


path = "/home/rafal/Datasets/shapeNetGraspable_30instances/*"
save_path = "/home/rafal/Models/OtherSide/sample_model"

save_epoch = lambda x: x % 100 == 0
makedirs(save_path)

last_epoch = get_last_epoch(save_path)
start_epoch = last_epoch if last_epoch is not None else 0

train_logs_path = './logs/train'
valid_logs_path = './logs/valid'

train, valid, test = get_sparse_multi_segmentation_multi_class_dataset(path, 0.8, 0.1, 0.1, seed=42)
log_step = lambda x: x % 20 == 0

train_logs = TrainLogs(train_logs_path)
valid_logs = ValidLogs(valid_logs_path)

batch_size = 32
epochs = start_epoch + 1000
lr = 1e-3

camera_model = graspable_model()
max_depth = train.max_depth
divisor = 5000

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid, batch_size=1, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    model_name = 'model_{}.pth'.format(last_epoch)
    model = torch.load(os.path.join(save_path, model_name))
    print('Model {} has been loaded.'.format(model_name
                                             ))
except FileNotFoundError:
    encoder_model = models.resnet34
    model = create_unet_model(encoder_model, n_out=1, img_size=(96, 128), n_in=1)
    print('Model has been created.')

model = model.to(device)

image_criterion = nn.L1Loss()
point_criterion = nn.L1Loss()
ssim_criterion = ssim.MS_SSIM(n_channels=1, window_size=4).cuda() if torch.cuda.is_available() else ssim.MS_SSIM(
    n_channels=1,
    window_size=4)

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

train_step = 0
valid_step = 0
for e in range(start_epoch, epochs + 1):
    print(f"Epoch: {e}/{epochs + 1}")
    for i, d in enumerate(tqdm(train_loader)):
        x, y, seg, obj, obj_back, cam, cam_back = d

        inputs = dictify_input(x)
        targets = dictify_output(y)

        cam = cam.float()
        cam_back = cam_back.float()

        obj_global = torch.matmul(cam, obj[3].float())
        obj_back_global = torch.matmul(cam_back, obj_back[3].float())

        center = obj[2].float()
        center_back = obj_back[2].float()

        seg_idx = obj[1].float()

        network_x = x[:, 3:4, ...]
        network_y = y

        network_x = network_x.to(device)
        network_y = network_y.to(device)
        seg = seg.to(device)

        opt.zero_grad()
        pred = model(network_x)

        mask = select_mask(seg[:, 1:2, ...], seg_idx.to(device))
        x_dm = distance_map(pred, camera_model, obj_back_global.to(device), cam_back.to(device),
                            center_back.to(device))
        y_dm = distance_map(network_y, camera_model, obj_back_global.to(device), cam_back.to(device),
                            center_back.to(device))

        pd_loss = mask_loss(x_dm, y_dm, mask, l1=1.0, l2=0.0, criterion=point_criterion)

        l1_image_loss = image_criterion(pred, network_y)
        l1_image_mask_loss = L1Loss_masked(pred, network_y, mask)
        ssim_loss = 1 - ssim_criterion(torch.clamp(torch.abs(pred), 0.0, 1.0),
                                       torch.clamp(torch.abs(network_y), 0.0, 1.0))

        loss = 0.9 * (pd_loss + l1_image_mask_loss) + 0.1 * l1_image_loss

        loss.backward()
        opt.step()
        train_step += 1

        if log_step(train_step):
            predictions = dictify_output(pred.cpu())
            metrics = dictify_metrics(loss, l1_image_loss, ssim_loss, pd_loss, l1_image_mask_loss)

            train_logs.update(inputs, targets, predictions, metrics, train_step)

    if save_epoch(e):
        torch.save(model, os.path.join(save_path, 'model_{}.pth'.format(e)))

    for i, d in enumerate(valid_loader):
        x, y, seg, obj, obj_back, cam, cam_back = d

        inputs = dictify_input(x)
        targets = dictify_output(y)

        cam = cam.float()
        cam_back = cam_back.float()

        obj_global = torch.matmul(cam, obj[3].float())
        obj_back_global = torch.matmul(cam_back, obj_back[3].float())

        center = obj[2].float()
        center_back = obj_back[2].float()

        seg_idx = obj[1].float()

        network_x = x[:, 3:4, ...]
        network_y = y

        network_x = network_x.to(device)
        network_y = network_y.to(device)

        seg = seg.to(device)

        mask = select_mask(seg, seg_idx.to(device))
        mask = mask[:, 1:2, ...]

        pred = model(network_x)

        torch.rand(pred.shape, device=pred.device)

        x_dm = distance_map(pred, camera_model, obj_back_global.to(device), cam_back.to(device),
                            center_back.to(device))

        y_dm = distance_map(network_y, camera_model, obj_back_global.to(device), cam_back.to(device),
                            center_back.to(device))

        pd_loss = point_criterion(x_dm, y_dm)

        l1_image_loss = image_criterion(pred, network_y)
        l1_image_mask_loss = L1Loss_masked(pred, network_y, mask)
        ssim_loss = 1 - ssim_criterion(torch.clamp(torch.abs(pred), 0.0, 1.0),
                                       torch.clamp(torch.abs(network_y), 0.0, 1.0))

        loss = 0.9 * (pd_loss + l1_image_mask_loss) + 0.1 * l1_image_loss

        valid_step += 1

        if log_step(valid_step):
            predictions = dictify_output(pred.cpu())
            metrics = dictify_metrics(loss, l1_image_loss, ssim_loss, pd_loss, l1_image_mask_loss)

            valid_logs.update(inputs, targets, predictions, metrics, valid_step)
