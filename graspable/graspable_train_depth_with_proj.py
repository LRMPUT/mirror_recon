import re

from fastai.vision import models
from fastai.vision.learner import create_unet_model
from piqa import ssim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from graspable.data import *
from graspable.summaries import *


class TrainLogs(Logs):

    def update(self, inputs, targets, predictions, metrics, step):
        self._writer.add_images('train/depth_obj', inputs['depth_obj'], step)
        self._writer.add_images('train/depth_obj_proj', inputs['depth_obj_proj'], step)
        self._writer.add_images('train/depth_obj_back', targets['depth_obj_back'], step)
        self._writer.add_images('train/depth_obj_back_pred', predictions['depth_obj_back'], step)
        self._writer.add_scalar("train/l1_loss", metrics['l1_loss'], step)
        self._writer.add_scalar("train/ssim_loss", metrics['ssim_loss'], step)
        self._writer.add_scalar("train/loss", metrics['loss'], step)


class ValidLogs(Logs):

    def update(self, inputs, targets, predictions, metrics, step):
        self._writer.add_images('valid/depth_obj', inputs['depth_obj'], step)
        self._writer.add_images('valid/depth_obj_proj', inputs['depth_obj_proj'], step)
        self._writer.add_images('valid/depth_obj_back', targets['depth_obj_back'], step)
        self._writer.add_images('valid/depth_obj_back_pred', predictions['depth_obj_back'], step)
        self._writer.add_scalar("valid/l1_loss", metrics['l1_loss'], step)
        self._writer.add_scalar("valid/ssim_loss", metrics['ssim_loss'], step)
        self._writer.add_scalar("valid/loss", metrics['ssim_loss'], step)


def dictify_input(x):
    data = {}
    data['depth_obj'] = x[:, 0:1, ...]
    data['depth_obj_proj'] = x[:, 1:2, ...]
    return data


def dictify_output(x):
    data = {}
    data['depth_obj_back'] = x[:, :1, ...]
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


path = "/mnt/m2/datasets/RS/dataset_10_categories_15instances_1object"

save_path = "/mnt/m2/Models/ral_3d_gen_art/graspable_2ch_depth"
save_epoch = lambda x: x % 25 == 0
makedirs(save_path)

last_epoch = get_last_epoch(save_path)
start_epoch = last_epoch if last_epoch is not None else 0

train_logs_path = './logs/train'
valid_logs_path = './logs/valid'

train, valid, test = get_depth_rotation_dataset(path, 0.9, 0.1, 0.0, seed=42)
log_step = lambda x: x % 20 == 0

train_logs = TrainLogs(train_logs_path)
valid_logs = ValidLogs(valid_logs_path)

batch_size = 32
epochs = start_epoch + 400
lr = 1e-3

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
    model = create_unet_model(encoder_model, n_out=1, img_size=(96, 128), n_in=2)
    print('Model has been created.')

model = model.to(device)

criterion = nn.L1Loss()
ssim_criterion = ssim.MS_SSIM(n_channels=1, window_size=4).cuda() if torch.cuda.is_available() else ssim.MS_SSIM(
    n_channels=1,
    window_size=4)

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=10,
                                                       threshold=0.01,
                                                       threshold_mode='abs',
                                                       verbose=True)

train_step = 0
valid_step = 0
for e in range(start_epoch, epochs + 1):
    print(f"Epoch: {e}/{epochs + 1}")
    for i, d in enumerate(tqdm(train_loader)):
        x, y = d[:2]
        inputs = dictify_input(x)
        targets = dictify_output(y)

        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()

        pred = model(x)

        l1_loss = criterion(pred, y)
        ssim_loss = 1 - ssim_criterion(torch.clamp(torch.abs(pred), 0.0, 1.0), torch.clamp(torch.abs(y), 0.0, 1.0))
        loss = 0.7 * l1_loss + 0.3 * ssim_loss
        loss.backward()
        opt.step()
        train_step += 1

        if log_step(train_step):
            predictions = dictify_output(pred.cpu())
            metrics = dictify_metrics(loss, l1_loss, ssim_loss)

            train_logs.update(inputs, targets, predictions, metrics, train_step)

    if save_epoch(e):
        torch.save(model, os.path.join(save_path, 'model_{}.pth'.format(e)))

    for i, d in enumerate(valid_loader):
        x, y = d[:2]
        inputs = dictify_input(x)
        targets = dictify_output(y)

        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        l1_loss = loss
        ssim_loss = 1 - ssim_criterion(torch.clamp(torch.abs(pred), 0.0, 1.0), torch.clamp(torch.abs(y), 0.0, 1.0))

        valid_step += 1

        if log_step(valid_step):
            predictions = dictify_output(pred.cpu())
            metrics = dictify_metrics(loss, l1_loss, ssim_loss)

            valid_logs.update(inputs, targets, predictions, metrics, valid_step)
