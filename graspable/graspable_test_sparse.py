import re

from piqa import ssim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from graspable.data import *
from graspable.plot import save_images_cascade
from graspable.summaries import *


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


path = '/media/SSD/Data/dataset_shapeNetGraspable_30instances_test'

load_path = '/home/rafal/Models/ral_3d_gen/graspable/5ch'

images_save_path = '/home/rafal/Test/ral_3d_gen/graspable/5ch'

makedirs(images_save_path)

last_epoch = get_last_epoch(load_path)
start_epoch = last_epoch if last_epoch is not None else 0

test_logs_path = './logs/test'

_, _, test = get_sparse_rotation_dataset(path, 0.0, 0.0, 1.0, seed=42)
log_step = lambda x: x % 20 == 0

test_logs = TestLogs(test_logs_path)

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
image_rows = 6
image_count = len(test_laoder)
labels = ['rgb', 'depth', 'depth_proj', 'depth_back', 'depth_pred']

results_data = dict(zip(labels, [[] for _ in range(len(labels))]))
l1_losses = []
ssim_losses = []
l1_losses_per_class = {}

for i, d in enumerate(tqdm(test_laoder)):
    x, y, cls = d
    inputs = dictify_input(x)
    targets = dictify_output(y)

    x = x.to(device)
    y = y.to(device)

    pred = model(x)
    predictions = dictify_prediction(pred.cpu().detach())

    results_data['rgb'].append(image_from_tensor(inputs['rgb_obj'][0]))
    results_data['depth'].append(image_from_tensor(inputs['depth_obj'][0]))
    results_data['depth_proj'].append(image_from_tensor(inputs['depth_obj_proj'][0]))
    results_data['depth_back'].append(image_from_tensor(targets['depth_obj_back'][0]))
    results_data['depth_pred'].append(image_from_tensor(predictions['depth_obj_pred'][0]))

    l1_loss = criterion(pred, y).cpu().detach().numpy()
    ssim_loss = 1 - ssim_criterion(torch.clamp(torch.abs(pred), 0.0, 1.0),
                                   torch.clamp(torch.abs(y), 0.0, 1.0)).cpu().detach().numpy()

    l1_losses.append(l1_loss)
    ssim_losses.append(ssim_loss)
    if not l1_losses_per_class.get(cls[0][0]):
        l1_losses_per_class[cls[0][0]] = []

    l1_losses_per_class[cls[0][0]].append(l1_loss)

print("L1 loss: ", np.mean(l1_losses))
print("SSIM loss: ", np.mean(ssim_losses))

for k in l1_losses_per_class.keys():
    print('Object: {}, Mean loss: {}, Count: {}'.format(k, np.mean(l1_losses_per_class[k]), len(l1_losses_per_class[k])))

for v in results_data.values():
    for i in range(len(v)):
        v[i] = np.squeeze(v[i])

for i in range(image_count // image_rows):
    results = slice_dict(results_data, image_rows * i, image_rows * (i + 1))
    save_images_cascade(
        os.path.join(images_save_path, 'figure_{:04d}.png'.format(i)), list(results.values()), list(results.keys()))
