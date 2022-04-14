import glob

from network.data import MultiSegmentationDataset
import cv2
import numpy as np
import os
import torch
import re
import argparse

parser = argparse.ArgumentParser(description='Remove empty scenes.')
parser.add_argument('--path', type=str, help='Path to dataset folder | regex format')

args = parser.parse_args()
paths = glob.glob(args.path)
paths = list(filter(os.path.isdir, paths))


def select_mask(x: torch.Tensor, index: torch.Tensor):
    index = index.view(1, 1)
    return torch.where(x == index, torch.ones(x.shape, device=x.device),
                       torch.zeros(x.shape, device=x.device))


def remove_files_at(x: dict, i):
    for k, v in x.items():
        # print(x[k][i])
        try:
            if os.path.isfile(x[k][i]):
                if os.path.exists(x[k][i]):
                    os.remove(x[k][i])
        except (TypeError, ValueError):
            pass


for p in paths:
    data = MultiSegmentationDataset.load(p)

    for i in range(len(data['seg'])):

        seg = torch.tensor(cv2.imread(data['seg_obj'][i], cv2.CV_16UC1).astype(np.float))
        seg_back = torch.tensor(cv2.imread(data['seg_obj_back'][i], cv2.CV_16UC1).astype(np.float))

        string = data['seg'][i]
        which = re.match(r".*segment(\d+)\.png", data['seg'][i]).group(1)
        image_id = int(which)

        object_data = MultiSegmentationDataset.object_data(data['objects'][i])
        idx = object_data[1]

        mask = select_mask(seg, torch.tensor(idx))
        mask_back = select_mask(seg_back, torch.tensor(idx))

        zero_count = torch.count_nonzero(mask)

        if torch.count_nonzero(mask) == 0 or torch.count_nonzero(mask_back) == 0:
            print(f" Removed idx: {image_id} in {os.path.split(p)[-1]}")
            remove_files_at(data, i)
