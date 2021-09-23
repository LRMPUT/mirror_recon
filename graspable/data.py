import os
import re

import cv2
from torch.utils.data import Dataset

from graspable.camera import *
from graspable.functions import *

"""
input: rgb, depth, projection back
output: depth back
"""

"""
even - front
odd - back
rgb, depth

depthIn(Idx-even)yaw0obj0 - cropped 
depthIn(Idx-even)yaw-179obj0.png - cropped depth back, optional
depthProj(Idx-even) - projected depth back
depthProj(Idx-even)yaw-179obj0 - cropped projected depth back, optional
distIn(Idx-even)yaw0obj0.png - dist cropped 

"""


def get_rotation_dataset(path, *args, seed=None):
    data = RotationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [RotationDataset(data[i]) for i in range(len(args))]
    return datasets


def get_sparse_rotation_dataset(path, *args, seed=None):
    data = RotationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [SparseRotationDataset(data[i]) for i in range(len(args))]
    return datasets


def get_rgbd_rotation_dataset(path, *args, seed=None):
    data = RotationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [RgbdRotationDataset(data[i]) for i in range(len(args))]
    return datasets


def get_depth_rotation_dataset(path, *args, seed=None):
    data = RotationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [DepthRotationDataset(data[i]) for i in range(len(args))]
    return datasets


def get_depth_dataset(path, *args, seed=None):
    data = RotationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [DepthDataset(data[i]) for i in range(len(args))]
    return datasets


def get_simple_rotation_dataset(path, *args, seed=None):
    data = SimpleRotationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [SimpleRotationDataset(data[i]) for i in range(len(args))]
    return datasets


def get_reconstruction_rotation_dataset(path, camera_model, *args, seed=None):
    data = RotationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [ReconstructedRotationDataset(data[i], camera_model) for i in range(len(args))]
    return datasets


class RotationDataset(Dataset):

    def __init__(self, data):
        self.data = data
        # self.max_depth = 31923.0
        self.max_depth = 32000.0
        # self.max_depth = 10e3

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0
        rgb_obj_proj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_proj'][item]).astype(np.float)) / 255.0

        objects = self.parse_objects(self.data['objects'][item])

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        dist_obj = tensor_from_1ch_image(cv2.imread(self.data['dist_obj'][item], cv2.CV_16UC1).astype(np.float))
        dist_obj /= np.linalg.norm(dist_obj.shape[-2:])

        input_tensor = torch.cat([rgb_obj, rgb_obj_proj, depth_obj, depth_obj_proj, dist_obj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        cls = [objects[0]]

        return input_tensor.float(), output_tensor.float(), cls

    def __len__(self):
        return len(self.data['rgb'])

    @staticmethod
    def load(path):
        data = {
            'rgb': [],
            'rgb_back': [],
            'rgb_obj': [],
            'rgb_obj_back': [],
            'rgb_proj': [],
            'rgb_obj_proj': [],

            'depth': [],
            'depth_back': [],
            'depth_obj': [],
            'depth_obj_back': [],
            'depth_proj': [],
            'depth_obj_proj': [],

            'dist_obj': [],

            'objects': [],
            'objects_back': []

        }

        files = os.listdir(path)
        files.sort()

        for file in files:
            match = re.match(r"rgbIn(\d+)yaw-179obj0.png", file)
            if match is not None:
                idx = match.group(1)
                chars = len(idx)
                idx = int(idx)

                if not os.path.exists(os.path.join(path, "rgbIn{:0{}d}yaw0obj0.png").format(idx, chars)):
                    continue

                data['rgb'].append(os.path.join(path, "rgb{:0{}d}.png").format(idx, chars))
                data['rgb_back'].append(os.path.join(path, "rgb{:0{}d}.png").format(idx + 1, chars))
                data['rgb_obj'].append(os.path.join(path, "rgbIn{:0{}d}yaw0obj0.png").format(idx, chars))
                data['rgb_obj_back'].append(os.path.join(path, "rgbIn{:0{}d}yaw-179obj0.png").format(idx, chars))
                data['rgb_proj'].append(os.path.join(path, "rgbProj{:0{}d}.png").format(idx, chars))
                data['rgb_obj_proj'].append(os.path.join(path, "rgbProj{:0{}d}yaw-179obj0.png").format(idx, chars))

                data['depth'].append(os.path.join(path, "depth{:0{}d}.png").format(idx, chars))
                data['depth_back'].append(os.path.join(path, "depth{:0{}d}.png").format(idx + 1, chars))
                data['depth_obj'].append(os.path.join(path, "depthIn{:0{}d}yaw0obj0.png").format(idx, chars))
                data['depth_obj_back'].append(os.path.join(path, "depthIn{:0{}d}yaw-179obj0.png").format(idx, chars))
                data['depth_proj'].append(os.path.join(path, "depthProj{:0{}d}.png").format(idx, chars))
                data['depth_obj_proj'].append(os.path.join(path, "depthProj{:0{}d}yaw-179obj0.png").format(idx, chars))

                data['dist_obj'].append(os.path.join(path, "distIn{:0{}d}yaw0obj0.png").format(idx, chars))

                data['objects'].append(os.path.join(path, "objects{:0{}d}.dat").format(idx, chars))
                data['objects_back'].append(os.path.join(path, "objects{:0{}d}.dat").format(idx + 1, chars))

        return data

    @staticmethod
    def parse_objects(path):
        def _parse_line(x: str):
            data = x.split(' ')
            cls = data[0]
            instance = data[1]
            center = list(map(float, data[2:4]))
            transform = list(map(float, data[4:]))

            return cls, instance, center, transform

        with open(path, 'r') as f:
            lines = f.readlines()

            return _parse_line(lines[-1])


class SimpleRotationDataset(RotationDataset):

    def __getitem__(self, item):
        rgb = cv2.imread(self.data['rgb'][item]).astype(np.float)
        depth = cv2.imread(self.data['depth'][item], cv2.CV_16UC1).astype(np.float)
        depth_back = cv2.imread(self.data['depth_back'][item], cv2.CV_16UC1).astype(np.float)
        depth_proj = cv2.imread(self.data['depth_proj'][item], cv2.CV_16UC1).astype(np.float)
        images = [rgb, depth, depth_back, depth_proj]

        rgb = tensor_from_nch_image(images[0]) / 255.0

        objects = self.parse_objects(self.data['objects'][item])

        depth = tensor_from_1ch_image(images[1]) / self.max_depth
        depth_back = tensor_from_1ch_image(images[2]) / self.max_depth
        depth_proj = tensor_from_1ch_image(images[3]) / self.max_depth

        input_tensor = torch.cat([rgb, depth, depth_proj], dim=0)
        output_tensor = torch.cat([depth_back], dim=0)
        cls = [objects[0]]

        return input_tensor.float(), output_tensor.float(), cls

    def __len__(self):
        return len(self.data['rgb'])

    @staticmethod
    def load(path):
        data = {
            'rgb': [],
            'rgb_back': [],
            'rgb_proj': [],

            'depth': [],
            'depth_back': [],
            'depth_proj': [],

            'objects': [],
            'objects_back': []

        }

        files = os.listdir(path)
        files.sort()

        for file in files:
            match = re.match(r"rgb(\d+).png", file)
            if match is not None:
                idx = match.group(1)
                chars = len(idx)
                idx = int(idx)

                if idx % 2 == 0:
                    if not os.path.exists(os.path.join(path, "rgb{:0{}d}.png").format(idx + 1, chars)):
                        continue

                    data['rgb'].append(os.path.join(path, "rgb{:0{}d}.png").format(idx, chars))
                    data['rgb_back'].append(os.path.join(path, "rgb{:0{}d}.png").format(idx + 1, chars))
                    data['rgb_proj'].append(os.path.join(path, "rgb{:0{}d}Proj.png").format(idx + 1, chars))

                    data['depth'].append(os.path.join(path, "depth{:0{}d}.png").format(idx, chars))
                    data['depth_back'].append(os.path.join(path, "depth{:0{}d}.png").format(idx + 1, chars))
                    data['depth_proj'].append(os.path.join(path, "depth{:0{}d}Proj.png").format(idx + 1, chars))

                    data['objects'].append(os.path.join(path, "objects{:0{}d}.dat").format(idx, chars))
                    data['objects_back'].append(os.path.join(path, "objects{:0{}d}.dat").format(idx + 1, chars))

        return data


class SparseRotationDataset(RotationDataset):

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        objects = self.parse_objects(self.data['objects'][item])

        input_tensor = torch.cat([rgb_obj, depth_obj, depth_obj_proj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        cls = [objects[0]]

        return input_tensor.float(), output_tensor.float(), cls


class RgbdRotationDataset(RotationDataset):

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        objects = self.parse_objects(self.data['objects'][item])

        input_tensor = torch.cat([rgb_obj, depth_obj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        cls = [objects[0]]

        return input_tensor.float(), output_tensor.float(), cls


class DepthRotationDataset(RotationDataset):

    def __getitem__(self, item):
        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        objects = self.parse_objects(self.data['objects'][item])

        input_tensor = torch.cat([depth_obj, depth_obj_proj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        cls = [objects[0]]

        return input_tensor.float(), output_tensor.float(), cls


class DepthDataset(RotationDataset):

    def __getitem__(self, item):
        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        objects = self.parse_objects(self.data['objects'][item])

        input_tensor = torch.cat([depth_obj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        cls = [objects[0]]

        return input_tensor.float(), output_tensor.float(), cls


class ReconstructedRotationDataset(RotationDataset):
    def __init__(self, data, camera_model):
        super().__init__(data)
        self.camera_model = camera_model

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0

        depth_obj = cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)
        depth_obj_back = cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)
        depth_obj_proj = cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)

        objects = self.parse_objects(self.data['objects'][item])
        objects_back = self.parse_objects(self.data['objects_back'][item])

        depth_obj_reproject = get_point_map_with_offcenter(depth_obj, self.camera_model, objects[3])
        depth_obj_reproject_back = get_point_map_with_offcenter(depth_obj_back, self.camera_model, objects_back[3])

        depth_obj = tensor_from_1ch_image(depth_obj) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(depth_obj_back) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(depth_obj_proj) / self.max_depth

        depth_obj_reproject = tensor_from_nch_image(depth_obj_reproject)
        depth_obj_reproject_back = tensor_from_nch_image(depth_obj_reproject_back)

        input_tensor = torch.cat([rgb_obj, depth_obj, depth_obj_proj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        reconstruction_tensor = torch.cat([depth_obj_reproject, depth_obj_reproject_back], dim=0)
        return input_tensor.float(), output_tensor.float(), reconstruction_tensor.float()
