import re
from point_cloud_utils.data import *
from utils.functions import *
import cv2
import glob
from algorithms.camera import *
import functools
from utils.misc import join_dict


def get_full_segmentation_dataset(path, *args, seed=None):
    data = SegmentationDataset.load(path)
    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [FullSegmentationDataset(data[i]) for i in range(len(args))]
    return datasets


def get_sparse_multi_segmentation_multi_class_dataset(path, *args, seed=None):
    paths = glob.glob(path)
    paths = list(filter(os.path.isdir, paths))
    data = [MultiSegmentationDataset.load(p) for p in paths]
    data = functools.reduce(join_dict, data)

    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [SparseMultiSegmentationDataset(data[i]) for i in range(len(args))]
    return datasets


def get_multi_segmentation_multi_class_dataset(path, *args, seed=None):
    paths = glob.glob(path)
    paths = list(filter(os.path.isdir, paths))
    data = [MultiSegmentationDataset.load(p) for p in paths]
    data = functools.reduce(join_dict, data)

    data = split_data_in_dict(data, *args, seed=seed)
    datasets = [MultiSegmentationDataset(data[i]) for i in range(len(args))]
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

        # input_tensor = torch.cat([rgb_obj, depth_obj, depth_obj_proj], dim=0)
        # output_tensor = torch.cat([depth_obj_back], dim=0)

        objects = self.parse_objects(self.data['objects'][item])
        objects_back = self.parse_objects(self.data['objects_back'][item])

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), obj, obj_back, cam, cam_back

    def __len__(self):
        return len(self.data['rgb'])

    @staticmethod
    def load_cam(path):
        def _parse_line(x: str):
            data = x.strip().split(' ')
            transform = np.reshape(list(map(float, data[-16:])), (4, 4))

            return transform

        with open(path, 'r') as f:
            lines = f.readlines()

            return [_parse_line(line) for line in lines]

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
            'objects_back': [],
            'cam': [],
            'cam_back': []

        }

        cam_poses = RotationDataset.load_cam(os.path.join(path, 'camPoses.txt'))

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
                data['cam'].append(cam_poses[idx])
                data['cam_back'].append(cam_poses[idx + 1])

        return data

    @staticmethod
    def parse_objects(path):
        def _parse_line(x: str):
            data = x.strip().split(' ')
            cls = data[0]
            instance = data[1]
            center = list(map(float, data[2:4]))
            transform = list(map(float, data[4:]))

            return cls, instance, center, transform

        with open(path, 'r') as f:
            lines = f.readlines()

            return _parse_line(lines[-1])

    @staticmethod
    def object_data(x):
        return [x[0], np.array(x[2]), np.reshape(x[3], (4, 4))]


class SegmentationDataset(Dataset):

    def __init__(self, data):
        self.data = data

        # self.max_depth = 31923.0
        self.max_depth = 32000.0
        # self.max_depth = 10e3

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0
        rgb_obj_proj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_proj'][item]).astype(np.float)) / 255.0

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

        # input_tensor = torch.cat([rgb_obj, depth_obj, depth_obj_proj], dim=0)
        # output_tensor = torch.cat([depth_obj_back], dim=0)

        objects = self.parse_objects(self.data['objects'][item])
        objects_back = self.parse_objects(self.data['objects_back'][item])

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), obj, obj_back, cam, cam_back

    def __len__(self):
        return len(self.data['rgb'])

    @staticmethod
    def load_cam(path):
        def _parse_line(x: str):
            data = x.strip().split(' ')
            transform = np.reshape(list(map(float, data[-16:])), (4, 4))

            return transform

        with open(path, 'r') as f:
            lines = f.readlines()

            return [_parse_line(line) for line in lines]

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
            'objects_back': [],
            'cam': [],
            'cam_back': []

        }

        cam_poses = RotationDataset.load_cam(os.path.join(path, 'camPoses.txt'))

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
                data['cam'].append(cam_poses[idx])
                data['cam_back'].append(cam_poses[idx + 1])

        return data

    @staticmethod
    def parse_objects(path):
        def _parse_line(x: str):
            data = x.strip().split(' ')
            cls = data[0]
            instance = data[1]
            segment = int(data[2])
            center = list(map(float, data[3:5]))
            transform = list(map(float, data[5:]))

            return cls, instance, segment, center, transform

        with open(path, 'r') as f:
            lines = f.readlines()

            return _parse_line(lines[-1])

    @staticmethod
    def object_data(x):
        return [x[0], x[2], np.array(x[3]), np.reshape(x[4], (4, 4))]


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
        objects_back = self.parse_objects(self.data['objects_back'][item])

        input_tensor = torch.cat([rgb_obj, depth_obj, depth_obj_proj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), obj, obj_back, cam, cam_back


class SparseSegmentationDataset(SegmentationDataset):

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        objects = self.parse_objects(self.data['objects'][item])
        objects_back = self.parse_objects(self.data['objects_back'][item])

        input_tensor = torch.cat([rgb_obj, depth_obj, depth_obj_proj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), obj, obj_back, cam, cam_back


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


class FullRotationDataset(RotationDataset):

    def __getitem__(self, item):
        rgb = tensor_from_nch_image(cv2.imread(self.data['rgb'][item]).astype(np.float)) / 255.0
        rgb_back = tensor_from_nch_image(cv2.imread(self.data['rgb_back'][item]).astype(np.float)) / 255.0

        depth = tensor_from_1ch_image(
            cv2.imread(self.data['depth'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0
        rgb_obj_back = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_back'][item]).astype(np.float)) / 255.0
        rgb_obj_proj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_proj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        dist_obj = tensor_from_1ch_image(cv2.imread(self.data['dist_obj'][item], cv2.CV_16UC1).astype(np.float))
        dist_obj /= np.linalg.norm(dist_obj.shape[-2:])

        input_tensor = torch.cat([rgb_obj, rgb_obj_proj, depth_obj, depth_obj_proj, dist_obj], dim=0)
        output_tensor = torch.cat([rgb_obj_back, depth_obj_back], dim=0)

        scene_tensor = torch.cat([rgb, rgb_back, depth, depth_back], dim=0)

        objects = self.parse_objects(self.data['objects'][item])
        objects_back = self.parse_objects(self.data['objects_back'][item])

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), scene_tensor.float(), obj, obj_back, cam, cam_back

    def __len__(self):
        return len(self.data['rgb'])


class FullSegmentationDataset(SegmentationDataset):

    def __getitem__(self, item):
        print(item)
        rgb = tensor_from_nch_image(cv2.imread(self.data['rgb'][item]).astype(np.float)) / 255.0
        rgb_back = tensor_from_nch_image(cv2.imread(self.data['rgb_back'][item]).astype(np.float)) / 255.0

        depth = tensor_from_1ch_image(
            cv2.imread(self.data['depth'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0
        rgb_obj_back = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_back'][item]).astype(np.float)) / 255.0
        rgb_obj_proj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_proj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        dist_obj = tensor_from_1ch_image(cv2.imread(self.data['dist_obj'][item], cv2.CV_16UC1).astype(np.float))
        dist_obj /= np.linalg.norm(dist_obj.shape[-2:])

        input_tensor = torch.cat([rgb_obj, rgb_obj_proj, depth_obj, depth_obj_proj, dist_obj], dim=0)
        output_tensor = torch.cat([rgb_obj_back, depth_obj_back], dim=0)

        scene_tensor = torch.cat([rgb, rgb_back, depth, depth_back], dim=0)

        objects = self.parse_objects(self.data['objects'][item])
        objects_back = self.parse_objects(self.data['objects_back'][item])

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), scene_tensor.float(), obj, obj_back, cam, cam_back

    def __len__(self):
        return len(self.data['rgb'])


class MultiSegmentationDataset(Dataset):

    def __init__(self, data):
        self.data = data

        # self.max_depth = 31923.0
        self.max_depth = 32000.0
        # self.max_depth = 10e3

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0
        rgb_obj_proj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_proj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        seg_obj = tensor_from_1ch_image(
            cv2.imread(self.data['seg_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        seg_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['seg_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        dist_obj = tensor_from_1ch_image(cv2.imread(self.data['dist_obj'][item], cv2.CV_16UC1).astype(np.float))
        dist_obj /= np.linalg.norm(dist_obj.shape[-2:])

        input_tensor = torch.cat([rgb_obj, rgb_obj_proj, depth_obj, depth_obj_proj, dist_obj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        seg_tensor = torch.cat([seg_obj, seg_obj_back], dim=0)

        objects = self.data['objects'][item]
        objects_back = self.data['objects_back'][item]

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), seg_tensor.float(), obj, obj_back, cam, cam_back

    def __len__(self):
        return len(self.data['rgb'])

    @staticmethod
    def load_cam(path):
        def _parse_line(x: str):
            data = x.strip().split(' ')
            transform = np.reshape(list(map(float, data[-16:])), (4, 4))

            return transform

        with open(path, 'r') as f:
            lines = f.readlines()

            return [_parse_line(line) for line in lines]

    @staticmethod
    def load(path):
        def _is_same_object(x, y):
            if x[0] == y[0] and x[1] == y[1] and x[2] == y[2]:
                return True
            return False

        def _object_matching(x, y):
            matches = []

            i_idx, j_idx = list(range(len(x))), list(range(len(y)))
            i_used, j_used = [0] * len(x), [0] * len(y)
            for i in i_idx:
                for j in j_idx:
                    if _is_same_object(x[i], y[i]) and i_used[i] == 0 and j_used[j] == 0:
                        matches.append([i, j])
                        i_used[i] = 1
                        j_used[j] = 1
            return matches

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

            'seg': [],
            'seg_back': [],
            'seg_obj': [],
            'seg_obj_back': [],

            'objects_file': [],
            'objects_file_back': [],

            'objects': [],
            'objects_back': [],
            'cam': [],
            'cam_back': []

        }

        cam_poses = MultiSegmentationDataset.load_cam(os.path.join(path, 'camPoses.txt'))

        objects = glob.glob(os.path.join(path, 'objects*'))
        objects = list(filter(lambda x: not os.path.isdir(x), objects))
        objects.sort()

        count = len(objects)

        for i in range(0, count, 2):
            try:
                a = MultiSegmentationDataset.parse_objects(objects[i])
                b = MultiSegmentationDataset.parse_objects(objects[i + 1])
            except IndexError:
                continue
            matches = _object_matching(a, b)

            for match in matches:
                m, n = match
                if m == 0 or n == 0:
                    continue
                which = re.match(r".*objects(\d+)\.dat", objects[i]).group(1)
                chars = len(which)
                idx = int(which)

                keys = list(data.keys())
                values = [os.path.join(path, "rgb{:0{}d}.png").format(idx, chars),
                          os.path.join(path, "rgb{:0{}d}.png").format(idx + 1, chars),
                          os.path.join(path, "rgbIn{:0{}d}yaw0obj{}.png").format(idx, chars, m - 1),
                          os.path.join(path, "rgbIn{:0{}d}yaw-179obj{}.png").format(idx, chars, m - 1),
                          os.path.join(path, "rgbProj{:0{}d}.png").format(idx, chars),
                          os.path.join(path, "rgbProj{:0{}d}yaw-179obj{}.png").format(idx, chars, m - 1),
                          os.path.join(path, "depth{:0{}d}.png").format(idx, chars),
                          os.path.join(path, "depth{:0{}d}.png").format(idx + 1, chars),
                          os.path.join(path, "depthIn{:0{}d}yaw0obj{}.png").format(idx, chars, m - 1),
                          os.path.join(path, "depthIn{:0{}d}yaw-179obj{}.png").format(idx, chars, m - 1),
                          os.path.join(path, "depthProj{:0{}d}.png").format(idx, chars),
                          os.path.join(path, "depthProj{:0{}d}yaw-179obj{}.png").format(idx, chars, m - 1),
                          os.path.join(path, "segment{:0{}d}.png").format(idx, chars),
                          os.path.join(path, "segment{:0{}d}.png").format(idx + 1, chars),
                          os.path.join(path, "segmentIn{:0{}d}yaw0obj{}.png").format(idx, chars, m - 1),
                          os.path.join(path, "segmentIn{:0{}d}yaw-179obj{}.png").format(idx, chars, m - 1),
                          objects[i],
                          objects[i + 1]]

                if not all(map(os.path.exists, values)):
                    continue

                for k, v in zip(keys, values):
                    data[k].append(v)

                data['objects'].append(a[m])
                data['objects_back'].append(b[m])
                data['cam'].append(cam_poses[idx])
                data['cam_back'].append(cam_poses[idx + 1])
                # data['rgb'].append(os.path.join(path, "rgb{:0{}d}.png").format(idx, chars))
                # data['rgb_back'].append(os.path.join(path, "rgb{:0{}d}.png").format(idx + 1, chars))
                # data['rgb_obj'].append(os.path.join(path, "rgbIn{:0{}d}yaw0obj{}.png").format(idx, chars, i - 1))
                # data['rgb_obj_back'].append(
                #     os.path.join(path, "rgbIn{:0{}d}yaw-179obj{}.png").format(idx, chars, i - 1))
                # data['rgb_proj'].append(os.path.join(path, "rgbProj{:0{}d}.png").format(idx, chars))
                # data['rgb_obj_proj'].append(
                #     os.path.join(path, "rgbProj{:0{}d}yaw-179obj{}.png").format(idx, chars, i - 1))
                #
                # data['depth'].append(os.path.join(path, "depth{:0{}d}.png").format(idx, chars))
                # data['depth_back'].append(os.path.join(path, "depth{:0{}d}.png").format(idx + 1, chars))
                # data['depth_obj'].append(os.path.join(path, "depthIn{:0{}d}yaw0obj{}.png").format(idx, chars, i - 1))
                # data['depth_obj_back'].append(
                #     os.path.join(path, "depthIn{:0{}d}yaw-179obj{}.png").format(idx, chars, i - 1))
                # data['depth_proj'].append(os.path.join(path, "depthProj{:0{}d}.png").format(idx, chars))
                # data['depth_obj_proj'].append(
                #     os.path.join(path, "depthProj{:0{}d}yaw-179obj{}.png").format(idx, chars, i - 1))
                #
                # data['seg'].append(os.path.join(path, "segment{:0{}d}.png").format(idx, chars))
                # data['seg_back'].append(os.path.join(path, "segment{:0{}d}.png").format(idx + 1, chars))
                # data['seg_obj'].append(os.path.join(path, "segmentIn{:0{}d}yaw0obj{}.png").format(idx, chars, i - 1))
                # data['seg_obj_back'].append(
                #     os.path.join(path, "segmentIn{:0{}d}yaw-179obj{}.png").format(idx, chars, i - 1))

        return data

    @staticmethod
    def parse_objects(path):
        def _parse_line(x: str):
            data = x.strip().split(' ')
            cls = data[0]
            instance = data[1]
            segment = int(data[2])
            center = list(map(float, data[3:5]))
            transform = list(map(float, data[5:]))

            return cls, instance, segment, center, transform

        with open(path, 'r') as f:
            lines = f.readlines()

            return list(map(_parse_line, lines))

    @staticmethod
    def object_data(x):
        return [x[0], x[2], np.array(x[3]), np.reshape(x[4], (4, 4))]


class SparseMultiSegmentationDataset(MultiSegmentationDataset):

    def __getitem__(self, item):
        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        seg_obj = tensor_from_1ch_image(
            cv2.imread(self.data['seg_obj'][item], cv2.CV_16UC1).astype(np.float))
        seg_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['seg_obj_back'][item], cv2.CV_16UC1).astype(np.float))

        objects = self.data['objects'][item]
        objects_back = self.data['objects_back'][item]

        input_tensor = torch.cat([rgb_obj, depth_obj, depth_obj_proj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        seg_tensor = torch.cat([seg_obj, seg_obj_back], dim=0)

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), seg_tensor.float(), obj, obj_back, cam, cam_back


class FullMultiSegmentationDataset(MultiSegmentationDataset):
    def __init__(self, data):
        super().__init__(data)

    def __getitem__(self, item):
        rgb = tensor_from_nch_image(cv2.imread(self.data['rgb'][item]).astype(np.float)) / 255.0
        rgb_back = tensor_from_nch_image(cv2.imread(self.data['rgb_back'][item]).astype(np.float)) / 255.0

        depth = tensor_from_1ch_image(
            cv2.imread(self.data['depth'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        rgb_obj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj'][item]).astype(np.float)) / 255.0
        rgb_obj_proj = tensor_from_nch_image(cv2.imread(self.data['rgb_obj_proj'][item]).astype(np.float)) / 255.0

        depth_obj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        depth_obj_proj = tensor_from_1ch_image(
            cv2.imread(self.data['depth_obj_proj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        seg_obj = tensor_from_1ch_image(
            cv2.imread(self.data['seg_obj'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth
        seg_obj_back = tensor_from_1ch_image(
            cv2.imread(self.data['seg_obj_back'][item], cv2.CV_16UC1).astype(np.float)) / self.max_depth

        input_tensor = torch.cat([rgb_obj, rgb_obj_proj, depth_obj, depth_obj_proj], dim=0)
        output_tensor = torch.cat([depth_obj_back], dim=0)
        seg_tensor = torch.cat([seg_obj, seg_obj_back], dim=0)

        scene_tensor = torch.cat([rgb, rgb_back, depth, depth_back], dim=0)

        objects = self.data['objects'][item]
        objects_back = self.data['objects_back'][item]

        obj = self.object_data(objects)
        obj_back = self.object_data(objects_back)

        cam = self.data['cam'][item]
        cam_back = self.data['cam_back'][item]

        return input_tensor.float(), output_tensor.float(), seg_tensor.float(), scene_tensor.float(), obj, obj_back, cam, cam_back

    def __len__(self):
        return len(self.data['rgb'])
