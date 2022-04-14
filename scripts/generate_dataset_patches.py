import numpy as np
import cv2
from utils.misc import file_to_lines, makedirs
import glob
import os
import re

path = '/home/rafal/Datasets/dataset_1_category_25instances_1object_box'
save_path = '/home/rafal/Datasets/dataset_1_category_25instances_1object_box_patches'
h, w = 96, 128

rgb_pattern = re.compile('.*rgb(\d+)\.png')
depth_pattern = re.compile('.*depth(\d+)\.png')
rgb_proj_name = 'rgbProj{:05d}.png'
depth_proj_name = 'depthProj{:05d}.png'

rgb_proj_odd_patch_name = 'rgbProj{:05d}yaw-179obj0.png'
depth_proj_odd_patch_name = 'depthProj{:05d}yaw-179obj0.png'

rgb_in_even_patch_name = 'rgbIn{:05d}yaw0obj0.png'
rgb_in_odd_patch_name = 'rgbIn{:05d}yaw-179obj0.png'
depth_in_even_patch_name = 'depthIn{:05d}yaw0obj0.png'
depth_in_odd_patch_name = 'depthIn{:05d}yaw-179obj0.png'


def get_center(path):
    lines = file_to_lines(path)
    center = lines[1].split(' ')[3:5]
    center = list(map(int, center))
    return center


def in_bounds(min_val, max_val, left, right):
    if min_val < left or max_val > right:
        return False
    return True


makedirs(save_path)

rgbs = list(filter(rgb_pattern.match, glob.glob(os.path.join(path, 'rgb*'))))
depths = list(filter(depth_pattern.match, glob.glob(os.path.join(path, 'depth*'))))
objects = glob.glob(os.path.join(path, 'objects*'))

rgbs.sort()
depths.sort()
objects.sort()

for i, (r, d, o) in enumerate(zip(rgbs, depths, objects)):
    rgb_image = cv2.imread(r)
    depth_image = cv2.imread(d, cv2.CV_16UC1)
    c_w, c_h = get_center(o)
    im_h, im_w = np.shape(rgb_image)[:2]

    rgb_idx = int(rgb_pattern.match(r).group(1))
    depth_idx = int(depth_pattern.match(d).group(1))
    assert rgb_idx == depth_idx

    if not in_bounds(c_h - h // 2, c_h + h // 2, 0, im_h):
        continue

    if not in_bounds(c_w - w // 2, c_w + w // 2, 0, im_w):
        continue

    rgb_patch = rgb_image[c_h - h // 2:c_h + h // 2, c_w - w // 2:c_w + w // 2]
    depth_patch = depth_image[c_h - h // 2:c_h + h // 2, c_w - w // 2:c_w + w // 2]

    cv2.imwrite(os.path.join(save_path, rgb_in_even_patch_name).format(rgb_idx), rgb_patch)
    cv2.imwrite(os.path.join(save_path, depth_in_even_patch_name).format(depth_idx), depth_patch)

    if rgb_idx % 2 == 0:
        cv2.imwrite(os.path.join(save_path, rgb_in_even_patch_name).format(rgb_idx), rgb_patch)
        cv2.imwrite(os.path.join(save_path, depth_in_even_patch_name).format(depth_idx), depth_patch)
    else:
        rgb_proj_image = cv2.imread(os.path.join(path, rgb_proj_name.format(rgb_idx - 1)))
        depth_proj_image = cv2.imread(os.path.join(path, depth_proj_name.format(depth_idx - 1)))

        rgb_proj_patch = rgb_proj_image[c_h - h // 2:c_h + h // 2, c_w - w // 2:c_w + w // 2]
        depth_proj_patch = depth_proj_image[c_h - h // 2:c_h + h // 2, c_w - w // 2:c_w + w // 2]

        cv2.imwrite(os.path.join(save_path, rgb_in_odd_patch_name).format(rgb_idx - 1), rgb_patch)
        cv2.imwrite(os.path.join(save_path, depth_in_odd_patch_name).format(depth_idx - 1), depth_patch)
        cv2.imwrite(os.path.join(save_path, rgb_proj_odd_patch_name).format(rgb_idx - 1), rgb_proj_patch)
        cv2.imwrite(os.path.join(save_path, depth_proj_odd_patch_name).format(depth_idx - 1), depth_proj_patch)
