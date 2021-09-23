import numpy as np
import random
import copy
import torch


def split(x, *args):
    if len(args) < 2:
        return [x]

    sum = 0
    for num in args:
        sum += float(num)

    chunks = []

    weights = np.array(args)
    weights /= np.linalg.norm(weights, ord=1)
    idx = np.cumsum(weights) * len(x)
    idx = np.insert(idx.astype(np.int32), 0, 0)

    for i in range(len(idx) - 1):
        chunks.append(x[idx[i]:idx[i + 1]])

    return chunks


def split_data_in_dict(data: dict, *args, seed=None):
    out = [{} for _ in range(len(args))]

    data = copy.deepcopy(data)
    for k in data.keys():
        if type(seed) == int:
            rand = random.Random(seed)
            rand.shuffle(data[k])
        chunks = split(data[k], *args)
        for i in range(len(args)):
            out[i][k] = chunks[i]

    return out


def tensor_from_1ch_image(image: np.array) -> torch.Tensor:
    return torch.from_numpy(image).unsqueeze(dim=0)


def tensor_from_nch_image(image: np.array) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1)


def image_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).numpy()
    return img.astype(np.float32)


def images_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(0, 2, 3, 1).numpy()
    return img.astype(np.float32)
