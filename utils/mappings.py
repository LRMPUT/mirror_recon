import numpy as np


def linear_map(value, minimum, maximum, ranges):
    value = (value - minimum) / (maximum - minimum)

    length = len(ranges) - 1
    back = int(np.floor(value * length))
    front = int(np.ceil(value * length))

    w_back = 1.0 - (value * length - back)
    w_front = 1.0 - (front - value * length)

    if w_back == 1.0 and w_front == 1.0:
        w_front = 0.0

    out = ranges[back] * w_back + ranges[front] * w_front
    return tuple(out)


def heat_map(value, minimum, maximum):
    return linear_map(value, minimum, maximum,
                      np.array(
                          [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


def apply_mapping(x, mapping):
    shape = np.shape(x)
    x = np.reshape(x, -1)
    out = np.vectorize(mapping)(x)
    channels = np.prod(np.shape(out)) // np.prod(shape)
    out = np.reshape(out, (channels,) + shape)
    axes = np.roll(np.arange(0, len(shape) + 1), -1)
    return np.transpose(out, axes=axes)
