import errno
import os
import re


def makedirs(dir):
    if os.path.exists(dir) is False:
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def up(path):
    return os.path.dirname(path)


def file_to_string(path):
    with open(path, 'r') as file:
        data = file.read()

    return data


def string_to_numbers(x: str, sep=' '):
    vals = x.split(sep)
    values = [float(v) for v in vals]
    return values


def natural_keys(text):
    def _atoi(txt):
        return int(txt) if txt.isdigit() else txt

    return [_atoi(c) for c in re.split(r'(\d+)', text)]


def slice_dict(x, begin, end):
    out = {}
    for k in x:
        out[k] = x[k][begin:end]

    return out
