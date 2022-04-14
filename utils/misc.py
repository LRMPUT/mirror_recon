import os
import errno
import re
import shutil


def makedirs(dir):
    if os.path.exists(dir) is False:
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def cleardirs(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def up(path):
    return os.path.dirname(path)


def file_to_string(path):
    with open(path, 'r') as file:
        data = file.read()

    return data


def file_to_lines(path):
    with open(path, 'r') as file:
        data = file.readlines()

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


def join_dict(a: dict, b: dict, join_func=lambda x, y: x + y):
    out = {}
    for k in a.keys():
        if k in b.keys():
            out[k] = join_func(a[k], b[k])
    return out


def save_txt(x, path):
    with open(path, "w") as f:
        f.write(x)
