import re
import cv2
from point_cloud_utils.data import *
from tqdm import tqdm

path = "/media/SSD/Data/dataset_10_categories_15instances_1object"

files = os.listdir(path)
files.sort()

max_value = 0

for file in tqdm(files):
    match = re.match(r"depth(\d+).png", file)
    if match is not None:
        idx = match.group(1)
        image = cv2.imread(os.path.join(path, file), cv2.CV_16UC1).astype(np.float)
        max_value = np.maximum(np.max(image), max_value)

print(f"Max value is: {max_value}")

# Max value is: 31923.0
