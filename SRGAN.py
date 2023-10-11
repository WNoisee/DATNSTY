import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from tensorflow.keras.utils import image_dataset_from_directory
import os
import math
import numpy as np
import tarfile
import urllib.request

# URL của tệp tgz
url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

# Thư mục để lưu trữ dữ liệu
data_dir = os.path.expanduser("~/.keras/datasets/BSR")

# Tải tệp tgz
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    urllib.request.urlretrieve(url, os.path.join(data_dir, "BSR_bsds500.tgz"))

# Giải nén tệp tgz
with tarfile.open(os.path.join(data_dir, "BSR_bsds500.tgz"), "r:gz") as tar:
    tar.extractall(data_dir)

root_dir = os.path.join(data_dir, "BSR", "BSDS500")
