"""Pascal VOC Dataset Segmentation Dataloader"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import Dataset


def pascalVoc2012_datafinder(list_file, img_dir, mask_dir, transform=None):
    images = open(list_file, "rt").read().split("\n")[:-1]
    transform = transform

    img_extension = ".jpg"
    mask_extension = ".png"

    image_root_dir = img_dir
    mask_root_dir = mask_dir

    images = sorted(images)

    result = []

    for image in images:
        file_name = img_dir + "/" + image + img_extension
        result.append(file_name)
    print("------------")
    print(result)

    # image_root_dir + images

    return result
