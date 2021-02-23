"""
Train a SegNet model
"""

from __future__ import print_function
import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.visualize import *
from dataloader.dataset import PascalVOCDataset, NUM_CLASSES

from models.unet import UNet
from models.segnet import SegNet
from models.pspnet import PSPNet

writer_train = SummaryWriter(log_dir="./logs/train")
writer_test = SummaryWriter(log_dir="./logs/test")


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES

NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
MOMENTUM = 0.9

# Arguments
parser = argparse.ArgumentParser(description="Train a SegNet model")

parser.add_argument("--model", required=True, type=str)
parser.add_argument("--data_root", required=True, type=str)
parser.add_argument("--test_path", required=True, type=str)
parser.add_argument("--img_dir", required=True, type=str)
parser.add_argument("--mask_dir", required=True, type=str)
parser.add_argument("--save_dir", required=True, type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--save_segmentation", type=bool, default=False)


args = parser.parse_args()
import torchvision.transforms as transforms


def inference(test_dataset):
    model.eval()

    for test_img in test_dataset:
        input_image = Image.open(test_img)

        print(test_img)
        # input_image.save("result/inference/" + test_img.split("/")[-1])

        img_tesnsor = transforms.ToTensor()(input_image)

        img_tesnsor = img_tesnsor.unsqueeze(0)
        if CUDA:
            img_tesnsor = img_tesnsor.cuda()

        iteration_time = time.time()
        predicted_tensor = model(img_tesnsor)
        print("iteration_time = ", time.time() - iteration_time, "[s]")

        if True:
            img = predicted_tensor.max(1)[1].squeeze(0)
            img = img.data.cpu().numpy()
            img = (img).astype("uint8")
            img = Image.fromarray(img)
            img.save("result/inference/" + test_img.split("/")[-1])


if __name__ == "__main__":
    data_root = args.data_root
    test_path = os.path.join(data_root, args.test_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    from dataloader.pascalVoc2012_datafinder import *

    test_dataset = pascalVoc2012_datafinder(
        list_file=test_path, img_dir=img_dir, mask_dir=mask_dir
    )

    if args.model == "unet":
        model = UNet(
            input_channels=NUM_INPUT_CHANNELS, output_channels=NUM_OUTPUT_CHANNELS
        )

    if CUDA:
        model = model.cuda(GPU_ID)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    inference(test_dataset)