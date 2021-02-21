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
import torch.nn as nn
from utils.visualize import *
from dataloader.dataset import PascalVOCDataset, NUM_CLASSES

# Arguments
parser = argparse.ArgumentParser(description="Train a SegNet model")

parser.add_argument("--data_root", required=True, type=str)
parser.add_argument("--train_path", required=True, type=str)
parser.add_argument("--img_dir", required=True, type=str)
parser.add_argument("--mask_dir", required=True, type=str)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--gpu", type=int)

args = parser.parse_args()

if __name__ == "__main__":
    data_root = args.data_root
    train_path = os.path.join(data_root, args.train_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    train_dataset = PascalVOCDataset(
        list_file=train_path, img_dir=img_dir, mask_dir=mask_dir
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()

    total_acc = np.zeros(NUM_CLASSES)
    for batch in train_dataloader:
        iteration_time = time.time()
        gt_tensor = torch.autograd.Variable(batch["mask"])

        pred_tensor = gt_tensor

        cls_num = 0

        for cls in range(NUM_CLASSES):
            gt_pixel = gt_tensor == cls
            pred_pixel = pred_tensor == cls
            TP_TN = gt_pixel == pred_pixel
            FP_FN = gt_pixel != pred_pixel
            TP = TP_TN == gt_pixel
            num_gt_pixel = torch.sum(pred_pixel.float())
            num_pred_pixel = torch.sum(pred_pixel.float())
            num_TP = torch.sum(TP.float())

            cls_acc = num_TP / (num_gt_pixel + num_pred_pixel - num_TP + 1e-6)
            if num_gt_pixel != 0:
                cls_num += 1
                total_acc[cls] += cls_acc
            else:
                total_acc[cls] += 1

        print(total_acc)

    print("--eval done--")
    avg_total_acc = total_acc / len(train_dataloader)
    print("avg_total_acc=", avg_total_acc)
    all_acc = np.mean(avg_total_acc)
    print("all_acc=", all_acc)