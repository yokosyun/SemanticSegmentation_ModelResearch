"""
Train a SegNet model
"""

from __future__ import print_function
import argparse
from dataloader.dataset import PascalVOCDataset, NUM_CLASSES
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.unet import UNet
from models.segnet import SegNet
from models.pspnet import PSPNet
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


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
parser.add_argument("--train_path", required=True, type=str)
parser.add_argument("--img_dir", required=True, type=str)
parser.add_argument("--mask_dir", required=True, type=str)
parser.add_argument("--save_dir", required=True, type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--save_segmentation", type=bool, default=False)


args = parser.parse_args()


def color_map(indices_image):
    batch_size = indices_image.size()[0]
    rows = indices_image.size()[1]
    cols = indices_image.size()[2]
    final_out = torch.empty(3, rows, cols)
    for batch in range(batch_size):
        result = torch.zeros(3, rows, cols)
        for row in range(rows):
            for col in range(cols):
                if indices_image[batch, row, col] == 0:
                    result[0, row, col] = 0
                    result[1, row, col] = 0
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 1:
                    result[0, row, col] = 128
                    result[1, row, col] = 0
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 2:
                    result[0, row, col] = 0
                    result[1, row, col] = 128
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 3:
                    result[0, row, col] = 128
                    result[1, row, col] = 128
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 4:
                    result[0, row, col] = 0
                    result[1, row, col] = 0
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 5:
                    result[0, row, col] = 128
                    result[1, row, col] = 0
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 6:
                    result[0, row, col] = 0
                    result[1, row, col] = 128
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 7:
                    result[0, row, col] = 128
                    result[1, row, col] = 128
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 8:
                    result[0, row, col] = 64
                    result[1, row, col] = 0
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 9:
                    result[0, row, col] = 192
                    result[1, row, col] = 0
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 10:
                    result[0, row, col] = 64
                    result[1, row, col] = 128
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 11:
                    result[0, row, col] = 192
                    result[1, row, col] = 128
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 12:
                    result[0, row, col] = 64
                    result[1, row, col] = 0
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 13:
                    result[0, row, col] = 192
                    result[1, row, col] = 0
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 14:
                    result[0, row, col] = 64
                    result[1, row, col] = 128
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 15:
                    result[0, row, col] = 192
                    result[1, row, col] = 128
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 16:
                    result[0, row, col] = 0
                    result[1, row, col] = 64
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 17:
                    result[0, row, col] = 128
                    result[1, row, col] = 64
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 18:
                    result[0, row, col] = 0
                    result[1, row, col] = 192
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 19:
                    result[0, row, col] = 128
                    result[1, row, col] = 192
                    result[2, row, col] = 0

                elif indices_image[batch, row, col] == 20:
                    result[0, row, col] = 0
                    result[1, row, col] = 64
                    result[2, row, col] = 128

                elif indices_image[batch, row, col] == 21:
                    result[0, row, col] = 128
                    result[1, row, col] = 64
                    result[2, row, col] = 128

        if batch == 0:
            final_out = result
            final_out = final_out.unsqueeze(0)
        else:
            final_out = torch.cat([final_out, result.unsqueeze(0)], axis=0)

    save_image(final_out, "result.jpg")


def train():
    is_better = True
    prev_loss = float("inf")

    model.train()

    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()
        idx = 0

        for batch in train_dataloader:
            iteration_time = time.time()
            input_tensor = torch.autograd.Variable(batch["image"])
            target_tensor = torch.autograd.Variable(batch["mask"])

            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            predicted_tensor = model(input_tensor)

            save_image(input_tensor / torch.max(input_tensor), "input_tensor.png")
            save_image(
                predicted_tensor.max(1)[1].unsqueeze(1).float(), "predicted_tensor.png"
            )
            save_image(target_tensor.unsqueeze(1).float(), "target_tensor.png")

            if args.save_segmentation:
                softmaxed_tensor = F.softmax(predicted_tensor, dim=1)
                softmaxed_tensor = torch.max(softmaxed_tensor, dim=1)
                color_map(softmaxed_tensor.indices)

            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_tensor)

            print("epoch=", epoch, " :idx=", idx, " :loss=", loss)
            idx += 1
            loss.backward()
            optimizer.step()

            loss_f += loss.float()

            print("iteration_time = ", time.time() - iteration_time, "[s]")

        # add log
        avg_train_loss = loss_f / len(train_dataloader)
        writer_train.add_scalar("avg_train_loss", avg_train_loss, epoch)

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, args.model + "_best.pth"),
            )

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch + 1, loss_f, delta))


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

    if args.model == "unet":
        model = UNet(
            input_channels=NUM_INPUT_CHANNELS, output_channels=NUM_OUTPUT_CHANNELS
        )
    elif args.model == "segnet":
        model = SegNet(
            input_channels=NUM_INPUT_CHANNELS, output_hannels=NUM_OUTPUT_CHANNELS
        )
    else:
        model = PSPNet(
            layers=50,
            bins=(1, 2, 3, 6),
            dropout=0.1,
            classes=NUM_OUTPUT_CHANNELS,
            use_ppm=True,
            pretrained=True,
        )

    class_weights = 1.0 / train_dataset.get_class_probability()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if CUDA:
        model = model.cuda(GPU_ID)

        class_weights = class_weights.cuda(GPU_ID)
        criterion = criterion.cuda(GPU_ID)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train()