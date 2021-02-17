"""Pascal VOC Dataset Segmentation Dataloader"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import Dataset

VOC_CLASSES = (
    "background",  # always index 0
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

NUM_CLASSES = len(VOC_CLASSES) + 1


class PascalVOCDataset(Dataset):
    """Pascal VOC 2012 Dataset"""

    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {"image": image, "mask": gt_mask}

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            raw_image = Image.open(mask_path).resize((224, 224))
            imx_t = np.array(raw_image).reshape(224 * 224)
            imx_t[imx_t == 255] = len(VOC_CLASSES)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / float(np.sum(values))

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path).convert("RGB")
        raw_image = raw_image.resize((224, 224))
        raw_image = transforms.ToTensor()(raw_image)
        return raw_image

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))

        imx_t = np.array(raw_image)
        # 255 is unlabeled so it need to set as next of last class
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
        imx_t[imx_t == 255] = len(VOC_CLASSES)
        tensor = torch.LongTensor(imx_t)

        return tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_root = os.path.join(
        "/media/yoko/SSD-PGU3/workspace/datasets", "VOCdevkit", "VOC2012"
    )
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    img_dir = os.path.join(data_root, "JPEGImages")
    mask_dir = os.path.join(data_root, "SegmentationObject")

    objects_dataset = PascalVOCDataset(
        list_file=list_file_path, img_dir=img_dir, mask_dir=mask_dir
    )

    print(objects_dataset.get_class_probability())

    sample = objects_dataset[0]
    image, mask = sample["image"], sample["mask"]

    image.transpose_(0, 2)
    image.transpose_(0, 1)

    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)

    plt.show()
