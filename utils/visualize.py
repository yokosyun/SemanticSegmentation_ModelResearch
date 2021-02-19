import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from dataloader.dataset import NUM_CLASSES

COLOR_MAP = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
        [128, 64, 128],
    ]
)


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(
            mask == True,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return image


def visualize_segmentation(input_tensor, predicted_tensor):
    image = input_tensor.cpu()
    image = torch.squeeze(image)
    image = transforms.ToPILImage()(image).convert("RGB")
    image = np.array(image)

    masks = predicted_tensor.max(1)[1].squeeze(1)
    masks = masks.cpu()
    masks = torch.squeeze(masks)
    masks = masks.numpy().copy()
    masks = np.array(masks)

    for class_id in range(NUM_CLASSES):
        mask = masks[:, :] == class_id
        masked_image = apply_mask(image, mask, COLOR_MAP[class_id, :])

    masked_image = torch.from_numpy(masked_image.astype(np.float32)).clone()
    masked_image = masked_image.permute(2, 0, 1)
    masked_image = masked_image.unsqueeze(0)
    masked_image = masked_image / torch.max(masked_image)
    save_image(masked_image, "masked_image.png")