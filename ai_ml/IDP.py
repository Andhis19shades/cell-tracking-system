import torchvision.transforms.functional as TF
from pathlib import Path
import tifffile as tiff
from torch.utils.data import Dataset
import torch
import numpy as np


class CellDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(self.image_dir.glob("*.tif"))
        self.masks = sorted(self.mask_dir.glob("*.tif"))

        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Number of images ({len(self.images)}) "
                f"does not match masks ({len(self.masks)})"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # -----------------------------
        # Read image and mask
        # -----------------------------
        image = tiff.imread(self.images[index])
        mask = tiff.imread(self.masks[index])

        # -----------------------------
        # Normalize image
        # -----------------------------
        image = image.astype(np.float32)

        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)

        # -----------------------------
        # Convert image to tensor
        # -----------------------------
        image = torch.tensor(image).unsqueeze(0)

        # -----------------------------
        # Convert mask to binary
        # 0 = background
        # >0 = cell
        # -----------------------------
        mask = (mask > 0).astype(np.float32)

        mask = torch.tensor(mask).unsqueeze(0)

        # -----------------------------
        # Resize
        # -----------------------------
        image = TF.resize(
            image,
            [512, 512]
        )

        mask = TF.resize(
            mask,
            [512, 512],
            interpolation=TF.InterpolationMode.NEAREST
        )

        # -----------------------------
        # Final mask shape:
        # [512, 512]
        # -----------------------------
        mask = mask.squeeze(0).long()

        return image, mask