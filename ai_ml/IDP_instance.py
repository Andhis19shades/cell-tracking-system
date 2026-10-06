import torch
import numpy as np
import tifffile as tiff

from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from scipy.ndimage import binary_erosion


class InstanceCellDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(
            self.image_dir.glob("*.tif")
        )

        self.masks = sorted(
            self.mask_dir.glob("*.tif")
        )

        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Image count ({len(self.images)}) "
                f"does not match mask count ({len(self.masks)})"
            )

    def __len__(self):
        return len(self.images)

    def create_three_class_mask(self, mask):

        """
        Converts CTC instance mask into:

        0 = Background
        1 = Cell interior
        2 = Cell boundary
        """

        # Output mask
        result = np.zeros(
            mask.shape,
            dtype=np.int64
        )

        # Find individual instance IDs
        instance_ids = np.unique(mask)

        # Remove background
        instance_ids = instance_ids[
            instance_ids != 0
        ]

        for instance_id in instance_ids:

            instance = (
                mask == instance_id
            )

            # Ignore extremely small regions
            if instance.sum() < 10:
                continue

            # Erode instance
            eroded = binary_erosion(
                instance,
                iterations=1
            )

            # Interior
            result[
                eroded
            ] = 1

            # Boundary
            boundary = (
                instance
                & ~eroded
            )

            result[
                boundary
            ] = 2

        return result

    def __getitem__(self, index):

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = tiff.imread(
            self.images[index]
        )

        # --------------------------------------------------
        # Load INSTANCE mask
        # --------------------------------------------------

        mask = tiff.imread(
            self.masks[index]
        )

        # --------------------------------------------------
        # Normalize image
        # --------------------------------------------------

        image = image.astype(
            np.float32
        )

        min_value = image.min()
        max_value = image.max()

        if max_value > min_value:

            image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        else:

            image = np.zeros_like(
                image,
                dtype=np.float32
            )

        # --------------------------------------------------
        # Convert image to tensor
        # --------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        image = image.unsqueeze(0)

        # --------------------------------------------------
        # Create 3-class mask
        # --------------------------------------------------

        mask = self.create_three_class_mask(
            mask
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        # --------------------------------------------------
        # Add channel temporarily
        # --------------------------------------------------

        mask = mask.unsqueeze(0)

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        image = TF.resize(
            image,
            [512, 512],
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            [512, 512],
            interpolation=InterpolationMode.NEAREST
        )

        # --------------------------------------------------
        # Remove mask channel
        # --------------------------------------------------

        mask = mask.squeeze(0)

        return image, mask