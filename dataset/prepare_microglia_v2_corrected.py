from pathlib import Path

import cv2
import numpy as np
import tifffile


BASE = Path(
    "/Users/abhyudaysingh/microglia_dataset"
)

IMAGE_PATH = (
    BASE
    / "images"
    / "microglia_example_5frames.tif"
)

MASK_PATH = (
    BASE
    / "masks"
    / "microglia_example_labels_3frames.tif"
)

IMAGE_OUT = (
    BASE
    / "v2_corrected_images"
)

MASK_OUT = (
    BASE
    / "v2_corrected_masks"
)

IMAGE_OUT.mkdir(
    parents=True,
    exist_ok=True
)

MASK_OUT.mkdir(
    parents=True,
    exist_ok=True
)


def normalize(image):

    image = image.astype(
        np.float32
    )

    low = np.percentile(
        image,
        1
    )

    high = np.percentile(
        image,
        99
    )

    if high <= low:
        low = float(image.min())
        high = float(image.max())

    if high <= low:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    image = (
        image - low
    ) / (
        high - low
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return (
        image * 255.0
    ).astype(
        np.uint8
    )


def make_target(instance_mask):

    """
    Corrected microglia target.

    Class 0:
        background

    Class 1:
        COMPLETE microglia,
        including thin processes

    Class 2:
        ONE-PIXEL OUTER RING surrounding
        the microglia.

    The outer ring is deliberately outside
    the cell so that class 1 preserves the
    entire process structure.
    """

    instance_mask = np.asarray(
        instance_mask,
        dtype=np.int32
    )

    cell = (
        instance_mask > 0
    ).astype(
        np.uint8
    )

    target = np.zeros(
        cell.shape,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # CLASS 1 = COMPLETE CELL
    # --------------------------------------------------------

    target[
        cell > 0
    ] = 1

    # --------------------------------------------------------
    # CLASS 2 = OUTER BOUNDARY RING
    #
    # Dilate the complete cell outward.
    # Only pixels outside the cell become boundary.
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    dilated = cv2.dilate(
        cell,
        kernel,
        iterations=1
    )

    outer_boundary = (
        (dilated > 0) &
        (cell == 0)
    )

    target[
        outer_boundary
    ] = 2

    return target


def main():

    print()
    print("=" * 72)
    print("CORRECTED MICROGLIA V2 DATASET PREPARATION")
    print("=" * 72)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            IMAGE_PATH
        )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            MASK_PATH
        )

    images = tifffile.imread(
        IMAGE_PATH
    )

    masks = tifffile.imread(
        MASK_PATH
    )

    print(
        "Images:",
        images.shape
    )

    print(
        "Masks :",
        masks.shape
    )

    usable = min(
        images.shape[0],
        masks.shape[0]
    )

    print(
        "Using:",
        usable,
        "frames"
    )

    for frame_index in range(
        usable
    ):

        image = normalize(
            images[frame_index]
        )

        instance_mask = masks[
            frame_index
        ]

        target = make_target(
            instance_mask
        )

        image_file = (
            IMAGE_OUT
            / f"microglia_{frame_index:04d}.png"
        )

        mask_file = (
            MASK_OUT
            / f"microglia_{frame_index:04d}.png"
        )

        if not cv2.imwrite(
            str(image_file),
            image
        ):
            raise RuntimeError(
                f"Failed to save {image_file}"
            )

        if not cv2.imwrite(
            str(mask_file),
            target
        ):
            raise RuntimeError(
                f"Failed to save {mask_file}"
            )

        background = int(
            np.sum(
                target == 0
            )
        )

        complete_cell = int(
            np.sum(
                target == 1
            )
        )

        outer_boundary = int(
            np.sum(
                target == 2
            )
        )

        original_cell_pixels = int(
            np.sum(
                instance_mask > 0
            )
        )

        print()
        print(
            f"Frame {frame_index}"
        )

        print(
            "  Original cell pixels :",
            original_cell_pixels
        )

        print(
            "  Class 1 complete cell:",
            complete_cell
        )

        print(
            "  Class 2 outer ring   :",
            outer_boundary
        )

        print(
            "  Background           :",
            background
        )

        if complete_cell != original_cell_pixels:
            raise RuntimeError(
                "ERROR: Class 1 does not preserve "
                "the complete original cell mask."
            )

        unique = np.unique(
            target
        )

        print(
            "  Classes              :",
            unique.tolist()
        )

    print()
    print("=" * 72)
    print("CORRECTED V2 DATASET READY")
    print("=" * 72)

    print(
        "Images:",
        IMAGE_OUT
    )

    print(
        "Masks :",
        MASK_OUT
    )

    print()
    print(
        "0 = background"
    )

    print(
        "1 = COMPLETE microglia"
    )

    print(
        "2 = OUTER boundary ring"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
