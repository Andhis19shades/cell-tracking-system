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
    / "v2_images"
)

MASK_OUT = (
    BASE
    / "v2_masks"
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

        low = image.min()
        high = image.max()

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
        0,
        1
    )

    return (
        image * 255
    ).astype(
        np.uint8
    )


def make_target(
    instance_mask
):
    """
    Microglia V2 target:

    0 = background
    1 = complete microglia body
    2 = boundary

    Unlike V1, thin processes are NOT removed
    from the body mask by erosion.
    """

    instance_mask = np.asarray(
        instance_mask
    )

    body = (
        instance_mask > 0
    ).astype(
        np.uint8
    )

    # Find one-pixel outer boundaries.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    eroded = cv2.erode(
        body,
        kernel,
        iterations=1
    )

    boundary = (
        (body == 1) &
        (eroded == 0)
    )

    target = np.zeros(
        body.shape,
        dtype=np.uint8
    )

    # IMPORTANT:
    # Whole cell remains class 1.
    target[body == 1] = 1

    # Boundary is explicitly marked as class 2.
    target[boundary] = 2

    return target


def main():

    print()
    print("=" * 70)
    print("MICROGLIA V2 DATASET PREPARATION")
    print("=" * 70)

    images = tifffile.imread(
        IMAGE_PATH
    )

    masks = tifffile.imread(
        MASK_PATH
    )

    usable = min(
        images.shape[0],
        masks.shape[0]
    )

    print(
        "Images:",
        images.shape
    )

    print(
        "Masks :",
        masks.shape
    )

    print(
        "Using:",
        usable,
        "frames"
    )

    for i in range(
        usable
    ):

        image = normalize(
            images[i]
        )

        target = make_target(
            masks[i]
        )

        image_file = (
            IMAGE_OUT /
            f"microglia_{i:04d}.png"
        )

        mask_file = (
            MASK_OUT /
            f"microglia_{i:04d}.png"
        )

        cv2.imwrite(
            str(image_file),
            image
        )

        cv2.imwrite(
            str(mask_file),
            target
        )

        print()
        print(
            f"Frame {i}"
        )

        print(
            "  background:",
            np.sum(target == 0)
        )

        print(
            "  body:",
            np.sum(target == 1)
        )

        print(
            "  boundary:",
            np.sum(target == 2)
        )

    print()
    print("=" * 70)
    print("V2 DATASET READY")
    print("=" * 70)
    print()
    print(
        "Images:",
        IMAGE_OUT
    )
    print(
        "Masks :",
        MASK_OUT
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
