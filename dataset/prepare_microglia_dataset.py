from pathlib import Path
import numpy as np
import tifffile
import cv2

BASE = Path("/Users/abhyudaysingh/microglia_dataset")

IMAGE_PATH = BASE / "images" / "microglia_example_5frames.tif"
MASK_PATH = BASE / "masks" / "microglia_example_labels_3frames.tif"

OUT_IMAGE_DIR = BASE / "processed_images"
OUT_MASK_DIR = BASE / "processed_masks"

OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


def normalize_image(image):
    image = np.asarray(image).astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        low = float(image.min())
        high = float(image.max())

    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - low) / (high - low)
    image = np.clip(image, 0.0, 1.0)

    return (image * 255.0).astype(np.uint8)


def instance_to_three_class(instance_mask):
    """
    Convert instance labels to the model target format:

        0 = background
        1 = cell interior
        2 = cell boundary
    """

    instance_mask = np.asarray(
        instance_mask,
        dtype=np.int32
    )

    target = np.zeros(
        instance_mask.shape,
        dtype=np.uint8
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    labels = np.unique(instance_mask)

    for label in labels:

        if label == 0:
            continue

        cell = (
            instance_mask == label
        ).astype(np.uint8)

        if cell.sum() == 0:
            continue

        # One-pixel erosion gives us an interior region.
        eroded = cv2.erode(
            cell,
            kernel,
            iterations=1
        )

        boundary = (
            (cell > 0) &
            (eroded == 0)
        )

        interior = (
            (cell > 0) &
            (eroded > 0)
        )

        target[interior] = 1
        target[boundary] = 2

    return target


def main():

    print()
    print("=" * 70)
    print("MICROGLIA DATASET PREPARATION")
    print("=" * 70)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image file not found:\n{IMAGE_PATH}"
        )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"Mask file not found:\n{MASK_PATH}"
        )

    images = tifffile.imread(
        IMAGE_PATH
    )

    masks = tifffile.imread(
        MASK_PATH
    )

    print("Raw image shape :", images.shape)
    print("Raw mask shape  :", masks.shape)
    print("Image dtype     :", images.dtype)
    print("Mask dtype      :", masks.dtype)

    if images.ndim != 3:
        raise ValueError(
            "Expected image stack with shape T x H x W."
        )

    if masks.ndim != 3:
        raise ValueError(
            "Expected mask stack with shape T x H x W."
        )

    image_frames = images.shape[0]
    mask_frames = masks.shape[0]

    usable_frames = min(
        image_frames,
        mask_frames
    )

    if usable_frames == 0:
        raise RuntimeError(
            "No usable image/mask frames."
        )

    print()
    print("Image frames :", image_frames)
    print("Mask frames  :", mask_frames)
    print("Usable       :", usable_frames)

    print()
    print("Converting labelled masks...")

    for frame_index in range(
        usable_frames
    ):

        image = images[frame_index]
        mask = masks[frame_index]

        if image.shape != mask.shape:
            raise ValueError(
                f"Frame {frame_index} shape mismatch: "
                f"{image.shape} vs {mask.shape}"
            )

        image_uint8 = normalize_image(
            image
        )

        target = instance_to_three_class(
            mask
        )

        image_name = (
            f"microglia_{frame_index:04d}.png"
        )

        mask_name = (
            f"microglia_{frame_index:04d}.png"
        )

        image_out = (
            OUT_IMAGE_DIR /
            image_name
        )

        mask_out = (
            OUT_MASK_DIR /
            mask_name
        )

        ok1 = cv2.imwrite(
            str(image_out),
            image_uint8
        )

        ok2 = cv2.imwrite(
            str(mask_out),
            target
        )

        if not ok1 or not ok2:
            raise RuntimeError(
                f"Could not write frame {frame_index}"
            )

        unique_values = np.unique(
            target
        )

        instance_count = len(
            np.unique(mask)
        ) - (
            1 if 0 in np.unique(mask)
            else 0
        )

        print(
            f"[OK] Frame {frame_index} | "
            f"instances={instance_count} | "
            f"classes={unique_values.tolist()}"
        )

    print()
    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)
    print(
        "Processed images :",
        OUT_IMAGE_DIR
    )
    print(
        "Processed masks  :",
        OUT_MASK_DIR
    )
    print()
    print(
        "Target encoding:"
    )
    print(
        "0 = background"
    )
    print(
        "1 = microglia interior"
    )
    print(
        "2 = microglia boundary"
    )
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
