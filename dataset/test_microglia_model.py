from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


BASE = Path(
    "/Users/abhyudaysingh/microglia_dataset"
)

IMAGE_DIR = BASE / "processed_images"

MASK_DIR = BASE / "processed_masks"

MODEL_PATH = (
    BASE
    / "checkpoints"
    / "microglia_instance_unet.pth"
)

OUTPUT_DIR = (
    BASE / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):
        return self.conv(x)


class InstanceUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.down1 = DoubleConv(
            1,
            64
        )

        self.down2 = DoubleConv(
            64,
            128
        )

        self.down3 = DoubleConv(
            128,
            256
        )

        self.bottleneck = DoubleConv(
            256,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.conv3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.conv2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.conv1 = DoubleConv(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            3,
            1
        )

    def forward(self, x):

        x1 = self.down1(x)

        x2 = self.pool(x1)
        x2 = self.down2(x2)

        x3 = self.pool(x2)
        x3 = self.down3(x3)

        x4 = self.pool(x3)
        x4 = self.bottleneck(x4)

        x = self.up3(x4)

        x = torch.cat(
            [x3, x],
            dim=1
        )

        x = self.conv3(x)

        x = self.up2(x)

        x = torch.cat(
            [x2, x],
            dim=1
        )

        x = self.conv2(x)

        x = self.up1(x)

        x = torch.cat(
            [x1, x],
            dim=1
        )

        x = self.conv1(x)

        return self.final(x)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{MODEL_PATH}"
    )


model = InstanceUNet().to(
    DEVICE
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


print()
print("=" * 70)
print("MICROGLIA MODEL SANITY TEST")
print("=" * 70)
print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("=" * 70)


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

image_files = sorted(
    IMAGE_DIR.glob("*.png")
)

if not image_files:

    raise RuntimeError(
        "No processed images found."
    )


for image_path in image_files:

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "[ERROR] Could not read",
            image_path.name
        )

        continue

    original = image.copy()

    image_tensor = (
        image.astype(
            np.float32
        ) / 255.0
    )

    image_tensor = torch.from_numpy(
        image_tensor
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        )[0].cpu().numpy()

    background = (
        prediction == 0
    )

    interior = (
        prediction == 1
    )

    boundary = (
        prediction == 2
    )

    cell_mask = (
        interior |
        boundary
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cell_mask,
            connectivity=8
        )
    )

    objects = []

    for label_id in range(
        1,
        num_labels
    ):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )

        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP
            ]
        )

        w = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT
            ]
        )

        # Ignore tiny noise fragments.
        if area < 50:
            continue

        objects.append(
            {
                "label": label_id,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    # --------------------------------------------------------
    # COLOR VISUALIZATION
    #
    # background = original grayscale
    # interior   = green
    # boundary   = red
    # detected objects get blue boxes
    # --------------------------------------------------------

    overlay = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    # Interior in cyan/green.
    overlay[
        interior
    ] = (
        60,
        210,
        170
    )

    # Boundary in red.
    overlay[
        boundary
    ] = (
        80,
        90,
        240
    )

    # Blend with original.
    overlay = cv2.addWeighted(
        cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2BGR
        ),
        0.45,
        overlay,
        0.55,
        0
    )

    # Draw connected objects.
    for index, obj in enumerate(
        objects,
        start=1
    ):

        x = obj["x"]
        y = obj["y"]
        w = obj["w"]
        h = obj["h"]

        cv2.rectangle(
            overlay,
            (
                x,
                y
            ),
            (
                x + w,
                y + h
            ),
            (
                255,
                170,
                60
            ),
            1
        )

        cv2.putText(
            overlay,
            str(index),
            (
                x,
                max(
                    12,
                    y - 3
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (
                255,
                220,
                100
            ),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_name = (
        image_path.stem
        + "_microglia_test.png"
    )

    output_path = (
        OUTPUT_DIR /
        output_name
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    print()
    print(
        image_path.name
    )

    print(
        "  Predicted background:",
        f"{background.sum():,}"
    )

    print(
        "  Predicted interior :",
        f"{interior.sum():,}"
    )

    print(
        "  Predicted boundary :",
        f"{boundary.sum():,}"
    )

    print(
        "  Connected objects  :",
        len(objects)
    )

    if objects:

        areas = [
            obj["area"]
            for obj in objects
        ]

        print(
            "  Smallest object    :",
            min(areas),
            "px"
        )

        print(
            "  Largest object     :",
            max(areas),
            "px"
        )

        print(
            "  Mean object area   :",
            f"{np.mean(areas):.1f} px"
        )

    print(
        "  Output             :",
        output_path
    )


print()
print("=" * 70)
print("SANITY TEST COMPLETE")
print("=" * 70)
print()
