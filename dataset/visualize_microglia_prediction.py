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

OUTPUT_DIR = BASE / "validation_visuals"

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


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.conv(x)


class InstanceUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

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
        x = torch.cat([x3, x], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv1(x)

        return self.final(x)


model = InstanceUNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


image_files = sorted(
    IMAGE_DIR.glob("*.png")
)

print()
print("=" * 70)
print("MICROGLIA VISUAL VALIDATION")
print("=" * 70)
print("Device:", DEVICE)
print("=" * 70)


for image_path in image_files:

    mask_path = (
        MASK_DIR /
        image_path.name
    )

    if not mask_path.exists():
        continue

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    ground_truth = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue

    if ground_truth is None:
        continue

    tensor = (
        image.astype(
            np.float32
        ) / 255.0
    )

    tensor = torch.from_numpy(
        tensor
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        prediction = torch.argmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    # ========================================================
    # CREATE RGB BASE
    # ========================================================

    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    # ========================================================
    # GROUND TRUTH COLOR
    #
    # interior = green
    # boundary = yellow
    # ========================================================

    gt_color = original_rgb.copy()

    gt_interior = (
        ground_truth == 1
    )

    gt_boundary = (
        ground_truth == 2
    )

    gt_color[
        gt_interior
    ] = (
        40,
        180,
        80
    )

    gt_color[
        gt_boundary
    ] = (
        60,
        220,
        240
    )

    gt_overlay = cv2.addWeighted(
        original_rgb,
        0.45,
        gt_color,
        0.55,
        0
    )

    # ========================================================
    # AI PREDICTION COLOR
    #
    # interior = cyan
    # boundary = magenta
    # ========================================================

    pred_color = original_rgb.copy()

    pred_interior = (
        prediction == 1
    )

    pred_boundary = (
        prediction == 2
    )

    pred_color[
        pred_interior
    ] = (
        200,
        210,
        50
    )

    pred_color[
        pred_boundary
    ] = (
        220,
        70,
        210
    )

    pred_overlay = cv2.addWeighted(
        original_rgb,
        0.45,
        pred_color,
        0.55,
        0
    )

    # ========================================================
    # DIFFERENCE MAP
    # ========================================================

    gt_cell = (
        ground_truth > 0
    )

    pred_cell = (
        prediction > 0
    )

    difference = np.zeros(
        (
            image.shape[0],
            image.shape[1],
            3
        ),
        dtype=np.uint8
    )

    true_positive = (
        gt_cell &
        pred_cell
    )

    false_positive = (
        ~gt_cell &
        pred_cell
    )

    false_negative = (
        gt_cell &
        ~pred_cell
    )

    difference[
        true_positive
    ] = (
        80,
        190,
        80
    )

    difference[
        false_positive
    ] = (
        70,
        70,
        230
    )

    difference[
        false_negative
    ] = (
        230,
        80,
        80
    )

    difference = cv2.addWeighted(
        original_rgb,
        0.25,
        difference,
        0.75,
        0
    )

    # ========================================================
    # LABELS
    # ========================================================

    def add_title(
        image,
        title
    ):

        output = image.copy()

        cv2.rectangle(
            output,
            (0, 0),
            (
                output.shape[1],
                34
            ),
            (8, 15, 20),
            -1
        )

        cv2.putText(
            output,
            title,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 250, 250),
            1,
            cv2.LINE_AA
        )

        return output

    original_view = add_title(
        original_rgb,
        "ORIGINAL MICROGLIA"
    )

    gt_view = add_title(
        gt_overlay,
        "GROUND TRUTH"
    )

    pred_view = add_title(
        pred_overlay,
        "AI PREDICTION"
    )

    diff_view = add_title(
        difference,
        "PREDICTION DIFFERENCE"
    )

    # ========================================================
    # 2 x 2 GRID
    # ========================================================

    row1 = np.hstack(
        [
            original_view,
            gt_view
        ]
    )

    row2 = np.hstack(
        [
            pred_view,
            diff_view
        ]
    )

    grid = np.vstack(
        [
            row1,
            row2
        ]
    )

    output_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_comparison.png"
    )

    cv2.imwrite(
        str(output_path),
        grid
    )

    # ========================================================
    # OBJECT COUNTS
    # ========================================================

    predicted_binary = (
        pred_cell.astype(np.uint8)
        * 255
    )

    gt_binary = (
        gt_cell.astype(np.uint8)
        * 255
    )

    predicted_count = (
        cv2.connectedComponents(
            predicted_binary,
            connectivity=8
        )[0] - 1
    )

    gt_count = (
        cv2.connectedComponents(
            gt_binary,
            connectivity=8
        )[0] - 1
    )

    intersection = np.logical_and(
        gt_cell,
        pred_cell
    ).sum()

    union = np.logical_or(
        gt_cell,
        pred_cell
    ).sum()

    iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    dice = (
        2 * intersection /
        (
            gt_cell.sum() +
            pred_cell.sum()
        )
        if (
            gt_cell.sum() +
            pred_cell.sum()
        ) > 0
        else 0.0
    )

    print()
    print(
        image_path.name
    )

    print(
        "  Ground-truth objects:",
        gt_count
    )

    print(
        "  Predicted objects    :",
        predicted_count
    )

    print(
        "  Binary Dice          :",
        f"{dice:.4f}"
    )

    print(
        "  Binary IoU           :",
        f"{iou:.4f}"
    )

    print(
        "  Saved                :",
        output_path
    )


print()
print("=" * 70)
print("VISUAL VALIDATION COMPLETE")
print("=" * 70)
print()
