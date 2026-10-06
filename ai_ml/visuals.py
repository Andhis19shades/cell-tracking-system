import torch
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from pathlib import Path

from model import UNet
from IDP import CellDataset


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = "/Users/abhyudaysingh/Downloads/ctc_format/00"
MASK_DIR = "/Users/abhyudaysingh/Downloads/ctc_format/00_GT/SEG"
MODEL_PATH = "/Users/abhyudaysingh/unet_model.pth"

OUTPUT_DIR = Path("/Users/abhyudaysingh/project_visuals")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device:", device)


# ============================================================
# DATASET
# ============================================================

dataset = CellDataset(
    IMAGE_DIR,
    MASK_DIR
)

print("Total Images:", len(dataset))


# ============================================================
# FIND A GOOD IMAGE FOR VISUALIZATION
# ============================================================

print("Finding a representative image...")

best_index = 0
best_objects = 0

raw_images = sorted(Path(IMAGE_DIR).glob("*.tif"))
raw_masks = sorted(Path(MASK_DIR).glob("*.tif"))

for i in range(len(raw_masks)):

    mask = tiff.imread(raw_masks[i])

    labels = np.unique(mask)
    labels = labels[labels != 0]

    if len(labels) > best_objects:
        best_objects = len(labels)
        best_index = i

print("Selected Image:", best_index)
print("Number of objects:", best_objects)


# ============================================================
# LOAD IMAGE + MASK
# ============================================================

image, binary_mask = dataset[best_index]

image = image.unsqueeze(0).to(device)


# ============================================================
# LOAD RAW INSTANCE MASK
# ============================================================

raw_mask = tiff.imread(raw_masks[best_index])

# Resize raw mask to 512x512 for visualization
raw_mask_tensor = torch.tensor(raw_mask).unsqueeze(0).float()

import torchvision.transforms.functional as TF

raw_mask_resized = TF.resize(
    raw_mask_tensor,
    [512, 512],
    interpolation=TF.InterpolationMode.NEAREST
)

raw_mask_resized = raw_mask_resized.squeeze(0).numpy()


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(image)

    probability = torch.sigmoid(output)

    prediction = (probability > 0.5).float()


prediction = prediction.squeeze().cpu().numpy()

input_image = image.squeeze().cpu().numpy()

binary_mask_np = binary_mask.numpy()


# ============================================================
# TRAINING LOSS VALUES
# ============================================================

epochs = [1, 2, 3, 4, 5]

losses = [
    0.3384,
    0.0415,
    0.0200,
    0.0163,
    0.0157
]


# ============================================================
# FIGURE 1
# DATASET + MASK
# ============================================================

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap="gray")
plt.title("Input Microscopy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(raw_mask_resized, cmap="nipy_spectral")
plt.title("Original CTC Instance Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_mask_np, cmap="gray")
plt.title("Binary Training Mask")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_dataset_preprocessing.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# MODEL PREDICTION
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(prediction, cmap="gray")
plt.title("U-Net Prediction")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(input_image, cmap="gray")
plt.imshow(
    prediction,
    cmap="Reds",
    alpha=0.45
)
plt.title("Prediction Overlay")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_prediction_overlay.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Average BCE Loss")
plt.title("Training Loss Curve")

plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "03_training_loss.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4
# THRESHOLD / PROBABILITY
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap="gray")
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(
    probability.squeeze().cpu().numpy(),
    cmap="viridis",
    vmin=0,
    vmax=1
)
plt.title("Predicted Probability")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap="gray")
plt.title("Thresholded Prediction")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_probability_threshold.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5
# SIMPLE PIPELINE DIAGRAM
# ============================================================

fig, ax = plt.subplots(figsize=(15, 4))

ax.axis("off")

steps = [
    "Raw TIFF\nImages",
    "Normalization\n& Resize",
    "Binary\nMasks",
    "U-Net\nEncoder",
    "Bottleneck",
    "U-Net\nDecoder",
    "Prediction",
    "Overlay"
]

x_positions = np.linspace(0.06, 0.94, len(steps))

for i, (x, text) in enumerate(zip(x_positions, steps)):

    ax.text(
        x,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="black"
        ),
        transform=ax.transAxes
    )

    if i < len(steps) - 1:

        ax.annotate(
            "",
            xy=(x_positions[i + 1] - 0.055, 0.5),
            xytext=(x + 0.055, 0.5),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="->",
                lw=2
            )
        )

plt.title(
    "Current Cell Segmentation Pipeline",
    fontsize=16
)

plt.savefig(
    OUTPUT_DIR / "05_pipeline.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("VISUAL GENERATION COMPLETE")
print("========================================")
print("Images saved to:")
print(OUTPUT_DIR)
print()
print("Generated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file)

print("========================================")