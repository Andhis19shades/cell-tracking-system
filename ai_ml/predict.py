import torch
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np

from model import UNet


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
# PATHS
# ============================================================

MODEL_PATH = "/Users/abhyudaysingh/unet_model.pth"

IMAGE_PATH = (
    "/Users/abhyudaysingh/"
    "Downloads/ctc_format/00/t0000.tif"
)


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


print("Model loaded successfully!")


# ============================================================
# LOAD IMAGE
# ============================================================

image = tiff.imread(IMAGE_PATH)


# ============================================================
# NORMALIZE IMAGE
# ============================================================

image = image.astype(np.float32)

image_min = image.min()
image_max = image.max()

if image_max > image_min:

    image = (
        (image - image_min)
        / (image_max - image_min)
    )

else:

    image = np.zeros_like(image)


# ============================================================
# CONVERT TO TENSOR
# ============================================================

image_tensor = torch.tensor(
    image
).unsqueeze(0).unsqueeze(0)


# ============================================================
# RESIZE
# ============================================================

import torchvision.transforms.functional as TF

image_tensor = TF.resize(
    image_tensor,
    [512, 512]
)


# ============================================================
# MOVE TO DEVICE
# ============================================================

image_tensor = image_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(image_tensor)

    probability = torch.sigmoid(output)

    prediction = (
        probability > 0.5
    ).float()


# ============================================================
# CONVERT BACK TO NUMPY
# ============================================================

prediction = prediction.squeeze().cpu().numpy()

display_image = image_tensor.squeeze().cpu().numpy()


# ============================================================
# OVERLAY
# ============================================================

overlay = np.stack(
    [
        display_image,
        display_image,
        display_image
    ],
    axis=-1
)

# Make predicted cells red
overlay[prediction == 1, 0] = 1.0
overlay[prediction == 1, 1] = 0.0
overlay[prediction == 1, 2] = 0.0


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    display_image,
    cmap="gray"
)

plt.title("Original Image")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    prediction,
    cmap="gray"
)

plt.title("AI Prediction")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(overlay)

plt.title("Overlay")

plt.axis("off")


plt.tight_layout()

plt.show()