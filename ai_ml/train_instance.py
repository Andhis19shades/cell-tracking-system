import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from IDP_instance import InstanceCellDataset
from model_instance import InstanceUNet


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

IMAGE_DIR = "/Users/abhyudaysingh/Downloads/ctc_format/00"

MASK_DIR = "/Users/abhyudaysingh/Downloads/ctc_format/00_GT/SEG"

MODEL_PATH = "/Users/abhyudaysingh/instance_unet_model.pth"


# ============================================================
# DATASET
# ============================================================

dataset = InstanceCellDataset(
    IMAGE_DIR,
    MASK_DIR
)

print("Dataset Loaded")
print("Total Images:", len(dataset))


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = InstanceUNet().to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# 0 = Background
# 1 = Cell Interior
# 2 = Cell Boundary

class_weights = torch.tensor(
    [0.2, 2.0, 5.0],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

epochs = 5


print()
print("=" * 60)
print("STARTING IMPROVED INSTANCE-AWARE TRAINING")
print("=" * 60)
print()

print("Batch Size:", 8)
print("Epochs:", epochs)
print("Learning Rate:", 0.0005)
print("Class Weights:", [0.2, 2.0, 5.0])
print()


# ============================================================
# TRAIN
# ============================================================

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for batch_index, (images, masks) in enumerate(loader):

        # Move data to device
        images = images.to(device)

        masks = masks.to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Accumulate loss
        # ----------------------------------------------------

        running_loss += loss.item()


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (batch_index + 1) % 20 == 0:

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Batch [{batch_index + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # --------------------------------------------------------
    # Average epoch loss
    # --------------------------------------------------------

    average_loss = (
        running_loss / len(loader)
    )

    print()
    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Average Loss: {average_loss:.4f}"
    )
    print()


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("IMPROVED INSTANCE TRAINING COMPLETE")
print("=" * 60)

print()
print("Model saved successfully!")

print()
print("Saved at:")
print(MODEL_PATH)