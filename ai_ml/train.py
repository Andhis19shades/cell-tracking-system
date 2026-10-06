import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import UNet
from IDP import CellDataset


# ==========================================
# DEVICE
# ==========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device:", device)


# ==========================================
# DATASET
# ==========================================

dataset = CellDataset(
    "/Users/abhyudaysingh/Downloads/ctc_format/00",
    "/Users/abhyudaysingh/Downloads/ctc_format/00_GT/SEG"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

print("Dataset Loaded")
print("Total Images:", len(dataset))


# ==========================================
# MODEL
# ==========================================

model = UNet().to(device)


# ==========================================
# LOSS
# ==========================================

criterion = nn.BCEWithLogitsLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# TRAINING
# ==========================================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    print(f"\nStarting Epoch {epoch + 1}/{epochs}")

    for batch_number, (images, masks) in enumerate(loader):

        images = images.to(device)

        masks = masks.unsqueeze(1).float().to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, masks)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_number + 1) % 20 == 0:
            print(
                f"Batch {batch_number + 1}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Average Loss: {avg_loss:.4f}"
    )


# ==========================================
# SAVE MODEL
# ==========================================

MODEL_PATH = "/Users/abhyudaysingh/unet_model.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n================================")
print("Training Complete!")
print("Model saved successfully!")
print("Saved at:", MODEL_PATH)
print("================================")