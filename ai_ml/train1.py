import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import UNet
from IDP import CellDataset

# Dataset
dataset = CellDataset(
    "/Users/abhyudaysingh/Downloads/ctc_format/00",
    "/Users/abhyudaysingh/Downloads/ctc_format/00_GT/SEG"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# Model
model = UNet()

# Loss Function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Everything Loaded Successfully!")