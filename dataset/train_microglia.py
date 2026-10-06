from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/Users/abhyudaysingh/microglia_dataset"
)

IMAGE_DIR = BASE_DIR / "processed_images"
MASK_DIR = BASE_DIR / "processed_masks"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
VALIDATION_DIR = BASE_DIR / "validation"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 512

EPOCHS = 25

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

VALIDATION_FRACTION = 0.33

RANDOM_SEED = 42

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# MODEL
# Same architecture as the current Instance U-Net.
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
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
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
            kernel_size=2,
            stride=2
        )

        self.conv3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.conv2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv1 = DoubleConv(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            3,
            kernel_size=1
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
# DATASET
# ============================================================

class MicrogliaDataset(Dataset):

    def __init__(
        self,
        pairs,
        augment=False
    ):

        self.pairs = pairs

        self.augment = augment

    def __len__(self):

        return len(
            self.pairs
        )

    def __getitem__(
        self,
        index
    ):

        image_path, mask_path = (
            self.pairs[index]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        image = cv2.resize(
            image,
            (
                INPUT_SIZE,
                INPUT_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            (
                INPUT_SIZE,
                INPUT_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        if self.augment:

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                mask = np.fliplr(
                    mask
                ).copy()

            if random.random() < 0.5:

                image = np.flipud(
                    image
                ).copy()

                mask = np.flipud(
                    mask
                ).copy()

            rotations = random.randint(
                0,
                3
            )

            if rotations:

                image = np.rot90(
                    image,
                    rotations
                ).copy()

                mask = np.rot90(
                    mask,
                    rotations
                ).copy()

            if random.random() < 0.25:

                noise = np.random.normal(
                    0,
                    4,
                    image.shape
                )

                image = np.clip(
                    image.astype(
                        np.float32
                    ) + noise,
                    0,
                    255
                ).astype(
                    np.uint8
                )

        image = (
            image.astype(
                np.float32
            ) / 255.0
        )

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask.astype(
                np.int64
            )
        )

        return image, mask


# ============================================================
# DICE
# ============================================================

def dice_score(
    logits,
    targets,
    num_classes=3,
):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    scores = []

    for class_id in range(
        num_classes
    ):

        pred = (
            predictions ==
            class_id
        )

        true = (
            targets ==
            class_id
        )

        intersection = (
            pred & true
        ).sum().float()

        denominator = (
            pred.sum() +
            true.sum()
        ).float()

        if denominator == 0:

            scores.append(
                torch.tensor(
                    1.0,
                    device=logits.device
                )
            )

        else:

            scores.append(
                (
                    2.0 *
                    intersection /
                    denominator
                )
            )

    return torch.stack(
        scores
    )


# ============================================================
# IOU
# ============================================================

def iou_score(
    logits,
    targets,
    num_classes=3,
):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    scores = []

    for class_id in range(
        num_classes
    ):

        pred = (
            predictions ==
            class_id
        )

        true = (
            targets ==
            class_id
        )

        intersection = (
            pred & true
        ).sum().float()

        union = (
            pred | true
        ).sum().float()

        if union == 0:

            scores.append(
                torch.tensor(
                    1.0,
                    device=logits.device
                )
            )

        else:

            scores.append(
                intersection /
                union
            )

    return torch.stack(
        scores
    )


# ============================================================
# LOSS
# ============================================================

def multiclass_dice_loss(
    logits,
    targets,
    num_classes=3,
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    losses = []

    for class_id in range(
        num_classes
    ):

        probability = probabilities[
            :, class_id
        ]

        target = (
            targets ==
            class_id
        ).float()

        intersection = (
            probability * target
        ).sum(
            dim=(1, 2)
        )

        denominator = (
            probability.sum(
                dim=(1, 2)
            ) +
            target.sum(
                dim=(1, 2)
            )
        )

        dice = (
            2.0 * intersection +
            1e-6
        ) / (
            denominator +
            1e-6
        )

        losses.append(
            1.0 - dice
        )

    return torch.stack(
        losses,
        dim=1
    ).mean()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    loss_values = []

    dice_values = []

    iou_values = []

    for images, masks in loader:

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        logits = model(
            images
        )

        ce = criterion(
            logits,
            masks
        )

        dice_loss = (
            multiclass_dice_loss(
                logits,
                masks
            )
        )

        loss = (
            0.65 * ce +
            0.35 * dice_loss
        )

        dice = dice_score(
            logits,
            masks
        )

        iou = iou_score(
            logits,
            masks
        )

        loss_values.append(
            loss.item()
        )

        dice_values.append(
            dice.cpu().numpy()
        )

        iou_values.append(
            iou.cpu().numpy()
        )

    mean_loss = float(
        np.mean(
            loss_values
        )
    )

    mean_dice = np.mean(
        np.stack(
            dice_values
        ),
        axis=0
    )

    mean_iou = np.mean(
        np.stack(
            iou_values
        ),
        axis=0
    )

    return (
        mean_loss,
        mean_dice,
        mean_iou
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

@torch.no_grad()
def save_validation_predictions(
    model,
    loader,
):

    model.eval()

    saved = 0

    for images, masks in loader:

        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )[0].cpu().numpy()

        image_out = (
            images[0]
            .cpu()
            .numpy()[0]
            * 255.0
        ).astype(
            np.uint8
        )

        mask_out = masks[
            0
        ].cpu().numpy().astype(
            np.uint8
        )

        pred_out = (
            predictions
            .astype(
                np.uint8
            )
        )

        cv2.imwrite(
            str(
                VALIDATION_DIR /
                "validation_input.png"
            ),
            image_out
        )

        cv2.imwrite(
            str(
                VALIDATION_DIR /
                "validation_ground_truth.png"
            ),
            mask_out
        )

        cv2.imwrite(
            str(
                VALIDATION_DIR /
                "validation_prediction.png"
            ),
            pred_out
        )

        saved += 1

        if saved >= 3:
            break


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("MICROGLIA U-NET FINE-TUNING")
    print("=" * 72)
    print(
        "Device:",
        DEVICE
    )
    print(
        "Image directory:",
        IMAGE_DIR
    )
    print(
        "Mask directory :",
        MASK_DIR
    )
    print("=" * 72)

    image_files = sorted(
        IMAGE_DIR.glob(
            "*.png"
        )
    )

    mask_files = sorted(
        MASK_DIR.glob(
            "*.png"
        )
    )

    if not image_files:
        raise RuntimeError(
            "No processed microglia images found."
        )

    if not mask_files:
        raise RuntimeError(
            "No processed microglia masks found."
        )

    mask_lookup = {
        path.name: path
        for path in mask_files
    }

    pairs = []

    for image_path in image_files:

        mask_path = mask_lookup.get(
            image_path.name
        )

        if mask_path is None:
            continue

        pairs.append(
            (
                image_path,
                mask_path
            )
        )

    if len(pairs) < 2:

        raise RuntimeError(
            "Need at least two matching image/mask pairs."
        )

    random.shuffle(
        pairs
    )

    validation_count = max(
        1,
        int(
            len(pairs) *
            VALIDATION_FRACTION
        )
    )

    if validation_count >= len(
        pairs
    ):

        validation_count = (
            len(pairs) - 1
        )

    validation_pairs = pairs[
        :validation_count
    ]

    training_pairs = pairs[
        validation_count:
    ]

    print()
    print(
        "Training pairs  :",
        len(training_pairs)
    )

    print(
        "Validation pairs:",
        len(validation_pairs)
    )

    print()

    train_dataset = MicrogliaDataset(
        training_pairs,
        augment=True
    )

    validation_dataset = MicrogliaDataset(
        validation_pairs,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = InstanceUNet().to(
        DEVICE
    )

    # --------------------------------------------------------
    # START FROM BACTERIA MODEL IF AVAILABLE
    # --------------------------------------------------------

    bacteria_model = Path(
        "/Users/abhyudaysingh/backend/instance_unet_model.pth"
    )

    if bacteria_model.exists():

        print()
        print(
            "Found existing bacteria model."
        )

        print(
            "Loading it as the initialization checkpoint..."
        )

        checkpoint = torch.load(
            bacteria_model,
            map_location=DEVICE
        )

        if isinstance(
            checkpoint,
            dict
        ) and "model_state_dict" in checkpoint:

            checkpoint = checkpoint[
                "model_state_dict"
            ]

        missing, unexpected = (
            model.load_state_dict(
                checkpoint,
                strict=False
            )
        )

        print(
            "Missing keys   :",
            len(missing)
        )

        print(
            "Unexpected keys:",
            len(unexpected)
        )

    else:

        print()
        print(
            "No bacteria model found."
        )

        print(
            "Starting from random initialization."
        )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    #
    # Boundary is a smaller class, so weighted CE helps prevent
    # the network from ignoring it.
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            0.25,
            1.0,
            3.0
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_loss = float(
        "inf"
    )

    best_path = (
        CHECKPOINT_DIR /
        "microglia_instance_unet.pth"
    )

    history = []

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_losses = []

        for images, masks in train_loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            ce = criterion(
                logits,
                masks
            )

            dice_loss = (
                multiclass_dice_loss(
                    logits,
                    masks
                )
            )

            loss = (
                0.65 * ce +
                0.35 * dice_loss
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        (
            validation_loss,
            validation_dice,
            validation_iou
        ) = validate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(
            validation_loss
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "background_dice": float(
                    validation_dice[0]
                ),
                "interior_dice": float(
                    validation_dice[1]
                ),
                "boundary_dice": float(
                    validation_dice[2]
                ),
                "mean_dice": float(
                    np.mean(
                        validation_dice
                    )
                ),
                "background_iou": float(
                    validation_iou[0]
                ),
                "interior_iou": float(
                    validation_iou[1]
                ),
                "boundary_iou": float(
                    validation_iou[2]
                ),
                "mean_iou": float(
                    np.mean(
                        validation_iou
                    )
                ),
                "learning_rate": current_lr
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.5f} | "
            f"val={validation_loss:.5f} | "
            f"dice={np.mean(validation_dice):.4f} | "
            f"iou={np.mean(validation_iou):.4f} | "
            f"lr={current_lr:.2e}"
        )

        if validation_loss < best_loss:

            best_loss = validation_loss

            torch.save(
                model.state_dict(),
                best_path
            )

            print(
                "  -> saved best microglia checkpoint"
            )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_path = (
        CHECKPOINT_DIR /
        "training_history.csv"
    )

    import csv

    if history:

        with open(
            history_path,
            "w",
            newline=""
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=history[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                history
            )

    # --------------------------------------------------------
    # RELOAD BEST MODEL
    # --------------------------------------------------------

    if best_path.exists():

        model.load_state_dict(
            torch.load(
                best_path,
                map_location=DEVICE
            )
        )

    save_validation_predictions(
        model,
        validation_loader
    )

    (
        final_loss,
        final_dice,
        final_iou
    ) = validate(
        model,
        validation_loader,
        criterion
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        "Best checkpoint:",
        best_path
    )

    print(
        "Final validation loss:",
        f"{final_loss:.6f}"
    )

    print(
        "Background Dice:",
        f"{final_dice[0]:.4f}"
    )

    print(
        "Interior Dice:",
        f"{final_dice[1]:.4f}"
    )

    print(
        "Boundary Dice:",
        f"{final_dice[2]:.4f}"
    )

    print(
        "Mean Dice:",
        f"{np.mean(final_dice):.4f}"
    )

    print(
        "Mean IoU:",
        f"{np.mean(final_iou):.4f}"
    )

    print()
    print(
        "Validation files:",
        VALIDATION_DIR
    )

    print(
        "Training history:",
        history_path
    )

    print("=" * 72)
    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This checkpoint is a PIPELINE TEST with very limited "
        "labelled data. It must not be treated as a robust "
        "general-purpose microglia model yet."
    )


if __name__ == "__main__":
    main()
