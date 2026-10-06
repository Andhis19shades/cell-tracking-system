from pathlib import Path
import random
import csv

import cv2
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/Users/abhyudaysingh/microglia_dataset"
)

IMAGE_DIR = (
    BASE_DIR /
    "v2_corrected_images"
)

MASK_DIR = (
    BASE_DIR /
    "v2_corrected_masks"
)

CHECKPOINT_DIR = (
    BASE_DIR /
    "checkpoints"
)

VALIDATION_DIR = (
    BASE_DIR /
    "validation_v2"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_SIZE = 512

EPOCHS = 30

BATCH_SIZE = 1

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-5

VALIDATION_FRACTION = 0.33

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
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
# Same architecture as the existing U-Net.
# ============================================================

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
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x
    ):

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

    def forward(
        self,
        x
    ):

        x1 = self.down1(x)

        x2 = self.pool(x1)
        x2 = self.down2(x2)

        x3 = self.pool(x2)
        x3 = self.down3(x3)

        x4 = self.pool(x3)
        x4 = self.bottleneck(x4)

        x = self.up3(x4)

        x = torch.cat(
            [
                x3,
                x
            ],
            dim=1
        )

        x = self.conv3(x)

        x = self.up2(x)

        x = torch.cat(
            [
                x2,
                x
            ],
            dim=1
        )

        x = self.conv2(x)

        x = self.up1(x)

        x = torch.cat(
            [
                x1,
                x
            ],
            dim=1
        )

        x = self.conv1(x)

        return self.final(x)


# ============================================================
# DATASET
# ============================================================

class MicrogliaDataset(
    Dataset
):

    def __init__(
        self,
        pairs,
        augment=False
    ):

        self.pairs = pairs

        self.augment = augment

    def __len__(
        self
    ):

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
                f"Could not read image: "
                f"{image_path}"
            )

        if mask is None:

            raise RuntimeError(
                f"Could not read mask: "
                f"{mask_path}"
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

        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------

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

            rotation = random.randint(
                0,
                3
            )

            if rotation:

                image = np.rot90(
                    image,
                    rotation
                ).copy()

                mask = np.rot90(
                    mask,
                    rotation
                ).copy()

            # Mild contrast variation.

            if random.random() < 0.5:

                alpha = random.uniform(
                    0.85,
                    1.15
                )

                beta = random.uniform(
                    -10,
                    10
                )

                image = cv2.convertScaleAbs(
                    image,
                    alpha=alpha,
                    beta=beta
                )

            # Mild noise.

            if random.random() < 0.25:

                noise = np.random.normal(
                    0,
                    2.5,
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
            ) /
            255.0
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

def dice_scores(
    logits,
    targets
):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    values = []

    for class_id in range(3):

        predicted = (
            predictions ==
            class_id
        )

        actual = (
            targets ==
            class_id
        )

        intersection = (
            predicted &
            actual
        ).sum().float()

        denominator = (
            predicted.sum() +
            actual.sum()
        ).float()

        if denominator.item() == 0:

            score = torch.tensor(
                1.0,
                device=logits.device
            )

        else:

            score = (
                2.0 *
                intersection /
                denominator
            )

        values.append(
            score
        )

    return torch.stack(
        values
    )


# ============================================================
# IOU
# ============================================================

def iou_scores(
    logits,
    targets
):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    values = []

    for class_id in range(3):

        predicted = (
            predictions ==
            class_id
        )

        actual = (
            targets ==
            class_id
        )

        intersection = (
            predicted &
            actual
        ).sum().float()

        union = (
            predicted |
            actual
        ).sum().float()

        if union.item() == 0:

            score = torch.tensor(
                1.0,
                device=logits.device
            )

        else:

            score = (
                intersection /
                union
            )

        values.append(
            score
        )

    return torch.stack(
        values
    )


# ============================================================
# SOFT DICE LOSS
# ============================================================

def soft_dice_loss(
    logits,
    targets
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    losses = []

    for class_id in range(3):

        probability = probabilities[
            :,
            class_id
        ]

        target = (
            targets ==
            class_id
        ).float()

        intersection = (
            probability *
            target
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
            2.0 *
            intersection +
            1e-6
        ) / (
            denominator +
            1e-6
        )

        losses.append(
            1.0 - dice
        )

    losses = torch.stack(
        losses,
        dim=1
    )

    return losses.mean()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    losses = []

    dice_list = []

    iou_list = []

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

        dice_loss = soft_dice_loss(
            logits,
            masks
        )

        loss = (
            0.70 * ce +
            0.30 * dice_loss
        )

        losses.append(
            loss.item()
        )

        dice_list.append(
            dice_scores(
                logits,
                masks
            ).cpu().numpy()
        )

        iou_list.append(
            iou_scores(
                logits,
                masks
            ).cpu().numpy()
        )

    mean_loss = float(
        np.mean(
            losses
        )
    )

    mean_dice = np.mean(
        np.stack(
            dice_list
        ),
        axis=0
    )

    mean_iou = np.mean(
        np.stack(
            iou_list
        ),
        axis=0
    )

    return (
        mean_loss,
        mean_dice,
        mean_iou
    )


# ============================================================
# VALIDATION VISUALIZATION
# ============================================================

@torch.no_grad()
def save_validation_examples(
    model,
    loader
):

    model.eval()

    saved = 0

    for images, masks in loader:

        images_device = images.to(
            DEVICE
        )

        logits = model(
            images_device
        )

        prediction = torch.argmax(
            logits,
            dim=1
        )[0].cpu().numpy()

        original = (
            images[0]
            .cpu()
            .numpy()[0] *
            255.0
        ).astype(
            np.uint8
        )

        ground_truth = (
            masks[0]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # AI CELL MASK
        # Class 1 = complete cell
        # Class 2 = boundary ring
        # ----------------------------------------------------

        predicted_cell = (
            prediction == 1
        ).astype(
            np.uint8
        ) * 255

        # ----------------------------------------------------
        # COLORS
        # ----------------------------------------------------

        original_rgb = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2BGR
        )

        gt_rgb = original_rgb.copy()
        pred_rgb = original_rgb.copy()

        gt_cell = (
            ground_truth == 1
        )

        gt_boundary = (
            ground_truth == 2
        )

        pred_cell = (
            prediction == 1
        )

        pred_boundary = (
            prediction == 2
        )

        # Ground truth = green
        gt_rgb[
            gt_cell
        ] = (
            50,
            190,
            90
        )

        gt_rgb[
            gt_boundary
        ] = (
            50,
            220,
            240
        )

        # Prediction = cyan
        pred_rgb[
            pred_cell
        ] = (
            210,
            190,
            50
        )

        pred_rgb[
            pred_boundary
        ] = (
            210,
            70,
            220
        )

        gt_rgb = cv2.addWeighted(
            original_rgb,
            0.40,
            gt_rgb,
            0.60,
            0
        )

        pred_rgb = cv2.addWeighted(
            original_rgb,
            0.40,
            pred_rgb,
            0.60,
            0
        )

        # ----------------------------------------------------
        # DIFFERENCE
        # ----------------------------------------------------

        gt_binary = (
            ground_truth > 0
        )

        pred_binary = (
            prediction == 1
        )

        tp = (
            gt_binary &
            pred_binary
        )

        fp = (
            ~gt_binary &
            pred_binary
        )

        fn = (
            gt_binary &
            ~pred_binary
        )

        diff = np.zeros(
            (
                original.shape[0],
                original.shape[1],
                3
            ),
            dtype=np.uint8
        )

        # Agreement
        diff[tp] = (
            70,
            190,
            90
        )

        # AI extra
        diff[fp] = (
            70,
            80,
            230
        )

        # AI missed
        diff[fn] = (
            230,
            70,
            70
        )

        diff = cv2.addWeighted(
            original_rgb,
            0.25,
            diff,
            0.75,
            0
        )

        # ----------------------------------------------------
        # TITLES
        # ----------------------------------------------------

        def title(
            image,
            text
        ):

            output = image.copy()

            cv2.rectangle(
                output,
                (0, 0),
                (
                    output.shape[1],
                    34
                ),
                (
                    8,
                    15,
                    20
                ),
                -1
            )

            cv2.putText(
                output,
                text,
                (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    240,
                    250,
                    250
                ),
                1,
                cv2.LINE_AA
            )

            return output

        original_view = title(
            original_rgb,
            "ORIGINAL"
        )

        gt_view = title(
            gt_rgb,
            "GROUND TRUTH"
        )

        pred_view = title(
            pred_rgb,
            "V2 PREDICTION"
        )

        diff_view = title(
            diff,
            "CELL MASK DIFFERENCE"
        )

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
            VALIDATION_DIR /
            f"v2_example_{saved}.png"
        )

        cv2.imwrite(
            str(output_path),
            grid
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
    print("MICROGLIA U-NET V2 TRAINING")
    print("=" * 72)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Images:",
        IMAGE_DIR
    )

    print(
        "Masks:",
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
            "No V2 images found."
        )

    if not mask_files:

        raise RuntimeError(
            "No V2 masks found."
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

        if mask_path is not None:

            pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

    if len(pairs) < 2:

        raise RuntimeError(
            "Need at least two image/mask pairs."
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

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = InstanceUNet().to(
        DEVICE
    )

    bacteria_checkpoint = Path(
        "/Users/abhyudaysingh/backend/"
        "instance_unet_model.pth"
    )

    if bacteria_checkpoint.exists():

        print()
        print(
            "Starting from existing "
            "bacteria U-Net weights."
        )

        state = torch.load(
            bacteria_checkpoint,
            map_location=DEVICE
        )

        if (
            isinstance(
                state,
                dict
            )
            and
            "model_state_dict"
            in state
        ):

            state = state[
                "model_state_dict"
            ]

        model.load_state_dict(
            state,
            strict=True
        )

    else:

        print()
        print(
            "Bacteria checkpoint not found."
        )

        print(
            "Using random initialization."
        )

    # --------------------------------------------------------
    # LOSS
    #
    # Boundary is still a minority class, but V2 now gives
    # class 1 the complete microglia structure.
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            0.25,
            1.0,
            2.0
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
        patience=4
    )

    best_loss = float(
        "inf"
    )

    best_path = (
        CHECKPOINT_DIR /
        "microglia_v2.pth"
    )

    history = []

    print()
    print("=" * 72)
    print("TRAINING STARTED")
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

            dice_loss = soft_dice_loss(
                logits,
                masks
            )

            loss = (
                0.70 * ce +
                0.30 * dice_loss
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
        ) = evaluate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(
            validation_loss
        )

        learning_rate = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        mean_dice = float(
            np.mean(
                validation_dice
            )
        )

        mean_iou = float(
            np.mean(
                validation_iou
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "background_dice": float(
                    validation_dice[0]
                ),
                "cell_dice": float(
                    validation_dice[1]
                ),
                "boundary_dice": float(
                    validation_dice[2]
                ),
                "mean_dice": mean_dice,
                "background_iou": float(
                    validation_iou[0]
                ),
                "cell_iou": float(
                    validation_iou[1]
                ),
                "boundary_iou": float(
                    validation_iou[2]
                ),
                "mean_iou": mean_iou,
                "learning_rate": learning_rate
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.5f} | "
            f"val={validation_loss:.5f} | "
            f"cellDice={validation_dice[1]:.4f} | "
            f"boundaryDice={validation_dice[2]:.4f} | "
            f"meanDice={mean_dice:.4f} | "
            f"meanIoU={mean_iou:.4f} | "
            f"lr={learning_rate:.2e}"
        )

        if validation_loss < best_loss:

            best_loss = (
                validation_loss
            )

            torch.save(
                model.state_dict(),
                best_path
            )

            print(
                "  -> saved best V2 model"
            )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_path = (
        CHECKPOINT_DIR /
        "microglia_v2_history.csv"
    )

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
    # LOAD BEST
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=DEVICE
        )
    )

    save_validation_examples(
        model,
        validation_loader
    )

    (
        final_loss,
        final_dice,
        final_iou
    ) = evaluate(
        model,
        validation_loader,
        criterion
    )

    print()
    print("=" * 72)
    print("V2 TRAINING COMPLETE")
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
        "Cell Dice:",
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
        "Cell IoU:",
        f"{final_iou[1]:.4f}"
    )

    print(
        "Mean IoU:",
        f"{np.mean(final_iou):.4f}"
    )

    print()
    print(
        "Validation visuals:",
        VALIDATION_DIR
    )

    print(
        "Training history:",
        history_path
    )

    print("=" * 72)

    print()
    print(
        "NOTE:"
    )

    print(
        "Only three labelled frames are available. "
        "This is still a pipeline/domain-adaptation test, "
        "not a validated general-purpose microglia model."
    )

    print()


if __name__ == "__main__":
    main()
