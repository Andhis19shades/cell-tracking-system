from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flask import jsonify, request

from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

import tifffile


# ============================================================
# IMPORT THE ORIGINAL WORKING CELL LAB BACKEND
# ============================================================

import app_core


app = app_core.app

jobs = app_core.jobs
jobs_lock = app_core.jobs_lock

BASE_DIR = app_core.BASE_DIR
UPLOAD_DIR = app_core.UPLOAD_DIR
ANALYSIS_DIR = app_core.ANALYSIS_DIR

DEVICE = app_core.DEVICE


# ============================================================
# MICROGLIA CONFIGURATION
# ============================================================

MICROGLIA_DATASET_DIR = (
    Path("/Users/abhyudaysingh/microglia_dataset")
)

MICROGLIA_MODEL_CANDIDATES = [
    MICROGLIA_DATASET_DIR
    / "checkpoints"
    / "microglia_instance_unet.pth",

    MICROGLIA_DATASET_DIR
    / "checkpoints"
    / "microglia_baseline_v1.pth",

    MICROGLIA_DATASET_DIR
    / "checkpoints"
    / "microglia_v2.pth",

    BASE_DIR
    / "microglia_instance_unet.pth",
]


DEFAULT_MICROGLIA_FRAME_STEP = 1

DEFAULT_MICROGLIA_BATCH_SIZE = 2

DEFAULT_MICROGLIA_PIXEL_SIZE_UM = 0.325

DEFAULT_TIP_MATCH_DISTANCE_PX = 45.0

DEFAULT_CELL_MATCH_DISTANCE_PX = 90.0

DEFAULT_MAX_MISSED_FRAMES = 2

DEFAULT_MIN_CELL_AREA_PX = 40

DEFAULT_MIN_SEED_DISTANCE_PX = 18

DEFAULT_MIN_SEED_FRACTION = 0.35

DEFAULT_SHOLL_STEP_PX = 5

DEFAULT_BORDER_MARGIN = 18


# ============================================================
# MICROGLIA MODEL
# ============================================================

MICROGLIA_MODEL = None
MICROGLIA_MODEL_PATH = None
MICROGLIA_MODEL_LOCK = threading.Lock()


def find_microglia_model_path():
    for path in MICROGLIA_MODEL_CANDIDATES:
        if path.exists():
            return path

    return None


def load_microglia_model():
    global MICROGLIA_MODEL
    global MICROGLIA_MODEL_PATH

    with MICROGLIA_MODEL_LOCK:

        if MICROGLIA_MODEL is not None:
            return MICROGLIA_MODEL

        model_path = find_microglia_model_path()

        if model_path is None:
            raise FileNotFoundError(
                "No microglia model was found.\n\n"
                "Expected one of:\n"
                + "\n".join(
                    str(path)
                    for path in MICROGLIA_MODEL_CANDIDATES
                )
            )

        print(
            "[MICROGLIA] Loading model:",
            model_path
        )

        # Re-use the exact U-Net architecture from the
        # working Cell Lab backend.
        model = app_core.InstanceUNet().to(DEVICE)

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

        if (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
        ):
            checkpoint = checkpoint["state_dict"]

        cleaned = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

        model.load_state_dict(
            cleaned,
            strict=True
        )

        model.eval()

        MICROGLIA_MODEL = model
        MICROGLIA_MODEL_PATH = model_path

        print(
            "[MICROGLIA] Model loaded successfully."
        )

        return MICROGLIA_MODEL


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(
    value,
    default=0.0
):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except (
        TypeError,
        ValueError
    ):
        return default


def clean_name(name):

    name = Path(name).name

    safe = "".join(
        character
        if character.isalnum()
        or character in "._-"
        else "_"
        for character in name
    )

    return safe or "microglia_upload"


def clean_stem(name):

    stem = Path(name).stem

    safe = "".join(
        character
        if character.isalnum()
        or character in "_-"
        else "_"
        for character in stem
    )

    return safe or "microglia_analysis"


def clamp(
    value,
    low,
    high
):
    return max(
        low,
        min(
            high,
            value
        )
    )


def euclidean(
    x1,
    y1,
    x2,
    y2
):
    return math.sqrt(
        (
            x2 - x1
        ) ** 2
        +
        (
            y2 - y1
        ) ** 2
    )


def is_near_border(
    x,
    y,
    width,
    height,
    margin=DEFAULT_BORDER_MARGIN
):
    return (
        x <= margin
        or y <= margin
        or x >= width - margin
        or y >= height - margin
    )


# ============================================================
# MICROGLIA IMAGE PREPROCESSING
# ============================================================

def choose_signal_channel(
    frame
):
    """
    Supports:
        grayscale
        RGB/BGR
        fluorescent-looking images
        brightfield-like images

    For colour images, a channel is selected when one channel
    has a substantially stronger intensity distribution.
    Otherwise a normal grayscale conversion is used.
    """

    if frame is None:
        raise ValueError(
            "Received an empty frame."
        )

    if frame.ndim == 2:
        return frame

    if frame.ndim != 3:
        raise ValueError(
            f"Unsupported frame shape: {frame.shape}"
        )

    if frame.shape[2] == 1:
        return frame[:, :, 0]

    channels = [
        frame[:, :, 0],
        frame[:, :, 1],
        frame[:, :, 2],
    ]

    scores = []

    for channel in channels:

        channel_float = (
            channel.astype(
                np.float32
            )
        )

        p50 = np.percentile(
            channel_float,
            50
        )

        p99 = np.percentile(
            channel_float,
            99
        )

        scores.append(
            max(
                0.0,
                p99 - p50
            )
        )

    order = np.argsort(scores)

    strongest = int(order[-1])
    second = int(order[-2])

    strongest_score = scores[
        strongest
    ]

    second_score = scores[
        second
    ]

    # If one channel has substantially more dynamic range,
    # treat it as a likely fluorescence/signal channel.
    if (
        strongest_score > 1.18
        * max(
            second_score,
            1.0
        )
    ):

        return channels[
            strongest
        ]

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


def normalize_microglia_frame(
    frame
):

    signal = choose_signal_channel(
        frame
    )

    signal = signal.astype(
        np.float32
    )

    low = np.percentile(
        signal,
        1
    )

    high = np.percentile(
        signal,
        99
    )

    if high <= low:

        low = float(
            signal.min()
        )

        high = float(
            signal.max()
        )

    if high <= low:

        normalized = np.zeros_like(
            signal,
            dtype=np.uint8
        )

    else:

        normalized = (
            signal - low
        ) / (
            high - low
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0
        )

        normalized = (
            normalized * 255.0
        ).astype(
            np.uint8
        )

    # Mild local contrast normalization.
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    normalized = clahe.apply(
        normalized
    )

    return normalized


# ============================================================
# MICROGLIA MODEL INFERENCE
# ============================================================

def predict_microglia_batch(
    model,
    frames
):

    if not frames:
        return []

    processed = []

    original_sizes = []

    for frame in frames:

        gray = normalize_microglia_frame(
            frame
        )

        original_sizes.append(
            gray.shape
        )

        resized = cv2.resize(
            gray,
            (
                app_core.INPUT_SIZE,
                app_core.INPUT_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )

        resized = (
            resized.astype(
                np.float32
            ) / 255.0
        )

        processed.append(
            resized
        )

    batch = np.stack(
        processed,
        axis=0
    )

    tensor = torch.from_numpy(
        batch
    )

    tensor = tensor.unsqueeze(
        1
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.inference_mode():

        output = model(
            tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction = (
        prediction
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    results = []

    for index, predicted in enumerate(
        prediction
    ):

        height, width = (
            original_sizes[index]
        )

        restored = cv2.resize(
            predicted,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_NEAREST
        )

        results.append(
            restored
        )

    return results


# ============================================================
# SEMANTIC -> INSTANCE MICROGLIA MASK
# ============================================================

def semantic_to_instance_mask(
    prediction
):
    """
    Converts the microglia U-Net semantic output into
    individual cell instances.

    Classes:
        0 = background
        1 = cell signal/interior
        2 = boundary

    The instance step uses distance-transform seeds +
    watershed so that touching microglia are not treated
    as one giant connected component.
    """

    semantic = np.asarray(
        prediction
    )

    cell = (
        (semantic == 1)
        |
        (semantic == 2)
    ).astype(
        np.uint8
    )

    if cell.sum() == 0:
        return np.zeros_like(
            cell,
            dtype=np.int32
        )

    # --------------------------------------------------------
    # Remove isolated single-pixel noise while preserving
    # thin processes.
    # --------------------------------------------------------

    small_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    cell = cv2.morphologyEx(
        cell,
        cv2.MORPH_CLOSE,
        small_kernel,
        iterations=1
    )

    # A very small dilation helps reconnect occasional holes
    # created in thin process predictions.
    cell = cv2.dilate(
        cell,
        small_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Distance transform
    # --------------------------------------------------------

    distance = cv2.distanceTransform(
        cell,
        cv2.DIST_L2,
        5
    )

    if float(
        distance.max()
    ) <= 0:

        labels, _ = cv2.connectedComponents(
            cell,
            connectivity=8
        )

        return labels.astype(
            np.int32
        )

    markers = np.zeros(
        cell.shape,
        dtype=np.int32
    )

    next_marker = 1

    # --------------------------------------------------------
    # Process connected components separately.
    # This keeps a busy image from producing an enormous
    # global marker set.
    # --------------------------------------------------------

    component_count, component_labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            cell,
            connectivity=8
        )
    )

    for component_id in range(
        1,
        component_count
    ):

        component_mask = (
            component_labels
            == component_id
        )

        area = int(
            component_mask.sum()
        )

        if area < DEFAULT_MIN_CELL_AREA_PX:
            continue

        component_distance = np.where(
            component_mask,
            distance,
            0.0
        )

        max_distance = float(
            component_distance.max()
        )

        if max_distance <= 0:
            continue

        seed_threshold = max(
            2.5,
            max_distance
            * DEFAULT_MIN_SEED_FRACTION
        )

        coordinates = peak_local_max(
            component_distance,
            min_distance=DEFAULT_MIN_SEED_DISTANCE_PX,
            threshold_abs=seed_threshold,
            labels=component_mask.astype(
                np.uint8
            ),
            exclude_border=False
        )

        # If no local maximum survives, use the global maximum.
        if len(coordinates) == 0:

            max_location = np.unravel_index(
                np.argmax(
                    component_distance
                ),
                component_distance.shape
            )

            coordinates = np.array(
                [
                    max_location
                ]
            )

        for y, x in coordinates:

            if component_mask[
                y,
                x
            ]:

                markers[
                    y,
                    x
                ] = next_marker

                next_marker += 1

    # --------------------------------------------------------
    # If markers failed, fall back to components.
    # --------------------------------------------------------

    if next_marker == 1:

        labels, _ = cv2.connectedComponents(
            cell,
            connectivity=8
        )

        return labels.astype(
            np.int32
        )

    # --------------------------------------------------------
    # Watershed
    # --------------------------------------------------------

    instance_labels = watershed(
        -distance,
        markers,
        mask=cell.astype(
            bool
        )
    )

    instance_labels = (
        np.asarray(
            instance_labels,
            dtype=np.int32
        )
    )

    # --------------------------------------------------------
    # Remove tiny fragments.
    # --------------------------------------------------------

    cleaned = np.zeros_like(
        instance_labels,
        dtype=np.int32
    )

    next_id = 1

    for label_id in np.unique(
        instance_labels
    ):

        if label_id == 0:
            continue

        area = int(
            np.sum(
                instance_labels
                == label_id
            )
        )

        if area < DEFAULT_MIN_CELL_AREA_PX:
            continue

        cleaned[
            instance_labels == label_id
        ] = next_id

        next_id += 1

    return cleaned


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def find_soma_center(
    cell_mask
):

    distance = cv2.distanceTransform(
        cell_mask.astype(
            np.uint8
        ),
        cv2.DIST_L2,
        5
    )

    minimum, maximum, minimum_location, maximum_location = (
        cv2.minMaxLoc(
            distance
        )
    )

    if maximum > 0:

        return (
            float(
                maximum_location[0]
            ),
            float(
                maximum_location[1]
            ),
            float(maximum)
        )

    ys, xs = np.where(
        cell_mask
    )

    if len(xs) == 0:

        return (
            0.0,
            0.0,
            0.0
        )

    return (
        float(
            xs.mean()
        ),
        float(
            ys.mean()
        ),
        0.0
    )


def circularity(
    area,
    perimeter
):

    if perimeter <= 0:
        return 0.0

    value = (
        4.0
        * math.pi
        * area
        / (
            perimeter
            * perimeter
        )
    )

    return float(
        clamp(
            value,
            0.0,
            1.0
        )
    )


def count_skeleton_clusters(
    binary_skeleton
):

    if not np.any(
        binary_skeleton
    ):
        return 0

    number, _, _, _ = (
        cv2.connectedComponentsWithStats(
            binary_skeleton.astype(
                np.uint8
            ),
            connectivity=8
        )
    )

    return max(
        0,
        int(number - 1)
    )


def skeleton_length_pixels(
    skeleton
):

    ys, xs = np.where(
        skeleton
    )

    if len(xs) == 0:
        return 0.0

    coordinates = set(
        zip(
            xs.tolist(),
            ys.tolist()
        )
    )

    diagonal = math.sqrt(
        2.0
    )

    length = 0.0

    for x, y in coordinates:

        if (
            x + 1,
            y
        ) in coordinates:

            length += 1.0

        if (
            x,
            y + 1
        ) in coordinates:

            length += 1.0

        if (
            x + 1,
            y + 1
        ) in coordinates:

            length += diagonal

        if (
            x - 1,
            y + 1
        ) in coordinates:

            length += diagonal

    return float(
        length
    )


# ============================================================
# SHOLL
# ============================================================

def sholl_profile(
    skeleton,
    soma_x,
    soma_y,
    pixel_size_um
):

    ys, xs = np.where(
        skeleton
    )

    if len(xs) == 0:

        return []

    dx = (
        xs.astype(
            np.float64
        )
        - soma_x
    )

    dy = (
        ys.astype(
            np.float64
        )
        - soma_y
    )

    radial_distance = np.sqrt(
        dx * dx
        +
        dy * dy
    )

    maximum_radius = float(
        radial_distance.max()
    )

    if maximum_radius < 3:
        return []

    profile = []

    radii = np.arange(
        DEFAULT_SHOLL_STEP_PX,
        maximum_radius + 1,
        DEFAULT_SHOLL_STEP_PX
    )

    coordinates = np.column_stack(
        [
            xs,
            ys
        ]
    )

    for radius in radii:

        annulus = (
            np.abs(
                radial_distance
                - radius
            )
            <= 1.5
        )

        points = coordinates[
            annulus
        ]

        if len(points) == 0:

            intersections = 0

        else:

            used = np.zeros(
                len(points),
                dtype=bool
            )

            intersections = 0

            for index in range(
                len(points)
            ):

                if used[
                    index
                ]:
                    continue

                intersections += 1

                delta = (
                    points
                    - points[index]
                )

                distance = np.sqrt(
                    delta[:, 0] ** 2
                    +
                    delta[:, 1] ** 2
                )

                used |= (
                    distance <= 2.5
                )

        profile.append(
            {
                "radius_px":
                    float(radius),

                "radius_um":
                    float(
                        radius
                        * pixel_size_um
                    ),

                "intersections":
                    int(intersections)
            }
        )

    return profile


# ============================================================
# PROCESS ENDPOINTS
# ============================================================

def skeleton_endpoints(
    skeleton
):

    if not np.any(
        skeleton
    ):
        return []

    skel = (
        skeleton.astype(
            np.uint8
        )
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    neighbour_sum = cv2.filter2D(
        skel,
        -1,
        kernel
    )

    neighbours = (
        neighbour_sum
        - skel
    )

    endpoint_pixels = (
        skeleton
        &
        (
            neighbours == 1
        )
    )

    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            endpoint_pixels.astype(
                np.uint8
            ),
            connectivity=8
        )
    )

    endpoints = []

    for component_id in range(
        1,
        number
    ):

        x = float(
            centroids[
                component_id,
                0
            ]
        )

        y = float(
            centroids[
                component_id,
                1
            ]
        )

        endpoints.append(
            {
                "x": x,
                "y": y
            }
        )

    return endpoints


# ============================================================
# MICROGLIA MORPHOLOGY
# ============================================================

def analyze_instance(
    instance_labels,
    cell_id,
    frame_index,
    pixel_size_um
):

    cell = (
        instance_labels
        == cell_id
    ).astype(
        np.uint8
    )

    area_px = int(
        cell.sum()
    )

    if area_px < DEFAULT_MIN_CELL_AREA_PX:
        return None

    contours, _ = cv2.findContours(
        cell,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    perimeter_px = float(
        cv2.arcLength(
            contour,
            True
        )
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area_px = float(
        cv2.contourArea(
            hull
        )
    )

    solidity = (
        area_px
        /
        hull_area_px
        if hull_area_px > 0
        else 0.0
    )

    area_um2 = (
        area_px
        * pixel_size_um
        * pixel_size_um
    )

    perimeter_um = (
        perimeter_px
        * pixel_size_um
    )

    ys, xs = np.where(
        cell
    )

    if len(xs) == 0:
        return None

    centroid_x = float(
        xs.mean()
    )

    centroid_y = float(
        ys.mean()
    )

    (
        soma_x,
        soma_y,
        soma_radius_px
    ) = find_soma_center(
        cell
    )

    # --------------------------------------------------------
    # Shape / ellipse
    # --------------------------------------------------------

    eccentricity = 0.0
    aspect_ratio = 0.0

    major_axis_px = 0.0
    minor_axis_px = 0.0

    if len(contour) >= 5:

        ellipse = cv2.fitEllipse(
            contour
        )

        major_axis_px = max(
            ellipse[1]
        )

        minor_axis_px = min(
            ellipse[1]
        )

        if major_axis_px > 0:

            aspect_ratio = (
                major_axis_px
                /
                max(
                    minor_axis_px,
                    1e-6
                )
            )

            ratio = (
                minor_axis_px
                /
                major_axis_px
            )

            ratio = clamp(
                ratio,
                0.0,
                1.0
            )

            eccentricity = math.sqrt(
                max(
                    0.0,
                    1.0 - ratio * ratio
                )
            )

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    skeleton = skeletonize(
        cell > 0
    )

    skeleton_length_px = (
        skeleton_length_pixels(
            skeleton
        )
    )

    skeleton_length_um = (
        skeleton_length_px
        * pixel_size_um
    )

    skel_uint8 = (
        skeleton.astype(
            np.uint8
        )
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    neighbour_sum = cv2.filter2D(
        skel_uint8,
        -1,
        kernel
    )

    neighbours = (
        neighbour_sum
        - skel_uint8
    )

    branch_pixels = (
        skeleton
        &
        (
            neighbours >= 3
        )
    )

    endpoint_pixels = (
        skeleton
        &
        (
            neighbours == 1
        )
    )

    branch_points = (
        count_skeleton_clusters(
            branch_pixels
        )
    )

    endpoints = (
        count_skeleton_clusters(
            endpoint_pixels
        )
    )

    tips = skeleton_endpoints(
        skeleton
    )

    # Remove endpoints inside the soma region.
    filtered_tips = []

    soma_exclusion_radius = max(
        DEFAULT_MIN_TIP_DISTANCE_PX,
        soma_radius_px * 1.20
    )

    for tip in tips:

        distance = euclidean(
            tip["x"],
            tip["y"],
            soma_x,
            soma_y
        )

        if (
            distance
            >=
            soma_exclusion_radius
        ):

            filtered_tips.append(
                {
                    "x": tip["x"],
                    "y": tip["y"],
                    "distance_from_soma_px":
                        distance
                }
            )

    # --------------------------------------------------------
    # Sholl
    # --------------------------------------------------------

    profile = sholl_profile(
        skeleton,
        soma_x,
        soma_y,
        pixel_size_um
    )

    if profile:

        maximum_sholl = max(
            row[
                "intersections"
            ]
            for row in profile
        )

        critical_row = max(
            profile,
            key=lambda row:
            row[
                "intersections"
            ]
        )

        critical_radius_um = (
            critical_row[
                "radius_um"
            ]
        )

    else:

        maximum_sholl = 0

        critical_radius_um = 0.0

    # --------------------------------------------------------
    # Soma proxy
    # --------------------------------------------------------

    soma_radius_um = (
        soma_radius_px
        * pixel_size_um
    )

    soma_area_proxy_um2 = (
        math.pi
        * soma_radius_um
        * soma_radius_um
    )

    # --------------------------------------------------------
    # Arbor complexity metrics
    # --------------------------------------------------------

    branch_density = (
        branch_points
        /
        max(
            skeleton_length_um,
            1e-6
        )
        * 100.0
    )

    branch_endpoint_ratio = (
        branch_points
        /
        max(
            endpoints,
            1
        )
    )

    return {
        "frame":
            int(frame_index),

        "cell_id":
            int(cell_id),

        "area_px":
            area_px,

        "area_um2":
            float(area_um2),

        "perimeter_px":
            perimeter_px,

        "perimeter_um":
            float(perimeter_um),

        "circularity":
            circularity(
                area_px,
                perimeter_px
            ),

        "solidity":
            float(
                solidity
            ),

        "eccentricity":
            float(
                eccentricity
            ),

        "aspect_ratio":
            float(
                aspect_ratio
            ),

        "major_axis_um":
            float(
                major_axis_px
                * pixel_size_um
            ),

        "minor_axis_um":
            float(
                minor_axis_px
                * pixel_size_um
            ),

        "centroid_x_px":
            centroid_x,

        "centroid_y_px":
            centroid_y,

        "centroid_x_um":
            centroid_x
            * pixel_size_um,

        "centroid_y_um":
            centroid_y
            * pixel_size_um,

        "soma_x_px":
            soma_x,

        "soma_y_px":
            soma_y,

        "soma_x_um":
            soma_x
            * pixel_size_um,

        "soma_y_um":
            soma_y
            * pixel_size_um,

        "soma_radius_um":
            float(
                soma_radius_um
            ),

        "soma_area_proxy_um2":
            float(
                soma_area_proxy_um2
            ),

        "skeleton_length_um":
            float(
                skeleton_length_um
            ),

        "branch_points":
            int(
                branch_points
            ),

        "endpoints":
            int(
                endpoints
            ),

        "branch_density_per_100um":
            float(
                branch_density
            ),

        "branch_endpoint_ratio":
            float(
                branch_endpoint_ratio
            ),

        "max_process_reach_um":
            float(
                max(
                    [
                        tip[
                            "distance_from_soma_px"
                        ]
                        for tip in filtered_tips
                    ],
                    default=0.0
                )
                * pixel_size_um
            ),

        "process_tip_count":
            int(
                len(
                    filtered_tips
                )
            ),

        "sholl_max_intersections":
            int(
                maximum_sholl
            ),

        "sholl_critical_radius_um":
            float(
                critical_radius_um
            ),

        "_skeleton":
            skeleton,

        "_tips":
            filtered_tips,

        "_profile":
            profile,
    }


# ============================================================
# MICROGLIA TRACK CLASS
# ============================================================

class MicrogliaTrack:

    def __init__(
        self,
        track_id,
        detection,
        frame_number,
        fps,
        width,
        height
    ):

        self.track_id = (
            int(track_id)
        )

        self.start_frame = (
            int(frame_number)
        )

        self.last_frame = (
            int(frame_number)
        )

        self.last_detection = (
            detection
        )

        self.missed = 0

        self.width = width
        self.height = height
        self.fps = fps

        self.rows = []

        self.tip_rows = []

        self.max_area_um2 = (
            detection[
                "area_um2"
            ]
        )

        self.start_border = (
            is_near_border(
                detection[
                    "centroid_x_px"
                ],
                detection[
                    "centroid_y_px"
                ],
                width,
                height
            )
        )

        row = self.make_row(
            detection,
            frame_number,
            None
        )

        self.rows.append(
            row
        )

    def make_row(
        self,
        detection,
        frame_number,
        previous
    ):

        time_seconds = (
            frame_number
            /
            self.fps
            if self.fps > 0
            else 0.0
        )

        centroid_displacement_um = 0.0
        centroid_speed_um_min = None

        soma_displacement_um = 0.0
        soma_speed_um_min = None

        area_change_percent = 0.0

        if previous is not None:

            dt_seconds = (
                frame_number
                - previous[
                    "frame"
                ]
            ) / self.fps

            centroid_displacement_px = (
                euclidean(
                    previous[
                        "centroid_x_px"
                    ],
                    previous[
                        "centroid_y_px"
                    ],
                    detection[
                        "centroid_x_px"
                    ],
                    detection[
                        "centroid_y_px"
                    ]
                )
            )

            soma_displacement_px = (
                euclidean(
                    previous[
                        "soma_x_px"
                    ],
                    previous[
                        "soma_y_px"
                    ],
                    detection[
                        "soma_x_px"
                    ],
                    detection[
                        "soma_y_px"
                    ]
                )
            )

            centroid_displacement_um = (
                centroid_displacement_px
                * app_pixel_size(
                    detection
                )
            )

            soma_displacement_um = (
                soma_displacement_px
                * app_pixel_size(
                    detection
                )
            )

            if dt_seconds > 0:

                dt_minutes = (
                    dt_seconds
                    / 60.0
                )

                centroid_speed_um_min = (
                    centroid_displacement_um
                    / dt_minutes
                )

                soma_speed_um_min = (
                    soma_displacement_um
                    / dt_minutes
                )

            previous_area = (
                previous[
                    "area_um2"
                ]
            )

            if previous_area > 0:

                area_change_percent = (
                    (
                        detection[
                            "area_um2"
                        ]
                        -
                        previous_area
                    )
                    /
                    previous_area
                    * 100.0
                )

        return {
            "track_id":
                self.track_id,

            "frame":
                int(frame_number),

            "time_seconds":
                float(time_seconds),

            "centroid_x_px":
                detection[
                    "centroid_x_px"
                ],

            "centroid_y_px":
                detection[
                    "centroid_y_px"
                ],

            "centroid_x_um":
                detection[
                    "centroid_x_um"
                ],

            "centroid_y_um":
                detection[
                    "centroid_y_um"
                ],

            "soma_x_px":
                detection[
                    "soma_x_px"
                ],

            "soma_y_px":
                detection[
                    "soma_y_px"
                ],

            "soma_x_um":
                detection[
                    "soma_x_um"
                ],

            "soma_y_um":
                detection[
                    "soma_y_um"
                ],

            "area_um2":
                detection[
                    "area_um2"
                ],

            "perimeter_um":
                detection[
                    "perimeter_um"
                ],

            "circularity":
                detection[
                    "circularity"
                ],

            "solidity":
                detection[
                    "solidity"
                ],

            "eccentricity":
                detection[
                    "eccentricity"
                ],

            "aspect_ratio":
                detection[
                    "aspect_ratio"
                ],

            "soma_radius_um":
                detection[
                    "soma_radius_um"
                ],

            "soma_area_proxy_um2":
                detection[
                    "soma_area_proxy_um2"
                ],

            "skeleton_length_um":
                detection[
                    "skeleton_length_um"
                ],

            "branch_points":
                detection[
                    "branch_points"
                ],

            "endpoints":
                detection[
                    "endpoints"
                ],

            "branch_density_per_100um":
                detection[
                    "branch_density_per_100um"
                ],

            "max_process_reach_um":
                detection[
                    "max_process_reach_um"
                ],

            "process_tip_count":
                detection[
                    "process_tip_count"
                ],

            "sholl_max_intersections":
                detection[
                    "sholl_max_intersections"
                ],

            "sholl_critical_radius_um":
                detection[
                    "sholl_critical_radius_um"
                ],

            "centroid_displacement_um":
                centroid_displacement_um,

            "centroid_speed_um_min":
                centroid_speed_um_min,

            "soma_displacement_um":
                soma_displacement_um,

            "soma_speed_um_min":
                soma_speed_um_min,

            "area_change_percent":
                float(
                    area_change_percent
                ),

            "start_at_border":
                self.start_border
        }

    def update(
        self,
        detection,
        frame_number
    ):

        previous_detection = (
            self.last_detection
        )

        previous_row = (
            self.rows[-1]
        )

        row = self.make_row(
            detection,
            frame_number,
            previous_row
        )

        self.rows.append(
            row
        )

        self.max_area_um2 = max(
            self.max_area_um2,
            detection[
                "area_um2"
            ]
        )

        self.last_detection = (
            detection
        )

        self.last_frame = (
            int(frame_number)
        )

        self.missed = 0

        # ----------------------------------------------------
        # Process-tip matching
        # ----------------------------------------------------

        previous_tips = (
            previous_detection[
                "_tips"
            ]
        )

        current_tips = (
            detection[
                "_tips"
            ]
        )

        tip_matches = match_tips(
            previous_tips,
            current_tips
        )

        dt_seconds = (
            frame_number
            - previous_row[
                "frame"
            ]
        ) / self.fps

        for tip_index, pair in enumerate(
            tip_matches,
            start=1
        ):

            previous_tip = pair[
                "previous"
            ]

            current_tip = pair[
                "current"
            ]

            displacement_px = (
                euclidean(
                    previous_tip[
                        "x"
                    ],
                    previous_tip[
                        "y"
                    ],
                    current_tip[
                        "x"
                    ],
                    current_tip[
                        "y"
                    ]
                )
            )

            displacement_um = (
                displacement_px
                * app_pixel_size(
                    detection
                )
            )

            previous_radial_px = (
                euclidean(
                    previous_tip[
                        "x"
                    ],
                    previous_tip[
                        "y"
                    ],
                    detection[
                        "soma_x_px"
                    ],
                    detection[
                        "soma_y_px"
                    ]
                )
            )

            current_radial_px = (
                euclidean(
                    current_tip[
                        "x"
                    ],
                    current_tip[
                        "y"
                    ],
                    detection[
                        "soma_x_px"
                    ],
                    detection[
                        "soma_y_px"
                    ]
                )
            )

            radial_change_px = (
                current_radial_px
                - previous_radial_px
            )

            radial_change_um = (
                radial_change_px
                * app_pixel_size(
                    detection
                )
            )

            if radial_change_um > 0.5:

                state = "extension"

            elif radial_change_um < -0.5:

                state = "retraction"

            else:

                state = "stable"

            speed_um_min = None

            if dt_seconds > 0:

                speed_um_min = (
                    displacement_um
                    /
                    (
                        dt_seconds
                        / 60.0
                    )
                )

            self.tip_rows.append(
                {
                    "track_id":
                        self.track_id,

                    "frame_from":
                        int(
                            previous_row[
                                "frame"
                            ]
                        ),

                    "frame_to":
                        int(
                            frame_number
                        ),

                    "tip_pair":
                        int(
                            tip_index
                        ),

                    "previous_tip_x_px":
                        previous_tip[
                            "x"
                        ],

                    "previous_tip_y_px":
                        previous_tip[
                            "y"
                        ],

                    "current_tip_x_px":
                        current_tip[
                            "x"
                        ],

                    "current_tip_y_px":
                        current_tip[
                            "y"
                        ],

                    "displacement_um":
                        float(
                            displacement_um
                        ),

                    "radial_change_um":
                        float(
                            radial_change_um
                        ),

                    "state":
                        state,

                    "speed_um_min":
                        (
                            float(
                                speed_um_min
                            )
                            if speed_um_min
                            is not None
                            else None
                        )
                }
            )


def app_pixel_size(
    detection
):

    return safe_float(
        detection.get(
            "_pixel_size_um",
            DEFAULT_MICROGLIA_PIXEL_SIZE_UM
        ),
        DEFAULT_MICROGLIA_PIXEL_SIZE_UM
    )


# ============================================================
# TIP MATCHING
# ============================================================

def match_tips(
    previous_tips,
    current_tips
):

    if not previous_tips:
        return []

    if not current_tips:
        return []

    distances = []

    for i, previous in enumerate(
        previous_tips
    ):

        row = []

        for j, current in enumerate(
            current_tips
        ):

            distance = euclidean(
                previous[
                    "x"
                ],
                previous[
                    "y"
                ],
                current[
                    "x"
                ],
                current[
                    "y"
                ]
            )

            row.append(
                distance
            )

        distances.append(
            row
        )

    matrix = np.asarray(
        distances,
        dtype=np.float64
    )

    row_indices, col_indices = (
        linear_sum_assignment(
            matrix
        )
    )

    matches = []

    for i, j in zip(
        row_indices,
        col_indices
    ):

        distance = float(
            matrix[
                i,
                j
            ]
        )

        if (
            distance
            <= DEFAULT_TIP_MATCH_DISTANCE_PX
        ):

            matches.append(
                {
                    "previous":
                        previous_tips[
                            i
                        ],

                    "current":
                        current_tips[
                            j
                        ]
                }
            )

    return matches


# ============================================================
# CELL MATCHING
# ============================================================

def cell_match_cost(
    previous,
    current
):

    distance = euclidean(
        previous[
            "soma_x_px"
        ],
        previous[
            "soma_y_px"
        ],
        current[
            "soma_x_px"
        ],
        current[
            "soma_y_px"
        ]
    )

    if distance > (
        DEFAULT_CELL_MATCH_DISTANCE_PX
    ):
        return None

    previous_area = max(
        safe_float(
            previous[
                "area_um2"
            ]
        ),
        1e-6
    )

    current_area = max(
        safe_float(
            current[
                "area_um2"
            ]
        ),
        1e-6
    )

    area_penalty = abs(
        math.log(
            current_area
            /
            previous_area
        )
    )

    distance_penalty = (
        distance
        /
        DEFAULT_CELL_MATCH_DISTANCE_PX
    )

    return (
        distance_penalty
        +
        0.25 * area_penalty
    )


# ============================================================
# MICROGLIA INSTANCE DETECTIONS
# ============================================================

def build_detections(
    prediction,
    pixel_size_um,
    frame_index
):

    instance_labels = (
        semantic_to_instance_mask(
            prediction
        )
    )

    detections = []

    for cell_id in np.unique(
        instance_labels
    ):

        if cell_id == 0:
            continue

        detection = analyze_instance(
            instance_labels,
            int(cell_id),
            frame_index,
            pixel_size_um
        )

        if detection is None:
            continue

        detection[
            "_instance_labels"
        ] = instance_labels

        detection[
            "_pixel_size_um"
        ] = pixel_size_um

        detections.append(
            detection
        )

    return (
        detections,
        instance_labels
    )


# ============================================================
# ANNOTATION
# ============================================================

def annotate_microglia_frame(
    frame,
    prediction,
    detections,
    tracks
):

    if frame.ndim == 2:

        base = cv2.cvtColor(
            frame,
            cv2.COLOR_GRAY2BGR
        )

    else:

        base = frame.copy()

    overlay = base.copy()

    semantic_cell = (
        (prediction == 1)
        |
        (prediction == 2)
    )

    if np.any(
        semantic_cell
    ):

        signal_color = np.zeros_like(
            overlay
        )

        signal_color[
            semantic_cell
        ] = (
            50,
            190,
            210
        )

        overlay = cv2.addWeighted(
            overlay,
            0.72,
            signal_color,
            0.28,
            0
        )

    # --------------------------------------------------------
    # Draw instance contours, IDs and skeletons.
    # --------------------------------------------------------

    for track_id, track in (
        tracks.items()
    ):

        if track.last_frame < 0:
            continue

        detection = (
            track.last_detection
        )

        if detection is None:
            continue

        cell = (
            detection[
                "_instance_labels"
            ]
            ==
            detection[
                "cell_id"
            ]
        )

        contours, _ = cv2.findContours(
            (
                cell.astype(
                    np.uint8
                )
                * 255
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (
                    80,
                    220,
                    150
                ),
                1
            )

        skeleton = detection[
            "_skeleton"
        ]

        overlay[
            skeleton
        ] = (
            190,
            80,
            230
        )

        soma_x = int(
            round(
                detection[
                    "soma_x_px"
                ]
            )
        )

        soma_y = int(
            round(
                detection[
                    "soma_y_px"
                ]
            )
        )

        cv2.circle(
            overlay,
            (
                soma_x,
                soma_y
            ),
            4,
            (
                50,
                190,
                255
            ),
            -1
        )

        cv2.putText(
            overlay,
            f"MG-{track_id}",
            (
                soma_x + 7,
                soma_y - 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                255,
                230,
                100
            ),
            1,
            cv2.LINE_AA
        )

        for tip in detection[
            "_tips"
        ]:

            x = int(
                round(
                    tip["x"]
                )
            )

            y = int(
                round(
                    tip["y"]
                )
            )

            cv2.circle(
                overlay,
                (
                    x,
                    y
                ),
                3,
                (
                    230,
                    190,
                    60
                ),
                -1
            )

    return overlay


# ============================================================
# VIDEO / TIFF INPUT
# ============================================================

def inspect_input(
    path
):

    suffix = (
        path.suffix.lower()
    )

    image_extensions = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    }

    if suffix in image_extensions:

        if suffix in {
            ".tif",
            ".tiff"
        }:

            stack = tifffile.imread(
                path
            )

            if stack.ndim == 2:

                return {
                    "type": "image_stack",
                    "frames": 1,
                    "width": stack.shape[1],
                    "height": stack.shape[0],
                    "fps": 1.0,
                    "duration": 0.0,
                    "stack": stack
                }

            if stack.ndim == 3:

                return {
                    "type": "image_stack",
                    "frames": stack.shape[0],
                    "width": stack.shape[2],
                    "height": stack.shape[1],
                    "fps": 1.0,
                    "duration": 0.0,
                    "stack": stack
                }

            if stack.ndim == 4:

                return {
                    "type": "image_stack",
                    "frames": stack.shape[0],
                    "width": stack.shape[2],
                    "height": stack.shape[1],
                    "fps": 1.0,
                    "duration": 0.0,
                    "stack": stack
                }

            raise RuntimeError(
                f"Unsupported TIFF shape: {stack.shape}"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        if image.ndim == 2:

            height, width = (
                image.shape
            )

        else:

            height, width = (
                image.shape[:2]
            )

        return {
            "type": "single_image",
            "frames": 1,
            "width": width,
            "height": height,
            "fps": 1.0,
            "duration": 0.0,
            "image": image
        }

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {path}"
        )

    fps = safe_float(
        cap.get(
            cv2.CAP_PROP_FPS
        ),
        1.0
    )

    if fps <= 0:
        fps = 1.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    duration = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "type": "video",
        "frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration
    }



# ============================================================
# MICROGLIA GRAPH GENERATION
# ============================================================

def save_microglia_plot(
    plt,
    output_dir,
    filename,
    title,
    xlabel,
    ylabel,
):
    path = output_dir / filename
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path.name


def generate_microglia_graphs(
    frame_df,
    track_df,
    summary_df,
    tip_df,
    output_dir,
):
    """
    Generate a broad Microglia analysis graph package.

    The graph package deliberately contains:
      - temporal morphology trends
      - temporal population trends
      - soma and process motility
      - process extension/retraction
      - cell trajectories
      - morphology-vs-motion relationships
      - semantic segmentation layer composition

    All plots are generated from the same analysis job so the UI can
    render a coherent scientific dashboard rather than unrelated charts.
    """

    graph_files = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --------------------------------------------------------
        # 1. Population over time
        # --------------------------------------------------------
        if not frame_df.empty:
            x = frame_df["time_seconds"].to_numpy(dtype=float)

            fig = plt.figure(figsize=(8.8, 5.0))
            plt.plot(
                x,
                frame_df["cell_count"].to_numpy(dtype=float),
                marker="o",
                linewidth=2,
            )
            graph_files["cell_count_vs_time"] = save_microglia_plot(
                plt,
                output_dir,
                "01_microglia_cell_count_vs_time.png",
                "Microglial Cell Count vs Time",
                "Time (s)",
                "Detected microglia",
            )

            # ----------------------------------------------------
            # 2. Mean cell area
            # ----------------------------------------------------
            if "mean_cell_area_um2" in frame_df.columns:
                fig = plt.figure(figsize=(8.8, 5.0))
                plt.plot(
                    x,
                    frame_df["mean_cell_area_um2"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                )
                graph_files["mean_cell_area_vs_time"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "02_mean_cell_area_vs_time.png",
                    "Mean Microglial Cell Area vs Time",
                    "Time (s)",
                    "Mean cell area (µm²)",
                )

            # ----------------------------------------------------
            # 3. Arbor / skeleton length
            # ----------------------------------------------------
            if "mean_skeleton_length_um" in frame_df.columns:
                fig = plt.figure(figsize=(8.8, 5.0))
                plt.plot(
                    x,
                    frame_df["mean_skeleton_length_um"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                )
                graph_files["arbor_length_vs_time"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "03_arbor_length_vs_time.png",
                    "Microglial Arbor / Skeleton Length",
                    "Time (s)",
                    "Mean arbor length (µm)",
                )

            # ----------------------------------------------------
            # 4. Branch points
            # ----------------------------------------------------
            if "mean_branch_points" in frame_df.columns:
                fig = plt.figure(figsize=(8.8, 5.0))
                plt.plot(
                    x,
                    frame_df["mean_branch_points"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                )
                graph_files["branch_points_vs_time"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "04_branch_points_vs_time.png",
                    "Mean Branch Points vs Time",
                    "Time (s)",
                    "Branch points / cell",
                )

            # ----------------------------------------------------
            # 5. Endpoints
            # ----------------------------------------------------
            if "mean_endpoints" in frame_df.columns:
                fig = plt.figure(figsize=(8.8, 5.0))
                plt.plot(
                    x,
                    frame_df["mean_endpoints"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                )
                graph_files["endpoints_vs_time"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "05_endpoints_vs_time.png",
                    "Mean Process Endpoints vs Time",
                    "Time (s)",
                    "Endpoints / cell",
                )

            # ----------------------------------------------------
            # 6. Soma radius
            # ----------------------------------------------------
            if "mean_soma_radius_um" in frame_df.columns:
                fig = plt.figure(figsize=(8.8, 5.0))
                plt.plot(
                    x,
                    frame_df["mean_soma_radius_um"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                )
                graph_files["soma_radius_vs_time"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "06_soma_radius_vs_time.png",
                    "Mean Soma Radius vs Time",
                    "Time (s)",
                    "Soma radius (µm)",
                )

            # ----------------------------------------------------
            # 7. Relative population
            # ----------------------------------------------------
            counts = frame_df["cell_count"].to_numpy(dtype=float)
            initial = counts[0] if len(counts) else 0.0

            if initial > 0:
                fig = plt.figure(figsize=(8.8, 5.0))
                plt.plot(
                    x,
                    counts / initial,
                    marker="o",
                    linewidth=2,
                )
                graph_files["relative_population"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "07_relative_population_vs_time.png",
                    "Relative Microglial Population",
                    "Time (s)",
                    "Population / initial population",
                )

            # ----------------------------------------------------
            # 8. Segmentation layers
            # ----------------------------------------------------
            layer_columns = {
                "background": "background_percent",
                "microglia_interior": "interior_percent",
                "boundary": "boundary_percent",
            }

            if all(
                column in frame_df.columns
                for column in layer_columns.values()
            ):
                fig = plt.figure(figsize=(9.0, 5.4))

                background = frame_df[
                    "background_percent"
                ].to_numpy(dtype=float)

                interior = frame_df[
                    "interior_percent"
                ].to_numpy(dtype=float)

                boundary = frame_df[
                    "boundary_percent"
                ].to_numpy(dtype=float)

                plt.stackplot(
                    x,
                    background,
                    interior,
                    boundary,
                    labels=[
                        "Background",
                        "Microglia interior",
                        "Boundary",
                    ],
                    alpha=0.78,
                )

                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    ncol=3,
                    frameon=False,
                )

                graph_files["segmentation_layer_composition"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "08_segmentation_layer_composition.png",
                    "Semantic Segmentation Layer Composition",
                    "Time (s)",
                    "Image area (%)",
                )

                # ------------------------------------------------
                # 9. Layer integrity / coverage
                # ------------------------------------------------
                cell_coverage = (
                    interior + boundary
                )

                fig = plt.figure(figsize=(8.8, 5.0))
                plt.plot(
                    x,
                    cell_coverage,
                    marker="o",
                    linewidth=2,
                )
                graph_files["segmentation_coverage"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "09_segmentation_coverage.png",
                    "Predicted Microglia Image Coverage",
                    "Time (s)",
                    "Microglia signal (%)",
                )

            # ----------------------------------------------------
            # 10. Detected cell count vs semantic coverage
            # ----------------------------------------------------
            if (
                "interior_percent" in frame_df.columns
                and "boundary_percent" in frame_df.columns
            ):
                coverage = (
                    frame_df["interior_percent"].to_numpy(dtype=float)
                    + frame_df["boundary_percent"].to_numpy(dtype=float)
                )

                fig = plt.figure(figsize=(8.8, 5.0))
                plt.scatter(
                    coverage,
                    frame_df["cell_count"].to_numpy(dtype=float),
                    s=34,
                    alpha=0.75,
                )
                graph_files["coverage_vs_cell_count"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "10_segmentation_coverage_vs_cell_count.png",
                    "Segmentation Coverage vs Detected Cell Count",
                    "Predicted microglia area (%)",
                    "Detected microglia",
                )

        # --------------------------------------------------------
        # Track-derived graphs
        # --------------------------------------------------------
        if not track_df.empty:
            track_work = track_df.copy()

            # ----------------------------------------------------
            # 11. Soma speed through time
            # ----------------------------------------------------
            if {
                "time_seconds",
                "soma_speed_um_min",
            }.issubset(track_work.columns):

                speed_df = (
                    track_work[
                        [
                            "time_seconds",
                            "soma_speed_um_min",
                        ]
                    ]
                    .dropna()
                    .groupby(
                        "time_seconds",
                        as_index=False,
                    )
                    .mean()
                )

                if not speed_df.empty:
                    fig = plt.figure(figsize=(8.8, 5.0))
                    plt.plot(
                        speed_df["time_seconds"],
                        speed_df["soma_speed_um_min"],
                        marker="o",
                        linewidth=2,
                    )
                    graph_files["soma_speed_vs_time"] = save_microglia_plot(
                        plt,
                        output_dir,
                        "11_soma_speed_vs_time.png",
                        "Mean Soma Speed vs Time",
                        "Time (s)",
                        "Soma speed (µm/min)",
                    )

            # ----------------------------------------------------
            # 12. Area vs arbor length
            # ----------------------------------------------------
            if {
                "area_um2",
                "skeleton_length_um",
            }.issubset(track_work.columns):

                valid = track_work[
                    [
                        "area_um2",
                        "skeleton_length_um",
                    ]
                ].dropna()

                if not valid.empty:
                    fig = plt.figure(figsize=(8.8, 5.0))
                    plt.scatter(
                        valid["area_um2"],
                        valid["skeleton_length_um"],
                        s=18,
                        alpha=0.55,
                    )
                    graph_files["area_vs_arbor_length"] = save_microglia_plot(
                        plt,
                        output_dir,
                        "12_cell_area_vs_arbor_length.png",
                        "Cell Area vs Arbor Length",
                        "Cell area (µm²)",
                        "Skeleton / arbor length (µm)",
                    )

            # ----------------------------------------------------
            # 13. Branching vs arbor length
            # ----------------------------------------------------
            if {
                "skeleton_length_um",
                "branch_points",
            }.issubset(track_work.columns):

                valid = track_work[
                    [
                        "skeleton_length_um",
                        "branch_points",
                    ]
                ].dropna()

                if not valid.empty:
                    fig = plt.figure(figsize=(8.8, 5.0))
                    plt.scatter(
                        valid["skeleton_length_um"],
                        valid["branch_points"],
                        s=18,
                        alpha=0.55,
                    )
                    graph_files["branching_vs_arbor"] = save_microglia_plot(
                        plt,
                        output_dir,
                        "13_branch_points_vs_arbor_length.png",
                        "Branch Points vs Arbor Length",
                        "Arbor length (µm)",
                        "Branch points",
                    )

            # ----------------------------------------------------
            # 14. Cell trajectories
            # ----------------------------------------------------
            if {
                "track_id",
                "centroid_x_um",
                "centroid_y_um",
            }.issubset(track_work.columns):

                fig = plt.figure(figsize=(9.0, 7.2))

                for track_id, group in track_work.groupby(
                    "track_id"
                ):
                    if len(group) > 1:
                        plt.plot(
                            group["centroid_x_um"],
                            group["centroid_y_um"],
                            linewidth=0.85,
                            alpha=0.55,
                        )

                plt.gca().invert_yaxis()

                graph_files["cell_trajectories"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "14_cell_trajectories.png",
                    "Microglial Cell Trajectories",
                    "X (µm)",
                    "Y (µm)",
                )

                # ------------------------------------------------
                # 15. Long-track trajectories
                # ------------------------------------------------
                if not summary_df.empty and "frames_tracked" in summary_df.columns:
                    long_ids = (
                        summary_df
                        .nlargest(
                            min(15, len(summary_df)),
                            "frames_tracked",
                        )["track_id"]
                        .tolist()
                    )

                    fig = plt.figure(figsize=(9.0, 7.2))

                    for track_id in long_ids:
                        group = track_work[
                            track_work["track_id"] == track_id
                        ]

                        if len(group) > 1:
                            plt.plot(
                                group["centroid_x_um"],
                                group["centroid_y_um"],
                                linewidth=1.25,
                            )

                    plt.gca().invert_yaxis()

                    graph_files["long_cell_trajectories"] = save_microglia_plot(
                        plt,
                        output_dir,
                        "15_long_cell_trajectories.png",
                        "Longest Microglial Cell Trajectories",
                        "X (µm)",
                        "Y (µm)",
                    )

            # ----------------------------------------------------
            # 16. Speed distribution
            # ----------------------------------------------------
            if "soma_speed_um_min" in track_work.columns:
                values = (
                    pd.to_numeric(
                        track_work["soma_speed_um_min"],
                        errors="coerce",
                    )
                    .dropna()
                    .to_numpy()
                )

                values = values[values >= 0]

                if len(values):
                    fig = plt.figure(figsize=(8.8, 5.0))
                    plt.hist(
                        values,
                        bins=min(
                            25,
                            max(5, len(values)),
                        ),
                    )
                    graph_files["soma_speed_distribution"] = save_microglia_plot(
                        plt,
                        output_dir,
                        "16_soma_speed_distribution.png",
                        "Soma Speed Distribution",
                        "Soma speed (µm/min)",
                        "Observations",
                    )

            # ----------------------------------------------------
            # 17. Sholl distribution
            # ----------------------------------------------------
            if "sholl_max_intersections" in track_work.columns:
                values = (
                    pd.to_numeric(
                        track_work["sholl_max_intersections"],
                        errors="coerce",
                    )
                    .dropna()
                    .to_numpy()
                )

                if len(values):
                    fig = plt.figure(figsize=(8.8, 5.0))
                    plt.hist(
                        values,
                        bins=min(
                            20,
                            max(5, len(np.unique(values))),
                        ),
                    )
                    graph_files["sholl_distribution"] = save_microglia_plot(
                        plt,
                        output_dir,
                        "17_sholl_complexity_distribution.png",
                        "Sholl Maximum Intersection Distribution",
                        "Maximum Sholl intersections",
                        "Observations",
                    )

        # --------------------------------------------------------
        # Process-tip graphs
        # --------------------------------------------------------
        if not tip_df.empty:
            if "speed_um_min" in tip_df.columns:
                speed_values = (
                    pd.to_numeric(
                        tip_df["speed_um_min"],
                        errors="coerce",
                    )
                    .dropna()
                    .to_numpy()
                )

                speed_values = speed_values[
                    speed_values >= 0
                ]

                if len(speed_values):
                    fig = plt.figure(figsize=(8.8, 5.0))
                    plt.hist(
                        speed_values,
                        bins=min(
                            25,
                            max(5, len(speed_values)),
                        ),
                    )
                    graph_files["process_tip_speed_distribution"] = save_microglia_plot(
                        plt,
                        output_dir,
                        "18_process_tip_speed_distribution.png",
                        "Process-tip Speed Distribution",
                        "Process-tip speed (µm/min)",
                        "Transitions",
                    )

            if "state" in tip_df.columns:
                state_counts = (
                    tip_df["state"]
                    .astype(str)
                    .str.lower()
                    .value_counts()
                )

                fig = plt.figure(figsize=(8.8, 5.0))
                labels = [
                    "Extension",
                    "Retraction",
                    "Stable",
                ]
                values = [
                    int(state_counts.get("extension", 0)),
                    int(state_counts.get("retraction", 0)),
                    int(state_counts.get("stable", 0)),
                ]

                plt.bar(
                    labels,
                    values,
                )

                graph_files["extension_retraction"] = save_microglia_plot(
                    plt,
                    output_dir,
                    "19_process_extension_retraction.png",
                    "Process Extension / Retraction Events",
                    "Process state",
                    "Events",
                )

        # --------------------------------------------------------
        # Fallback graph if an unusually small dataset gives very
        # few eligible plots.
        # --------------------------------------------------------
        if len(graph_files) < 7:
            if not frame_df.empty:
                x = frame_df["time_seconds"].to_numpy(
                    dtype=float
                )

                fallback_columns = [
                    (
                        "cell_count",
                        "Detected Cell Count",
                        "Detected microglia",
                    ),
                    (
                        "mean_cell_area_um2",
                        "Mean Cell Area",
                        "Area (µm²)",
                    ),
                    (
                        "mean_skeleton_length_um",
                        "Mean Arbor Length",
                        "Length (µm)",
                    ),
                    (
                        "mean_branch_points",
                        "Mean Branch Points",
                        "Branch points",
                    ),
                    (
                        "mean_endpoints",
                        "Mean Endpoints",
                        "Endpoints",
                    ),
                    (
                        "mean_soma_radius_um",
                        "Mean Soma Radius",
                        "Radius (µm)",
                    ),
                ]

                for index, (
                    column,
                    title,
                    ylabel,
                ) in enumerate(
                    fallback_columns,
                    start=1,
                ):
                    if column not in frame_df.columns:
                        continue

                    fig = plt.figure(figsize=(8.8, 5.0))
                    plt.plot(
                        x,
                        frame_df[column].to_numpy(
                            dtype=float
                        ),
                        marker="o",
                    )

                    fallback_name = (
                        f"fallback_{index}_{column}.png"
                    )

                    key = (
                        f"fallback_{column}"
                    )

                    graph_files[key] = save_microglia_plot(
                        plt,
                        output_dir,
                        fallback_name,
                        title,
                        "Time (s)",
                        ylabel,
                    )

                    if len(graph_files) >= 7:
                        break

    except Exception as graph_error:
        print(
            "[MICROGLIA] Graph generation warning:",
            graph_error,
        )

    return graph_files



# ============================================================
# MICROGLIA JOB
# ============================================================

def run_microglia_job(
    job_id,
    input_path,
    output_dir,
    process_every_n_frames,
    inference_batch_size,
    pixel_size_um,
    frame_interval_seconds
):

    analysis_start = (
        time.perf_counter()
    )

    try:

        model = load_microglia_model()

        input_info = inspect_input(
            input_path
        )

        total_frames = int(
            input_info[
                "frames"
            ]
        )

        width = int(
            input_info[
                "width"
            ]
        )

        height = int(
            input_info[
                "height"
            ]
        )

        source_fps = safe_float(
            input_info[
                "fps"
            ],
            1.0
        )

        if source_fps <= 0:
            source_fps = 1.0

        if frame_interval_seconds is not None:

            source_dt_seconds = (
                frame_interval_seconds
            )

        else:

            source_dt_seconds = (
                1.0
                /
                source_fps
            )

        analyzed_dt_seconds = (
            source_dt_seconds
            *
            process_every_n_frames
        )

        duration_seconds = (
            total_frames
            * source_dt_seconds
        )

        stem = clean_stem(
            input_path.stem
        )

        video_output_codec = "not_created"

        metadata = {
            "filename":
                input_path.name,

            "experiment_name":
                stem,

            "analysis_mode":
                "microglia",

            "device":
                str(DEVICE),

            "model":
                (
                    str(
                        MICROGLIA_MODEL_PATH
                    )
                    if MICROGLIA_MODEL_PATH
                    else "microglia model"
                ),

            "segmentation":
                "Microglia semantic segmentation + watershed instance separation",

            "classes": {
                "0":
                    "background",

                "1":
                    "microglia signal / interior",

                "2":
                    "boundary"
            },

            "tracking":
                "Soma-centre weighted nearest-neighbour / Hungarian assignment",

            "morphology":
                (
                    "Cell geometry + skeleton topology + Sholl analysis"
                ),

            "motility":
                (
                    "Soma movement + process-tip displacement + "
                    "extension/retraction classification"
                ),

            "total_frames":
                total_frames,

            "width":
                width,

            "height":
                height,

            "fps":
                source_fps,

            "source_frame_interval_seconds":
                source_dt_seconds,

            "analyzed_frame_interval_seconds":
                analyzed_dt_seconds,

            "duration_seconds":
                duration_seconds,

            "pixel_calibration_um_per_px":
                pixel_size_um,

            "analysis_frame_step":
                process_every_n_frames,

            "inference_batch_size":
                inference_batch_size,

            "scientific_note":
                (
                    "Morphology-derived measurements describe image structure "
                    "and movement. Morphology alone does not establish "
                    "microglial activation, apoptosis, metabolism or a "
                    "specific biological state."
                ),

            "proliferation_note":
                (
                    "Population change is an imaging-derived estimate and "
                    "should not be interpreted as a definitive proliferation "
                    "rate without sufficient time-lapse duration and biological "
                    "validation."
                ),

            "process_motion_note":
                (
                    "Process-tip extension/retraction is estimated by matching "
                    "skeleton endpoints between analyzed frames."
                ),

            "video_output_codec":
                video_output_codec,

            "video_browser_compatibility":
                (
                    "H.264/yuv420p MP4 when FFmpeg is available; "
                    "otherwise OpenCV mp4v fallback."
                )
        }

        with jobs_lock:

            jobs[job_id].update(
                {
                    "status":
                        "processing",

                    "metadata":
                        metadata,

                    "progress":
                        0.0,

                    "current_frame":
                        0,

                    "total_frames":
                        total_frames,

                    "elapsed_seconds":
                        0.0,

                    "frames_per_second":
                        0.0,

                    "eta_seconds":
                        None,

                    "message":
                        "Preparing microglia analysis..."
                }
            )

        # ----------------------------------------------------
        # Output video
        # ----------------------------------------------------

        # Write to an intermediate MP4 first. It is converted to
        # browser-friendly H.264 after analysis when FFmpeg exists.
        raw_annotated_path = (
            output_dir
            /
            f"{stem}_microglia_annotated_raw.mp4"
        )

        annotated_path = (
            output_dir
            /
            f"{stem}_microglia_annotated.mp4"
        )

        video_writer = None
        video_output_codec = "not_created"

        if width > 0 and height > 0:

            output_fps = (
                source_fps
                if source_fps > 0
                else 1.0
            )

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
            )

            video_writer = (
                cv2.VideoWriter(
                    str(
                        raw_annotated_path
                    ),
                    fourcc,
                    output_fps,
                    (
                        width,
                        height
                    )
                )
            )

            if not video_writer.isOpened():

                video_writer.release()

                video_writer = None

            else:
                video_output_codec = "mp4v-intermediate"

        # ----------------------------------------------------
        # Tracking state
        # ----------------------------------------------------

        active_tracks = {}

        finished_tracks = {}

        next_track_id = 1

        frame_rows = []

        # ----------------------------------------------------
        # Frame iterator
        # ----------------------------------------------------

        def video_frame_iterator():

            cap = cv2.VideoCapture(
                str(input_path)
            )

            if not cap.isOpened():

                raise RuntimeError(
                    f"Could not open video: {input_path}"
                )

            try:

                frame_index = 0

                while True:

                    ok, frame = cap.read()

                    if not ok:
                        break

                    yield (
                        frame_index,
                        frame
                    )

                    frame_index += 1

            finally:

                cap.release()

        def image_frame_iterator():

            image = input_info[
                "image"
            ]

            yield (
                0,
                image
            )

        def stack_frame_iterator():

            stack = input_info[
                "stack"
            ]

            for index in range(
                stack.shape[0]
            ):

                yield (
                    index,
                    stack[index]
                )

        if input_info[
            "type"
        ] == "video":

            iterator = (
                video_frame_iterator()
            )

        elif input_info[
            "type"
        ] == "single_image":

            iterator = (
                image_frame_iterator()
            )

        else:

            iterator = (
                stack_frame_iterator()
            )

        pending = []

        # ----------------------------------------------------
        # Helper: finalize tracks
        # ----------------------------------------------------

        def finalize_expired_tracks(
            current_frame
        ):

            expired = []

            for track_id, track in list(
                active_tracks.items()
            ):

                if track.last_frame == (
                    current_frame
                ):

                    continue

                if (
                    current_frame
                    -
                    track.last_frame
                    >
                    DEFAULT_MAX_MISSED_FRAMES
                    *
                    process_every_n_frames
                ):

                    expired.append(
                        track_id
                    )

            for track_id in expired:

                finished_tracks[
                    track_id
                ] = active_tracks.pop(
                    track_id
                )

        # ----------------------------------------------------
        # Process batch
        # ----------------------------------------------------

        def process_frames(
            items
        ):

            nonlocal next_track_id

            if not items:
                return

            frames_for_model = [
                item[
                    "frame"
                ]
                for item in items
                if item[
                    "should_process"
                ]
            ]

            predictions = (
                predict_microglia_batch(
                    model,
                    frames_for_model
                )
            )

            prediction_index = 0

            for item in items:

                frame_index = int(
                    item[
                        "frame_index"
                    ]
                )

                frame = item[
                    "frame"
                ]

                should_process = (
                    item[
                        "should_process"
                    ]
                )

                if should_process:

                    prediction = (
                        predictions[
                            prediction_index
                        ]
                    )

                    prediction_index += 1

                    (
                        detections,
                        instance_labels
                    ) = build_detections(
                        prediction,
                        pixel_size_um,
                        frame_index
                    )

                    # ------------------------------------------------
                    # Add hidden metadata used by tracker/renderer.
                    # ------------------------------------------------

                    for detection in detections:

                        detection[
                            "_instance_labels"
                        ] = instance_labels

                        detection[
                            "_pixel_size_um"
                        ] = pixel_size_um

                    # ------------------------------------------------
                    # Build matching matrix.
                    # ------------------------------------------------

                    track_ids = list(
                        active_tracks.keys()
                    )

                    cost_matrix = []

                    for track_id in track_ids:

                        previous = (
                            active_tracks[
                                track_id
                            ].last_detection
                        )

                        row = []

                        for detection in detections:

                            cost = (
                                cell_match_cost(
                                    previous,
                                    detection
                                )
                            )

                            if cost is None:

                                row.append(
                                    1e6
                                )

                            else:

                                row.append(
                                    cost
                                )

                        cost_matrix.append(
                            row
                        )

                    matched_tracks = set()

                    matched_detections = set()

                    if (
                        cost_matrix
                        and detections
                    ):

                        matrix = np.asarray(
                            cost_matrix,
                            dtype=np.float64
                        )

                        row_indices, col_indices = (
                            linear_sum_assignment(
                                matrix
                            )
                        )

                        for row_index, col_index in zip(
                            row_indices,
                            col_indices
                        ):

                            cost = float(
                                matrix[
                                    row_index,
                                    col_index
                                ]
                            )

                            if cost >= 1e5:
                                continue

                            track_id = (
                                track_ids[
                                    row_index
                                ]
                            )

                            detection = (
                                detections[
                                    col_index
                                ]
                            )

                            active_tracks[
                                track_id
                            ].update(
                                detection,
                                frame_index
                            )

                            matched_tracks.add(
                                track_id
                            )

                            matched_detections.add(
                                col_index
                            )

                    # ------------------------------------------------
                    # Unmatched tracks become temporarily missed.
                    # ------------------------------------------------

                    for track_id in track_ids:

                        if (
                            track_id
                            not in
                            matched_tracks
                        ):

                            active_tracks[
                                track_id
                            ].missed += 1

                    # ------------------------------------------------
                    # New tracks.
                    # ------------------------------------------------

                    for detection_index, detection in enumerate(
                        detections
                    ):

                        if (
                            detection_index
                            in
                            matched_detections
                        ):
                            continue

                        track = (
                            MicrogliaTrack(
                                next_track_id,
                                detection,
                                frame_index,
                                source_fps,
                                width,
                                height
                            )
                        )

                        active_tracks[
                            next_track_id
                        ] = track

                        next_track_id += 1

                    # ------------------------------------------------
                    # Frame-level metrics.
                    # ------------------------------------------------

                    active_current = [
                        track
                        for track in
                        active_tracks.values()
                        if (
                            track.last_frame
                            ==
                            frame_index
                        )
                    ]

                    cell_count = len(
                        active_current
                    )

                    area_values = [
                        track.last_detection[
                            "area_um2"
                        ]
                        for track in active_current
                    ]

                    branch_values = [
                        track.last_detection[
                            "branch_points"
                        ]
                        for track in active_current
                    ]

                    endpoint_values = [
                        track.last_detection[
                            "endpoints"
                        ]
                        for track in active_current
                    ]

                    process_lengths = [
                        track.last_detection[
                            "skeleton_length_um"
                        ]
                        for track in active_current
                    ]

                    soma_radii = [
                        track.last_detection[
                            "soma_radius_um"
                        ]
                        for track in active_current
                    ]

                    total_pixels = float(
                        prediction.shape[0] *
                        prediction.shape[1]
                    )

                    background_percent = (
                        float(
                            np.sum(
                                prediction == 0
                            )
                        )
                        / max(total_pixels, 1.0)
                        * 100.0
                    )

                    interior_percent = (
                        float(
                            np.sum(
                                prediction == 1
                            )
                        )
                        / max(total_pixels, 1.0)
                        * 100.0
                    )

                    boundary_percent = (
                        float(
                            np.sum(
                                prediction == 2
                            )
                        )
                        / max(total_pixels, 1.0)
                        * 100.0
                    )

                    predicted_signal_percent = (
                        interior_percent
                        +
                        boundary_percent
                    )

                    frame_rows.append(
                        {
                            "frame":
                                frame_index,

                            "time_seconds":
                                frame_index
                                *
                                source_dt_seconds,

                            "cell_count":
                                cell_count,

                            "background_percent":
                                background_percent,

                            "interior_percent":
                                interior_percent,

                            "boundary_percent":
                                boundary_percent,

                            "predicted_signal_percent":
                                predicted_signal_percent,

                            "mean_cell_area_um2":
                                (
                                    float(
                                        np.mean(
                                            area_values
                                        )
                                    )
                                    if area_values
                                    else 0.0
                                ),

                            "mean_skeleton_length_um":
                                (
                                    float(
                                        np.mean(
                                            process_lengths
                                        )
                                    )
                                    if process_lengths
                                    else 0.0
                                ),

                            "mean_branch_points":
                                (
                                    float(
                                        np.mean(
                                            branch_values
                                        )
                                    )
                                    if branch_values
                                    else 0.0
                                ),

                            "mean_endpoints":
                                (
                                    float(
                                        np.mean(
                                            endpoint_values
                                        )
                                    )
                                    if endpoint_values
                                    else 0.0
                                ),

                            "mean_soma_radius_um":
                                (
                                    float(
                                        np.mean(
                                            soma_radii
                                        )
                                    )
                                    if soma_radii
                                    else 0.0
                                )
                        }
                    )

                    # ------------------------------------------------
                    # Annotation
                    # ------------------------------------------------

                    if video_writer is not None:

                        annotated = (
                            annotate_microglia_frame(
                                frame,
                                prediction,
                                detections,
                                active_tracks
                            )
                        )

                        if (
                            annotated.shape[1]
                            != width
                            or
                            annotated.shape[0]
                            != height
                        ):

                            annotated = cv2.resize(
                                annotated,
                                (
                                    width,
                                    height
                                )
                            )

                        video_writer.write(
                            annotated
                        )

                else:

                    if video_writer is not None:

                        original_frame = frame

                        if original_frame.ndim == 2:

                            original_frame = (
                                cv2.cvtColor(
                                    original_frame,
                                    cv2.COLOR_GRAY2BGR
                                )
                            )

                        if (
                            original_frame.shape[1]
                            != width
                            or
                            original_frame.shape[0]
                            != height
                        ):

                            original_frame = cv2.resize(
                                original_frame,
                                (
                                    width,
                                    height
                                )
                            )

                        video_writer.write(
                            original_frame
                        )

        # ----------------------------------------------------
        # Iterate frames with mini-batching.
        # ----------------------------------------------------

        for frame_index, frame in iterator:

            should_process = (
                frame_index
                %
                process_every_n_frames
                == 0
            )

            pending.append(
                {
                    "frame_index":
                        frame_index,

                    "frame":
                        frame,

                    "should_process":
                        should_process
                }
            )

            if (
                len(pending)
                >=
                inference_batch_size
            ):

                process_frames(
                    pending
                )

                pending = []

            elapsed = (
                time.perf_counter()
                -
                analysis_start
            )

            processed_frames = (
                frame_index + 1
            )

            effective_fps = (
                processed_frames
                /
                elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = max(
                0,
                total_frames
                -
                processed_frames
            )

            eta = (
                remaining
                /
                effective_fps
                if effective_fps > 0
                else None
            )

            progress = (
                processed_frames
                /
                max(
                    total_frames,
                    1
                )
                * 100.0
            )

            with jobs_lock:

                jobs[job_id].update(
                    {
                        "progress":
                            progress,

                        "current_frame":
                            processed_frames,

                        "total_frames":
                            total_frames,

                        "elapsed_seconds":
                            elapsed,

                        "frames_per_second":
                            effective_fps,

                        "eta_seconds":
                            eta,

                        "message":
                            (
                                "Microglia segmentation, "
                                "tracking and morphology..."
                            )
                    }
                )

        if pending:

            process_frames(
                pending
            )

        # ----------------------------------------------------
        # Close video writer.
        # ----------------------------------------------------

        if video_writer is not None:

            video_writer.release()

            # ------------------------------------------------
            # Convert to H.264 for reliable browser playback.
            # Fast-start moves the MP4 metadata to the front
            # so Chrome can begin streaming immediately.
            # ------------------------------------------------

            ffmpeg = shutil.which(
                "ffmpeg"
            )

            if (
                ffmpeg
                and raw_annotated_path.exists()
                and raw_annotated_path.stat().st_size > 0
            ):
                try:
                    command = [
                        ffmpeg,
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(raw_annotated_path),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(annotated_path),
                    ]

                    conversion = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                    )

                    if (
                        conversion.returncode == 0
                        and annotated_path.exists()
                        and annotated_path.stat().st_size > 0
                    ):
                        raw_annotated_path.unlink(
                            missing_ok=True
                        )
                        video_output_codec = (
                            "H.264 / yuv420p"
                        )

                    else:
                        print(
                            "[MICROGLIA] FFmpeg H.264 conversion failed:",
                            conversion.stderr.strip(),
                        )
                        annotated_path = (
                            raw_annotated_path
                        )
                        video_output_codec = (
                            "mp4v-intermediate"
                        )

                except Exception as video_error:
                    print(
                        "[MICROGLIA] H.264 conversion warning:",
                        video_error,
                    )
                    annotated_path = (
                        raw_annotated_path
                    )
                    video_output_codec = (
                        "mp4v-intermediate"
                    )

            else:
                # FFmpeg is not available; preserve the generated
                # MP4 rather than losing the annotation video.
                annotated_path = (
                    raw_annotated_path
                )
                video_output_codec = (
                    "mp4v-intermediate"
                )

        # ----------------------------------------------------
        # Move remaining tracks to finished.
        # ----------------------------------------------------

        for track_id, track in (
            active_tracks.items()
        ):

            finished_tracks[
                track_id
            ] = track

        active_tracks.clear()

        # ----------------------------------------------------
        # Track tables
        # ----------------------------------------------------

        track_rows = []

        tip_rows = []

        for track_id in sorted(
            finished_tracks.keys()
        ):

            track = finished_tracks[
                track_id
            ]

            for row in track.rows:

                track_rows.append(
                    row
                )

            for row in track.tip_rows:

                tip_rows.append(
                    row
                )

        track_df = pd.DataFrame(
            track_rows
        )

        tip_df = pd.DataFrame(
            tip_rows
        )

        frame_df = pd.DataFrame(
            frame_rows
        )

        # ----------------------------------------------------
        # Remove private fields:
        # already not part of public rows.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Per-track summary
        # ----------------------------------------------------

        summaries = []

        for track_id in sorted(
            finished_tracks.keys()
        ):

            track = finished_tracks[
                track_id
            ]

            rows = track.rows

            if not rows:
                continue

            start_frame = rows[0][
                "frame"
            ]

            end_frame = rows[-1][
                "frame"
            ]

            duration = (
                (
                    end_frame
                    -
                    start_frame
                )
                *
                source_dt_seconds
            )

            soma_speed_values = [
                row[
                    "soma_speed_um_min"
                ]
                for row in rows
                if row[
                    "soma_speed_um_min"
                ]
                is not None
            ]

            centroid_speed_values = [
                row[
                    "centroid_speed_um_min"
                ]
                for row in rows
                if row[
                    "centroid_speed_um_min"
                ]
                is not None
            ]

            area_values = [
                row[
                    "area_um2"
                ]
                for row in rows
            ]

            process_values = [
                row[
                    "skeleton_length_um"
                ]
                for row in rows
            ]

            branch_values = [
                row[
                    "branch_points"
                ]
                for row in rows
            ]

            endpoint_values = [
                row[
                    "endpoints"
                ]
                for row in rows
            ]

            total_soma_distance = (
                sum(
                    row[
                        "soma_displacement_um"
                    ]
                    for row in rows
                )
            )

            total_centroid_distance = (
                sum(
                    row[
                        "centroid_displacement_um"
                    ]
                    for row in rows
                )
            )

            extensions = 0
            retractions = 0
            stable = 0

            track_tip_rows = [
                row
                for row in tip_rows
                if row[
                    "track_id"
                ]
                == track_id
            ]

            tip_speed_values = [
                row[
                    "speed_um_min"
                ]
                for row in track_tip_rows
                if row[
                    "speed_um_min"
                ]
                is not None
            ]

            for row in track_tip_rows:

                state = row[
                    "state"
                ]

                if state == "extension":
                    extensions += 1

                elif state == "retraction":
                    retractions += 1

                else:
                    stable += 1

            if (
                len(area_values) >= 2
                and area_values[0] > 0
            ):

                area_change_percent = (
                    (
                        area_values[-1]
                        -
                        area_values[0]
                    )
                    /
                    area_values[0]
                    * 100.0
                )

            else:

                area_change_percent = 0.0

            end_detection = (
                track.last_detection
            )

            end_border = (
                is_near_border(
                    end_detection[
                        "centroid_x_px"
                    ],
                    end_detection[
                        "centroid_y_px"
                    ],
                    width,
                    height
                )
            )

            max_area = max(
                area_values
            )

            final_area = area_values[
                -1
            ]

            shrinkage_fraction = (
                (
                    max_area
                    -
                    final_area
                )
                /
                max(
                    max_area,
                    1e-6
                )
            )

            disappearance_candidate = (
                (
                    not end_border
                )
                and
                (
                    shrinkage_fraction
                    >= 0.30
                )
                and
                (
                    len(rows)
                    >= 3
                )
            )

            summaries.append(
                {
                    "track_id":
                        track_id,

                    "frames_tracked":
                        len(rows),

                    "start_frame":
                        start_frame,

                    "end_frame":
                        end_frame,

                    "duration_seconds":
                        duration,

                    "mean_soma_speed_um_min":
                        (
                            float(
                                np.mean(
                                    soma_speed_values
                                )
                            )
                            if soma_speed_values
                            else None
                        ),

                    "max_soma_speed_um_min":
                        (
                            float(
                                np.max(
                                    soma_speed_values
                                )
                            )
                            if soma_speed_values
                            else None
                        ),

                    "mean_centroid_speed_um_min":
                        (
                            float(
                                np.mean(
                                    centroid_speed_values
                                )
                            )
                            if centroid_speed_values
                            else None
                        ),

                    "max_centroid_speed_um_min":
                        (
                            float(
                                np.max(
                                    centroid_speed_values
                                )
                            )
                            if centroid_speed_values
                            else None
                        ),

                    "mean_process_tip_speed_um_min":
                        (
                            float(
                                np.mean(
                                    tip_speed_values
                                )
                            )
                            if tip_speed_values
                            else None
                        ),

                    "max_process_tip_speed_um_min":
                        (
                            float(
                                np.max(
                                    tip_speed_values
                                )
                            )
                            if tip_speed_values
                            else None
                        ),

                    "total_soma_distance_um":
                        total_soma_distance,

                    "total_centroid_distance_um":
                        total_centroid_distance,

                    "mean_cell_area_um2":
                        float(
                            np.mean(
                                area_values
                            )
                        ),

                    "max_cell_area_um2":
                        float(
                            np.max(
                                area_values
                            )
                        ),

                    "area_change_percent":
                        float(
                            area_change_percent
                        ),

                    "mean_skeleton_length_um":
                        float(
                            np.mean(
                                process_values
                            )
                        ),

                    "mean_branch_points":
                        float(
                            np.mean(
                                branch_values
                            )
                        ),

                    "mean_endpoints":
                        float(
                            np.mean(
                                endpoint_values
                            )
                        ),

                    "extension_events":
                        int(
                            extensions
                        ),

                    "retraction_events":
                        int(
                            retractions
                        ),

                    "stable_tip_events":
                        int(
                            stable
                        ),

                    "start_at_border":
                        bool(
                            track.start_border
                        ),

                    "end_at_border":
                        bool(
                            end_border
                        ),

                    "disappearance_like_candidate":
                        bool(
                            disappearance_candidate
                        )
                }
            )

        summary_df = pd.DataFrame(
            summaries
        )

        # ----------------------------------------------------
        # Population statistics
        # ----------------------------------------------------

        population_rows = []

        if not frame_df.empty:

            for row in frame_df.itertuples():

                population_rows.append(
                    {
                        "time_seconds":
                            row.time_seconds,

                        "cell_count":
                            row.cell_count
                    }
                )

        population_df = pd.DataFrame(
            population_rows
        )

        apparent_growth_rate = None
        estimated_doubling_time = None

        if (
            len(frame_df) >= 3
            and np.all(
                frame_df[
                    "cell_count"
                ].to_numpy()
                > 0
            )
        ):

            times_minutes = (
                frame_df[
                    "time_seconds"
                ].to_numpy(
                    dtype=float
                )
                /
                60.0
            )

            counts = (
                frame_df[
                    "cell_count"
                ].to_numpy(
                    dtype=float
                )
            )

            try:

                slope = float(
                    np.polyfit(
                        times_minutes,
                        np.log(
                            counts
                        ),
                        1
                    )[0]
                )

                if math.isfinite(
                    slope
                ):

                    apparent_growth_rate = (
                        slope
                    )

                    if slope > 0:

                        estimated_doubling_time = (
                            math.log(2.0)
                            / slope
                        )

            except Exception:

                apparent_growth_rate = None
                estimated_doubling_time = None

        population_summary = {
            "average_cell_count":
                (
                    float(
                        frame_df[
                            "cell_count"
                        ].mean()
                    )
                    if not frame_df.empty
                    else 0.0
                ),

            "maximum_cell_count":
                (
                    int(
                        frame_df[
                            "cell_count"
                        ].max()
                    )
                    if not frame_df.empty
                    else 0
                ),

            "minimum_cell_count":
                (
                    int(
                        frame_df[
                            "cell_count"
                        ].min()
                    )
                    if not frame_df.empty
                    else 0
                ),

            "apparent_growth_rate_per_min":
                apparent_growth_rate,

            "estimated_doubling_time_min":
                estimated_doubling_time,

            "interpretation_note":
                (
                    "Population growth and doubling values are "
                    "image-derived estimates and require a sufficiently "
                    "long time-lapse and biological validation before "
                    "being interpreted as proliferation."
                )
        }

        # ----------------------------------------------------
        # Overall microglia summary
        # ----------------------------------------------------

        all_soma_speeds = []

        all_tip_speeds = []

        for summary in summaries:

            if (
                summary[
                    "mean_soma_speed_um_min"
                ]
                is not None
            ):

                all_soma_speeds.append(
                    summary[
                        "mean_soma_speed_um_min"
                    ]
                )

            if (
                summary[
                    "mean_process_tip_speed_um_min"
                ]
                is not None
            ):

                all_tip_speeds.append(
                    summary[
                        "mean_process_tip_speed_um_min"
                    ]
                )

        extension_total = int(
            summary_df[
                "extension_events"
            ].sum()
        ) if not summary_df.empty else 0

        retraction_total = int(
            summary_df[
                "retraction_events"
            ].sum()
        ) if not summary_df.empty else 0

        disappearance_candidates = int(
            summary_df[
                "disappearance_like_candidate"
            ].sum()
        ) if not summary_df.empty else 0

        microglia_summary = {
            "cells_tracked":
                int(
                    len(
                        summary_df
                    )
                ),

            "cell_observations":
                int(
                    len(
                        track_df
                    )
                ),

            "mean_cell_area_um2":
                (
                    float(
                        track_df[
                            "area_um2"
                        ].mean()
                    )
                    if not track_df.empty
                    else 0.0
                ),

            "mean_soma_area_proxy_um2":
                (
                    float(
                        track_df[
                            "soma_area_proxy_um2"
                        ].mean()
                    )
                    if not track_df.empty
                    else 0.0
                ),

            "mean_skeleton_length_um":
                (
                    float(
                        track_df[
                            "skeleton_length_um"
                        ].mean()
                    )
                    if not track_df.empty
                    else 0.0
                ),

            "mean_branch_points":
                (
                    float(
                        track_df[
                            "branch_points"
                        ].mean()
                    )
                    if not track_df.empty
                    else 0.0
                ),

            "mean_endpoints":
                (
                    float(
                        track_df[
                            "endpoints"
                        ].mean()
                    )
                    if not track_df.empty
                    else 0.0
                ),

            "mean_sholl_max_intersections":
                (
                    float(
                        track_df[
                            "sholl_max_intersections"
                        ].mean()
                    )
                    if not track_df.empty
                    else 0.0
                ),

            "mean_soma_speed_um_min":
                (
                    float(
                        np.mean(
                            all_soma_speeds
                        )
                    )
                    if all_soma_speeds
                    else None
                ),

            "mean_process_tip_speed_um_min":
                (
                    float(
                        np.mean(
                            all_tip_speeds
                        )
                    )
                    if all_tip_speeds
                    else None
                ),

            "process_extension_events":
                extension_total,

            "process_retraction_events":
                retraction_total,

            "disappearance_like_candidates":
                disappearance_candidates,

            "frame_interval_seconds":
                analyzed_dt_seconds,

            "pixel_size_um_per_px":
                pixel_size_um
        }

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        cell_csv = (
            output_dir
            / f"{stem}_microglia_cell_tracks.csv"
        )

        tip_csv = (
            output_dir
            / f"{stem}_microglia_process_tips.csv"
        )

        frame_csv = (
            output_dir
            / f"{stem}_microglia_frame_metrics.csv"
        )

        summary_csv = (
            output_dir
            / f"{stem}_microglia_track_summary.csv"
        )

        population_csv = (
            output_dir
            / f"{stem}_microglia_population.csv"
        )

        track_df.to_csv(
            cell_csv,
            index=False
        )

        tip_df.to_csv(
            tip_csv,
            index=False
        )

        frame_df.to_csv(
            frame_csv,
            index=False
        )

        summary_df.to_csv(
            summary_csv,
            index=False
        )

        population_df.to_csv(
            population_csv,
            index=False
        )

        # ----------------------------------------------------
        # Excel
        # ----------------------------------------------------

        excel_path = (
            output_dir
            / f"{stem}_microglia_analysis.xlsx"
        )

        summary_sheet = pd.DataFrame(
            [
                {
                    "Metric":
                        "Cells tracked",

                    "Value":
                        microglia_summary[
                            "cells_tracked"
                        ]
                },

                {
                    "Metric":
                        "Mean cell area (µm²)",

                    "Value":
                        microglia_summary[
                            "mean_cell_area_um2"
                        ]
                },

                {
                    "Metric":
                        "Mean soma area proxy (µm²)",

                    "Value":
                        microglia_summary[
                            "mean_soma_area_proxy_um2"
                        ]
                },

                {
                    "Metric":
                        "Mean skeleton/process length (µm)",

                    "Value":
                        microglia_summary[
                            "mean_skeleton_length_um"
                        ]
                },

                {
                    "Metric":
                        "Mean branch points",

                    "Value":
                        microglia_summary[
                            "mean_branch_points"
                        ]
                },

                {
                    "Metric":
                        "Mean endpoints",

                    "Value":
                        microglia_summary[
                            "mean_endpoints"
                        ]
                },

                {
                    "Metric":
                        "Mean Sholl maximum",

                    "Value":
                        microglia_summary[
                            "mean_sholl_max_intersections"
                        ]
                },

                {
                    "Metric":
                        "Mean soma speed (µm/min)",

                    "Value":
                        microglia_summary[
                            "mean_soma_speed_um_min"
                        ]
                },

                {
                    "Metric":
                        "Mean process-tip speed (µm/min)",

                    "Value":
                        microglia_summary[
                            "mean_process_tip_speed_um_min"
                        ]
                },

                {
                    "Metric":
                        "Process extensions",

                    "Value":
                        extension_total
                },

                {
                    "Metric":
                        "Process retractions",

                    "Value":
                        retraction_total
                },

                {
                    "Metric":
                        "Apparent population growth/min",

                    "Value":
                        population_summary[
                            "apparent_growth_rate_per_min"
                        ]
                },

                {
                    "Metric":
                        "Estimated doubling time/min",

                    "Value":
                        population_summary[
                            "estimated_doubling_time_min"
                        ]
                }
            ]
        )

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl"
        ) as writer:

            summary_sheet.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

            track_df.to_excel(
                writer,
                sheet_name="Cell Tracks",
                index=False
            )

            tip_df.to_excel(
                writer,
                sheet_name="Process Tips",
                index=False
            )

            frame_df.to_excel(
                writer,
                sheet_name="Frame Metrics",
                index=False
            )

            summary_df.to_excel(
                writer,
                sheet_name="Track Summary",
                index=False
            )

            population_df.to_excel(
                writer,
                sheet_name="Population",
                index=False
            )

        # ----------------------------------------------------
        # Graphs
        # ----------------------------------------------------

        graph_files = generate_microglia_graphs(
            frame_df,
            track_df,
            summary_df,
            tip_df,
            output_dir,
        )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        report = {
            "experiment":
                metadata,

            "summary":
                {
                    "averageCells":
                        population_summary[
                            "average_cell_count"
                        ],

                    "maximumCells":
                        population_summary[
                            "maximum_cell_count"
                        ],

                    "trackedObjects":
                        microglia_summary[
                            "cells_tracked"
                        ],

                    "meanArea":
                        microglia_summary[
                            "mean_cell_area_um2"
                        ],

                    "meanProcessLength":
                        microglia_summary[
                            "mean_skeleton_length_um"
                        ],

                    "meanBranchPoints":
                        microglia_summary[
                            "mean_branch_points"
                        ],

                    "meanEndpoints":
                        microglia_summary[
                            "mean_endpoints"
                        ],

                    "meanShollMaximum":
                        microglia_summary[
                            "mean_sholl_max_intersections"
                        ],

                    "meanSomaSpeed":
                        microglia_summary[
                            "mean_soma_speed_um_min"
                        ],

                    "meanProcessTipSpeed":
                        microglia_summary[
                            "mean_process_tip_speed_um_min"
                        ],

                    "processExtensions":
                        extension_total,

                    "processRetractions":
                        retraction_total,

                    "apparentGrowthRatePerMin":
                        population_summary[
                            "apparent_growth_rate_per_min"
                        ],

                    "estimatedDoublingTimeMin":
                        population_summary[
                            "estimated_doubling_time_min"
                        ]
                },

            "microglia":
                microglia_summary,

            "population_dynamics":
                population_summary,

            "graphs":
                graph_files,

            "graph_count":
                int(
                    len(graph_files)
                ),

            "files":
                {
                    "cell_tracks":
                        cell_csv.name,

                    "process_tips":
                        tip_csv.name,

                    "frame_metrics":
                        frame_csv.name,

                    "track_summary":
                        summary_csv.name,

                    "population":
                        population_csv.name,

                    "excel":
                        excel_path.name,

                    "annotated_video":
                        (
                            annotated_path.name
                            if annotated_path.exists()
                            else None
                        )
                }
        }

        report_path = (
            output_dir
            / f"{stem}_microglia_report.txt"
        )

        with report_path.open(
            "w",
            encoding="utf-8"
        ) as handle:

            handle.write(
                "CELL LAB — MICROGLIA QUANTITATIVE ANALYSIS\n"
            )

            handle.write(
                "=" * 72
                + "\n\n"
            )

            handle.write(
                "EXPERIMENT\n"
            )

            handle.write(
                "-" * 72
                + "\n"
            )

            handle.write(
                f"File: {metadata['filename']}\n"
            )

            handle.write(
                f"Resolution: {width} x {height} px\n"
            )

            handle.write(
                f"Source FPS: {source_fps:.4f}\n"
            )

            handle.write(
                f"Source frame interval: "
                f"{source_dt_seconds:.4f} s\n"
            )

            handle.write(
                f"Analyzed frame interval: "
                f"{analyzed_dt_seconds:.4f} s\n"
            )

            handle.write(
                f"Pixel calibration: "
                f"{pixel_size_um} µm/px\n"
            )

            handle.write(
                f"Device: {DEVICE}\n"
            )

            handle.write(
                f"Model: {MICROGLIA_MODEL_PATH}\n"
            )

            handle.write(
                f"Annotated video codec: {video_output_codec}\n"
            )

            handle.write(
                f"Generated graph count: {len(graph_files)}\n\n"
            )

            handle.write(
                "MICROGLIAL MORPHOLOGY\n"
            )

            handle.write(
                "-" * 72
                + "\n"
            )

            handle.write(
                f"Cells tracked: "
                f"{microglia_summary['cells_tracked']}\n"
            )

            handle.write(
                f"Mean cell area: "
                f"{microglia_summary['mean_cell_area_um2']:.3f} µm²\n"
            )

            handle.write(
                f"Mean soma area proxy: "
                f"{microglia_summary['mean_soma_area_proxy_um2']:.3f} µm²\n"
            )

            handle.write(
                f"Mean skeleton/arbor length: "
                f"{microglia_summary['mean_skeleton_length_um']:.3f} µm\n"
            )

            handle.write(
                f"Mean branch points: "
                f"{microglia_summary['mean_branch_points']:.3f}\n"
            )

            handle.write(
                f"Mean endpoints: "
                f"{microglia_summary['mean_endpoints']:.3f}\n"
            )

            handle.write(
                f"Mean Sholl maximum: "
                f"{microglia_summary['mean_sholl_max_intersections']:.3f}\n\n"
            )

            handle.write(
                "MICROGLIAL MOTILITY\n"
            )

            handle.write(
                "-" * 72
                + "\n"
            )

            handle.write(
                f"Mean soma speed: "
                f"{microglia_summary['mean_soma_speed_um_min']}\n"
            )

            handle.write(
                f"Mean process-tip speed: "
                f"{microglia_summary['mean_process_tip_speed_um_min']}\n"
            )

            handle.write(
                f"Process extensions: "
                f"{extension_total}\n"
            )

            handle.write(
                f"Process retractions: "
                f"{retraction_total}\n\n"
            )

            handle.write(
                "POPULATION\n"
            )

            handle.write(
                "-" * 72
                + "\n"
            )

            handle.write(
                f"Average detected cells: "
                f"{population_summary['average_cell_count']:.3f}\n"
            )

            handle.write(
                f"Maximum detected cells: "
                f"{population_summary['maximum_cell_count']}\n"
            )

            handle.write(
                f"Apparent growth rate: "
                f"{population_summary['apparent_growth_rate_per_min']}\n"
            )

            handle.write(
                f"Estimated doubling time: "
                f"{population_summary['estimated_doubling_time_min']}\n\n"
            )

            handle.write(
                "SCIENTIFIC LIMITATIONS\n"
            )

            handle.write(
                "-" * 72
                + "\n"
            )

            handle.write(
                metadata[
                    "scientific_note"
                ]
                + "\n"
            )

            handle.write(
                metadata[
                    "proliferation_note"
                ]
                + "\n"
            )

            handle.write(
                metadata[
                    "process_motion_note"
                ]
                + "\n"
            )

        report[
            "files"
        ][
            "report"
        ] = report_path.name

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        result_path = (
            output_dir
            / "result.json"
        )

        with result_path.open(
            "w",
            encoding="utf-8"
        ) as handle:

            json.dump(
                report,
                handle,
                indent=2,
                default=str
            )

        total_elapsed = max(
            0.001,
            time.perf_counter()
            -
            analysis_start
        )

        with jobs_lock:

            jobs[job_id].update(
                {
                    "status":
                        "complete",

                    "progress":
                        100.0,

                    "current_frame":
                        total_frames,

                    "total_frames":
                        total_frames,

                    "elapsed_seconds":
                        total_elapsed,

                    "frames_per_second":
                        (
                            total_frames
                            /
                            total_elapsed
                            if total_frames > 0
                            else 0.0
                        ),

                    "eta_seconds":
                        0.0,

                    "message":
                        "Microglia analysis complete.",

                    "result":
                        report,

                    "error":
                        None
                }
            )

    except Exception as error:

        traceback.print_exc()

        with jobs_lock:

            jobs[job_id].update(
                {
                    "status":
                        "error",

                    "message":
                        str(error),

                    "error":
                        traceback.format_exc()
                }
            )


# ============================================================
# MICROGLIA UPLOAD INTERCEPTOR
# ============================================================

@app.before_request
def microglia_request_interceptor():

    if request.path != "/api/analyze":
        return None

    # If analysis_mode is absent, leave the original
    # Cell Lab API completely untouched.
    analysis_mode = (
        request.form.get(
            "analysis_mode",
            "bacteria"
        )
        .strip()
        .lower()
    )

    if analysis_mode != "microglia":
        return None

    if "video" not in request.files:

        return jsonify(
            {
                "error":
                    (
                        "No microscopy file was uploaded. "
                        "Use multipart/form-data field 'video'."
                    )
            }
        ), 400

    uploaded = request.files[
        "video"
    ]

    if not uploaded.filename:

        return jsonify(
            {
                "error":
                    "Uploaded file has no filename."
            }
        ), 400

    original_name = clean_name(
        uploaded.filename
    )

    allowed_extensions = {
        ".avi",
        ".mp4",
        ".mov",
        ".mkv",
        ".mpg",
        ".mpeg",
        ".webm",
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    }

    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in allowed_extensions:

        return jsonify(
            {
                "error":
                    (
                        f"Unsupported microscopy file type: "
                        f"{extension}. "
                        f"Allowed: "
                        f"{', '.join(sorted(allowed_extensions))}"
                    )
            }
        ), 400

    try:

        process_every_n_frames = int(
            request.form.get(
                "process_every_n_frames",
                DEFAULT_MICROGLIA_FRAME_STEP
            )
        )

    except ValueError:

        process_every_n_frames = (
            DEFAULT_MICROGLIA_FRAME_STEP
        )

    process_every_n_frames = int(
        clamp(
            process_every_n_frames,
            1,
            10
        )
    )

    try:

        inference_batch_size = int(
            request.form.get(
                "inference_batch_size",
                DEFAULT_MICROGLIA_BATCH_SIZE
            )
        )

    except ValueError:

        inference_batch_size = (
            DEFAULT_MICROGLIA_BATCH_SIZE
        )

    inference_batch_size = int(
        clamp(
            inference_batch_size,
            1,
            8
        )
    )

    pixel_size_raw = (
        request.form.get(
            "pixel_size_um",
            ""
        )
        .strip()
    )

    if pixel_size_raw:

        try:

            pixel_size_um = float(
                pixel_size_raw
            )

            if pixel_size_um <= 0:
                pixel_size_um = (
                    DEFAULT_MICROGLIA_PIXEL_SIZE_UM
                )

        except ValueError:

            pixel_size_um = (
                DEFAULT_MICROGLIA_PIXEL_SIZE_UM
            )

    else:

        pixel_size_um = (
            DEFAULT_MICROGLIA_PIXEL_SIZE_UM
        )

    interval_raw = (
        request.form.get(
            "frame_interval_seconds",
            ""
        )
        .strip()
    )

    frame_interval_seconds = None

    if interval_raw:

        try:

            value = float(
                interval_raw
            )

            if value > 0:
                frame_interval_seconds = value

        except ValueError:
            frame_interval_seconds = None

    job_id = uuid.uuid4().hex[:12]

    job_upload_dir = (
        UPLOAD_DIR
        /
        job_id
    )

    output_dir = (
        ANALYSIS_DIR
        /
        job_id
    )

    job_upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_path = (
        job_upload_dir
        /
        original_name
    )

    uploaded.save(
        input_path
    )

    with jobs_lock:

        jobs[job_id] = {
            "id":
                job_id,

            "status":
                "queued",

            "progress":
                0.0,

            "current_frame":
                0,

            "total_frames":
                0,

            "elapsed_seconds":
                0.0,

            "frames_per_second":
                0.0,

            "eta_seconds":
                None,

            "message":
                "Microglia dataset uploaded.",

            "filename":
                original_name,

            "result":
                None,

            "error":
                None,

            "analysis_mode":
                "microglia"
        }

    thread = threading.Thread(
        target=run_microglia_job,
        args=(
            job_id,
            input_path,
            output_dir,
            process_every_n_frames,
            inference_batch_size,
            pixel_size_um,
            frame_interval_seconds
        ),
        daemon=True
    )

    thread.start()

    return jsonify(
        {
            "job_id":
                job_id,

            "status":
                "queued",

            "filename":
                original_name,

            "analysis_mode":
                "microglia",

            "process_every_n_frames":
                process_every_n_frames,

            "inference_batch_size":
                inference_batch_size,

            "pixel_size_um":
                pixel_size_um,

            "frame_interval_seconds":
                frame_interval_seconds,

            "message":
                "Microglia analysis started."
        }
    )


# ============================================================
# MICROGLIA INFORMATION ENDPOINT
# ============================================================

@app.get(
    "/api/microglia/info"
)
def microglia_info():

    model_path = (
        find_microglia_model_path()
    )

    return jsonify(
        {
            "mode":
                "microglia",

            "device":
                str(DEVICE),

            "model_exists":
                model_path is not None,

            "model":
                (
                    str(model_path)
                    if model_path
                    else None
                ),

            "default_pixel_size_um":
                DEFAULT_MICROGLIA_PIXEL_SIZE_UM,

            "default_frame_step":
                DEFAULT_MICROGLIA_FRAME_STEP,

            "metrics":
                [
                    "cell area",
                    "soma area proxy",
                    "perimeter",
                    "circularity",
                    "solidity",
                    "eccentricity",
                    "aspect ratio",
                    "skeleton/arbor length",
                    "branch points",
                    "endpoints",
                    "branch density",
                    "maximum process reach",
                    "Sholl maximum intersections",
                    "Sholl critical radius",
                    "soma velocity",
                    "process-tip velocity",
                    "process extension",
                    "process retraction",
                    "population count",
                    "semantic layer composition",
                    "microglia signal coverage",
                    "trajectory analysis",
                    "soma speed distribution",
                    "morphology-motion relationships",
                    "extension/retraction event balance"
                ]
        }
    )


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 76)
    print("CELL LAB — UNIFIED QUANTITATIVE MICROSCOPY BACKEND")
    print("=" * 76)
    print(
        f"Base directory : {BASE_DIR}"
    )
    print(
        f"Device         : {DEVICE}"
    )
    print(
        f"Bacteria core  : "
        f"{Path(app_core.__file__).resolve()}"
    )

    model_path = (
        find_microglia_model_path()
    )

    print(
        f"Microglia model: "
        f"{model_path if model_path else 'NOT FOUND'}"
    )

    print(
        f"Uploads        : {UPLOAD_DIR}"
    )

    print(
        f"Analysis       : {ANALYSIS_DIR}"
    )

    print(
        f"FFmpeg         : {shutil.which('ffmpeg') or 'NOT FOUND'}"
    )

    print("=" * 76)

    print()
    print(
        "Bacteria endpoint:"
    )

    print(
        "  /api/analyze"
    )

    print(
        "Microglia endpoint:"
    )

    print(
        "  /api/analyze  + analysis_mode=microglia"
    )

    print()
    print(
        "Microglia info:"
    )

    print(
        "  http://127.0.0.1:5000/api/microglia/info"
    )

    print()
    print(
        "Starting server at http://127.0.0.1:5000"
    )
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )