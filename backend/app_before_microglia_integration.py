"""
CELL LAB — Unified Quantitative Microscopy Backend
=================================================

Single-file Flask backend for the Cell Lab React dashboard.

What it provides
----------------
1. Upload a microscopy video.
2. Instance-U-Net segmentation:
      0 = background
      1 = cell interior
      2 = cell boundary
3. Faster batched inference on MPS/CUDA/CPU.
4. Cell detection and centroid tracking.
5. Whole-cell motility analysis.
6. Cell morphology analysis:
      area, perimeter, circularity, solidity, extent,
      eccentricity, major/minor axis, orientation,
      equivalent diameter, intensity, aspect ratio.
7. Morphodynamic / protrusion-like boundary-change metrics.
8. Population dynamics:
      cell count, appearance/loss events,
      apparent growth rate, exponential growth rate,
      estimated doubling time, candidate division events.
9. Death/apoptosis-like morphology proxy.
   IMPORTANT: this is a candidate-event score, NOT a diagnosis of apoptosis.
10. CSV files, quantitative plots, report, annotated MP4.
11. Progress API with percentage, frames, throughput, elapsed time and ETA.

Place beside this file:
    instance_unet_model.pth

Install:
    pip install flask flask-cors opencv-python torch numpy pandas matplotlib scipy

Optional but strongly recommended on macOS for H.264 MP4 output:
    brew install ffmpeg

Run:
    python3 app.py

API:
    POST /api/analyze
    GET  /api/status/<job_id>
    GET  /api/results/<job_id>
    GET  /analysis/<job_id>/<filename>
    GET  /api/health
"""

from __future__ import annotations

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
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "instance_unet_model.pth"
UPLOAD_DIR = BASE_DIR / "uploads"
ANALYSIS_DIR = BASE_DIR / "analysis"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# The model was trained using 512 x 512 inputs.
# Keep this for accuracy.
INPUT_SIZE = 512

# Speed/accuracy compromise.
# 1 = every source frame.
# 2 = every second source frame.
DEFAULT_PROCESS_EVERY_N_FRAMES = 2

# Batched inference improves GPU utilization without changing
# the network architecture or input resolution.
DEFAULT_INFERENCE_BATCH_SIZE = 4

# Detection filtering.
MIN_CELL_AREA = 25
MAX_CELL_AREA = None

# Tracking.
MAX_TRACK_DISTANCE = 60.0
MAX_MISSED_FRAMES = 3
MOVEMENT_THRESHOLD_PX_S = 1.0

# Avoid counting tracks entering/leaving through the frame edge
# as biological birth/death events.
BORDER_MARGIN = 18

# Candidate division heuristic.
DIVISION_RADIUS = 55.0
DIVISION_MIN_SEPARATION = 4.0

# Candidate death/apoptosis-like heuristic.
MIN_EVENT_TRACK_FRAMES = 5
SHRINKAGE_SCORE_SCALE = 0.40
COMPACTNESS_INCREASE_SCALE = 1.00
DEATH_SCORE_THRESHOLD = 0.65


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[CELL LAB] Device: {DEVICE}")


# ============================================================
# INSTANCE U-NET
# This matches the uploaded model_instance.py architecture.
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
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

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.conv1 = DoubleConv(128, 64)

        # 0 = background
        # 1 = interior
        # 2 = boundary
        self.final = nn.Conv2d(64, 3, kernel_size=1)

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


MODEL = None
MODEL_LOCK = threading.Lock()


def load_model():
    global MODEL

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Put instance_unet_model.pth beside app.py."
            )

        print(f"[CELL LAB] Loading model: {MODEL_PATH}")

        model = InstanceUNet().to(DEVICE)
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        cleaned = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

        model.load_state_dict(cleaned)
        model.eval()

        MODEL = model

        print("[CELL LAB] Instance U-Net loaded successfully.")
        return MODEL


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def clean_name(name: str) -> str:
    name = Path(name).name
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in name
    )
    return safe or "uploaded_video.avi"


def clean_stem(name: str) -> str:
    stem = Path(name).stem
    safe = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in stem
    )
    return safe or "microscopy_analysis"


def clamp(value, low, high):
    return max(low, min(high, value))


def normalized_score(value, scale):
    if scale <= 0:
        return 0.0
    return clamp(value / scale, 0.0, 1.0)


def point_distance(a, b):
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    return math.sqrt(dx * dx + dy * dy)


def is_near_border(x, y, width, height):
    return (
        x <= BORDER_MARGIN
        or y <= BORDER_MARGIN
        or x >= width - BORDER_MARGIN
        or y >= height - BORDER_MARGIN
    )


def safe_linear_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return 0.0

    try:
        slope = float(np.polyfit(x, y, 1)[0])
        return slope if math.isfinite(slope) else 0.0
    except Exception:
        return 0.0


# ============================================================
# IMAGE / MODEL INPUT
# ============================================================

def normalize_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    minimum = float(gray.min())
    maximum = float(gray.max())

    if maximum > minimum:
        gray = (gray - minimum) / (maximum - minimum)
    else:
        gray.fill(0.0)

    return gray


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    gray = normalize_gray(frame)

    resized = cv2.resize(
        gray,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    return resized.astype(np.float32)


def predict_classes_batch(
    model,
    frames: list[np.ndarray],
) -> list[np.ndarray]:
    if not frames:
        return []

    prepared = np.stack(
        [prepare_frame(frame) for frame in frames],
        axis=0,
    )

    tensor = torch.from_numpy(prepared)
    tensor = tensor.unsqueeze(1).to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)
        prediction = torch.argmax(
            output,
            dim=1,
        )

    prediction = (
        prediction
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    results = []

    for index, frame in enumerate(frames):
        height, width = frame.shape[:2]

        restored = cv2.resize(
            prediction[index],
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        results.append(restored)

    return results


# ============================================================
# MORPHOLOGY
# ============================================================

def mask_change_metrics(
    previous_detection: dict,
    current_detection: dict,
):
    """
    Compare two cell masks after approximately aligning their centroids.

    The resulting quantities are deliberately described as
    protrusion/retraction-LIKE boundary remodeling proxies.
    They are not individual tentacle measurements.
    """

    prev_mask = previous_detection.get("mask")
    curr_mask = current_detection.get("mask")

    if prev_mask is None or curr_mask is None:
        return {
            "protrusion_area_px2": 0.0,
            "retraction_area_px2": 0.0,
            "shape_change_index": 0.0,
            "protrusion_extent_px": 0.0,
            "retraction_extent_px": 0.0,
        }

    px, py, _, _ = previous_detection["bbox"]
    cx, cy, _, _ = current_detection["bbox"]

    prev_center = (
        float(previous_detection["x"]),
        float(previous_detection["y"]),
    )

    curr_center = (
        float(current_detection["x"]),
        float(current_detection["y"]),
    )

    shift_x = int(round(curr_center[0] - prev_center[0]))
    shift_y = int(round(curr_center[1] - prev_center[1]))

    shifted_px = px + shift_x
    shifted_py = py + shift_y

    curr_h, curr_w = curr_mask.shape[:2]
    prev_h, prev_w = prev_mask.shape[:2]

    min_x = min(shifted_px, cx)
    min_y = min(shifted_py, cy)
    max_x = max(
        shifted_px + prev_w,
        cx + curr_w,
    )
    max_y = max(
        shifted_py + prev_h,
        cy + curr_h,
    )

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    if canvas_w <= 0 or canvas_h <= 0:
        return {
            "protrusion_area_px2": 0.0,
            "retraction_area_px2": 0.0,
            "shape_change_index": 0.0,
            "protrusion_extent_px": 0.0,
            "retraction_extent_px": 0.0,
        }

    prev_canvas = np.zeros(
        (canvas_h, canvas_w),
        dtype=np.uint8,
    )

    curr_canvas = np.zeros(
        (canvas_h, canvas_w),
        dtype=np.uint8,
    )

    prev_x0 = shifted_px - min_x
    prev_y0 = shifted_py - min_y
    curr_x0 = cx - min_x
    curr_y0 = cy - min_y

    prev_canvas[
        prev_y0:prev_y0 + prev_h,
        prev_x0:prev_x0 + prev_w,
    ] = prev_mask

    curr_canvas[
        curr_y0:curr_y0 + curr_h,
        curr_x0:curr_x0 + curr_w,
    ] = curr_mask

    prev_bool = prev_canvas > 0
    curr_bool = curr_canvas > 0

    protrusion = curr_bool & ~prev_bool
    retraction = prev_bool & ~curr_bool
    union = curr_bool | prev_bool

    protrusion_area = float(np.sum(protrusion))
    retraction_area = float(np.sum(retraction))
    union_area = float(np.sum(union))

    shape_change = (
        (protrusion_area + retraction_area) / union_area
        if union_area > 0
        else 0.0
    )

    def max_radius(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return 0.0

        dx = xs + min_x - curr_center[0]
        dy = ys + min_y - curr_center[1]
        return float(np.sqrt(dx * dx + dy * dy).max())

    return {
        "protrusion_area_px2": protrusion_area,
        "retraction_area_px2": retraction_area,
        "shape_change_index": shape_change,
        "protrusion_extent_px": max_radius(protrusion),
        "retraction_extent_px": max_radius(retraction),
    }


def object_morphology(
    region: np.ndarray,
    x0: int,
    y0: int,
    frame: np.ndarray,
):
    """Compute 2D region-shape features similar to standard microscopy tools."""

    area = int(np.sum(region))

    if area <= 0:
        return None

    mask_u8 = (region.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if contours:
        contour = max(
            contours,
            key=cv2.contourArea,
        )
        perimeter = float(
            cv2.arcLength(contour, True)
        )
        hull = cv2.convexHull(contour)
        hull_area = float(
            cv2.contourArea(hull)
        )
    else:
        contour = None
        perimeter = 0.0
        hull_area = float(area)

    circularity = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )

    solidity = (
        area / hull_area
        if hull_area > 0
        else 0.0
    )

    ys, xs = np.where(region)

    if len(xs) == 0:
        return None

    global_x = xs.astype(np.float64) + x0
    global_y = ys.astype(np.float64) + y0

    mean_x = float(global_x.mean())
    mean_y = float(global_y.mean())

    dx = global_x - mean_x
    dy = global_y - mean_y

    covariance = np.cov(
        np.vstack([dx, dy]),
        bias=True,
    )

    if covariance.shape != (2, 2):
        covariance = np.zeros((2, 2), dtype=float)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        major_lambda = max(float(eigenvalues[0]), 0.0)
        minor_lambda = max(float(eigenvalues[1]), 0.0)

        major_axis = 4.0 * math.sqrt(major_lambda)
        minor_axis = 4.0 * math.sqrt(minor_lambda)

        if major_axis > 0:
            eccentricity = math.sqrt(
                max(
                    0.0,
                    1.0 - (minor_axis ** 2) / (major_axis ** 2),
                )
            )
        else:
            eccentricity = 0.0

        vx = float(eigenvectors[0, 0])
        vy = float(eigenvectors[1, 0])
        orientation = math.degrees(
            math.atan2(vy, vx)
        )
    except Exception:
        major_axis = 0.0
        minor_axis = 0.0
        eccentricity = 0.0
        orientation = 0.0

    ys_bbox, xs_bbox = np.where(region)
    bbox_w = int(xs_bbox.max() - xs_bbox.min() + 1)
    bbox_h = int(ys_bbox.max() - ys_bbox.min() + 1)

    bbox_area = max(
        1,
        bbox_w * bbox_h,
    )

    extent = area / bbox_area
    aspect_ratio = (
        bbox_w / bbox_h
        if bbox_h > 0
        else 0.0
    )

    equivalent_diameter = math.sqrt(
        4.0 * area / math.pi
    )

    compactness = (
        perimeter * perimeter /
        (4.0 * math.pi * area)
        if area > 0
        else 0.0
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    local_gray = gray[
        y0:y0 + region.shape[0],
        x0:x0 + region.shape[1],
    ]

    pixel_values = local_gray[region]

    mean_intensity = (
        float(pixel_values.mean())
        if len(pixel_values)
        else 0.0
    )

    intensity_std = (
        float(pixel_values.std())
        if len(pixel_values)
        else 0.0
    )

    return {
        "perimeter_px": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
        "major_axis_length_px": major_axis,
        "minor_axis_length_px": minor_axis,
        "orientation_deg": orientation,
        "equivalent_diameter_px": equivalent_diameter,
        "compactness": compactness,
        "bbox_width_px": bbox_w,
        "bbox_height_px": bbox_h,
        "aspect_ratio": aspect_ratio,
        "mean_intensity": mean_intensity,
        "intensity_std": intensity_std,
    }


# ============================================================
# CELL DETECTION
# ============================================================

def detect_cells(
    prediction: np.ndarray,
    frame: np.ndarray,
) -> list[dict]:
    interior = (
        prediction == 1
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    interior = cv2.morphologyEx(
        interior,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    labels, count = ndimage.label(
        interior,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    if count == 0:
        return []

    objects = ndimage.find_objects(labels)
    detections = []

    for label_id, object_slice in enumerate(
        objects,
        start=1,
    ):
        if object_slice is None:
            continue

        ys, xs = object_slice
        region = (
            labels[ys, xs] == label_id
        )

        area = int(
            region.sum()
        )

        if area < MIN_CELL_AREA:
            continue

        if (
            MAX_CELL_AREA is not None
            and area > MAX_CELL_AREA
        ):
            continue

        coordinates = np.argwhere(
            region
        )

        if len(coordinates) == 0:
            continue

        cy_local = float(
            coordinates[:, 0].mean()
        )

        cx_local = float(
            coordinates[:, 1].mean()
        )

        y0 = int(ys.start)
        x0 = int(xs.start)
        h = int(ys.stop - ys.start)
        w = int(xs.stop - xs.start)

        cx = float(x0 + cx_local)
        cy = float(y0 + cy_local)

        morphology = object_morphology(
            region,
            x0,
            y0,
            frame,
        )

        if morphology is None:
            continue

        detections.append(
            {
                "x": cx,
                "y": cy,
                "area": area,
                "label": label_id,
                "bbox": (
                    x0,
                    y0,
                    w,
                    h,
                ),
                "mask": (
                    region.astype(np.uint8)
                ),
                **morphology,
            }
        )

    return detections


# ============================================================
# TRACKING
# ============================================================

class Track:
    def __init__(
        self,
        track_id: int,
        detection: dict,
        frame_number: int,
        fps: float,
    ):
        self.track_id = track_id
        self.start_frame = frame_number
        self.last_frame = frame_number
        self.end_frame = None
        self.missed = 0

        self.positions = [
            self._row(
                detection,
                frame_number,
                fps,
                previous=None,
            )
        ]

        self.last_detection = detection

    def _row(
        self,
        detection,
        frame_number,
        fps,
        previous,
    ):
        time_seconds = (
            frame_number / fps
            if fps > 0
            else 0.0
        )

        displacement = 0.0
        speed = 0.0
        area_change_rate = 0.0
        area_change_percent = 0.0
        circularity_change = 0.0
        eccentricity_change = 0.0
        perimeter_change = 0.0
        intensity_change = 0.0

        protrusion_area = 0.0
        retraction_area = 0.0
        shape_change_index = 0.0
        protrusion_extent = 0.0
        retraction_extent = 0.0

        if previous is not None:
            dx = (
                detection["x"] -
                previous["detection"]["x"]
            )
            dy = (
                detection["y"] -
                previous["detection"]["y"]
            )

            displacement = math.sqrt(
                dx * dx + dy * dy
            )

            dt = (
                frame_number -
                previous["frame"]
            ) / fps if fps > 0 else 0.0

            speed = (
                displacement / dt
                if dt > 0
                else 0.0
            )

            previous_area = safe_float(
                previous["detection"]["area"]
            )

            current_area = safe_float(
                detection["area"]
            )

            area_change = (
                current_area - previous_area
            )

            area_change_rate = (
                area_change / dt
                if dt > 0
                else 0.0
            )

            area_change_percent = (
                100.0 * area_change /
                previous_area
                if previous_area > 0
                else 0.0
            )

            circularity_change = (
                safe_float(
                    detection["circularity"]
                ) -
                safe_float(
                    previous["detection"]["circularity"]
                )
            )

            eccentricity_change = (
                safe_float(
                    detection["eccentricity"]
                ) -
                safe_float(
                    previous["detection"]["eccentricity"]
                )
            )

            perimeter_change = (
                safe_float(
                    detection["perimeter_px"]
                ) -
                safe_float(
                    previous["detection"]["perimeter_px"]
                )
            )

            intensity_change = (
                safe_float(
                    detection["mean_intensity"]
                ) -
                safe_float(
                    previous["detection"]["mean_intensity"]
                )
            )

            boundary = mask_change_metrics(
                previous["detection"],
                detection,
            )

            protrusion_area = boundary[
                "protrusion_area_px2"
            ]

            retraction_area = boundary[
                "retraction_area_px2"
            ]

            shape_change_index = boundary[
                "shape_change_index"
            ]

            protrusion_extent = boundary[
                "protrusion_extent_px"
            ]

            retraction_extent = boundary[
                "retraction_extent_px"
            ]

            protrusion_area_rate = (
                protrusion_area / dt
                if dt > 0
                else 0.0
            )

            retraction_area_rate = (
                retraction_area / dt
                if dt > 0
                else 0.0
            )

            protrusion_extent_rate = (
                protrusion_extent / dt
                if dt > 0
                else 0.0
            )

            retraction_extent_rate = (
                retraction_extent / dt
                if dt > 0
                else 0.0
            )
        else:
            protrusion_area_rate = 0.0
            retraction_area_rate = 0.0
            protrusion_extent_rate = 0.0
            retraction_extent_rate = 0.0

        return {
            "track_id": self.track_id,
            "frame": frame_number,
            "time_seconds": time_seconds,
            "x": detection["x"],
            "y": detection["y"],
            "area_px": detection["area"],
            "perimeter_px": detection["perimeter_px"],
            "circularity": detection["circularity"],
            "solidity": detection["solidity"],
            "extent": detection["extent"],
            "eccentricity": detection["eccentricity"],
            "major_axis_length_px": detection[
                "major_axis_length_px"
            ],
            "minor_axis_length_px": detection[
                "minor_axis_length_px"
            ],
            "orientation_deg": detection[
                "orientation_deg"
            ],
            "equivalent_diameter_px": detection[
                "equivalent_diameter_px"
            ],
            "compactness": detection[
                "compactness"
            ],
            "bbox_width_px": detection[
                "bbox_width_px"
            ],
            "bbox_height_px": detection[
                "bbox_height_px"
            ],
            "aspect_ratio": detection[
                "aspect_ratio"
            ],
            "mean_intensity": detection[
                "mean_intensity"
            ],
            "intensity_std": detection[
                "intensity_std"
            ],
            "displacement_px": displacement,
            "speed_px_s": speed,
            "area_change_rate_px2_s": area_change_rate,
            "area_change_percent": area_change_percent,
            "circularity_change": circularity_change,
            "eccentricity_change": eccentricity_change,
            "perimeter_change_px": perimeter_change,
            "intensity_change": intensity_change,
            "protrusion_area_px2": protrusion_area,
            "retraction_area_px2": retraction_area,
            "protrusion_area_rate_px2_s": protrusion_area_rate,
            "retraction_area_rate_px2_s": retraction_area_rate,
            "protrusion_extent_px": protrusion_extent,
            "retraction_extent_px": retraction_extent,
            "protrusion_extent_rate_px_s": protrusion_extent_rate,
            "retraction_extent_rate_px_s": retraction_extent_rate,
            "shape_change_index": shape_change_index,
        }

    @property
    def last_position(self):
        return self.positions[-1]

    def add(
        self,
        detection: dict,
        frame_number: int,
        fps: float,
    ):
        previous = {
            "frame": self.last_frame,
            "detection": self.last_detection,
        }

        row = self._row(
            detection,
            frame_number,
            fps,
            previous,
        )

        self.positions.append(row)
        self.last_frame = frame_number
        self.last_detection = detection
        self.missed = 0


class CentroidTracker:
    """Deterministic nearest-neighbour tracker with event bookkeeping."""

    def __init__(
        self,
        fps: float,
        max_distance: float = MAX_TRACK_DISTANCE,
        max_missed: int = MAX_MISSED_FRAMES,
    ):
        self.fps = fps
        self.max_distance = max_distance
        self.max_missed = max_missed

        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.active_ids: set[int] = set()

    def update(
        self,
        detections: list[dict],
        frame_number: int,
        frame_width: int,
        frame_height: int,
    ):
        previous_active = {
            track_id: self.tracks[track_id].last_position.copy()
            for track_id in self.active_ids
        }

        track_ids = list(self.active_ids)
        candidates = []

        for track_id in track_ids:
            track = self.tracks[track_id]
            px = track.last_position["x"]
            py = track.last_position["y"]

            for detection_index, detection in enumerate(detections):
                dx = detection["x"] - px
                dy = detection["y"] - py
                distance = math.sqrt(dx * dx + dy * dy)

                if distance <= self.max_distance:
                    candidates.append(
                        (
                            distance,
                            track_id,
                            detection_index,
                        )
                    )

        candidates.sort(key=lambda item: item[0])

        matched_tracks = set()
        matched_detections = set()

        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            self.tracks[track_id].add(
                detections[detection_index],
                frame_number,
                self.fps,
            )

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        ended_ids = []

        for track_id in list(self.active_ids):
            if track_id in matched_tracks:
                continue

            track = self.tracks[track_id]
            track.missed += 1

            if track.missed > self.max_missed:
                track.end_frame = track.last_frame
                self.active_ids.remove(track_id)
                ended_ids.append(track_id)

        new_ids = []
        new_id_detection_pairs = []

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue

            track = Track(
                self.next_id,
                detection,
                frame_number,
                self.fps,
            )

            self.tracks[self.next_id] = track
            self.active_ids.add(self.next_id)

            new_ids.append(self.next_id)
            new_id_detection_pairs.append(
                (
                    self.next_id,
                    detection,
                )
            )

            self.next_id += 1

        # Candidate division event heuristic:
        # two or more new tracks appear close to one previous active track.
        division_parents = []

        for old_id, old_position in previous_active.items():
            nearby_new = []

            for new_id, detection in new_id_detection_pairs:
                distance = math.sqrt(
                    (
                        detection["x"] - old_position["x"]
                    ) ** 2
                    +
                    (
                        detection["y"] - old_position["y"]
                    ) ** 2
                )

                if distance <= DIVISION_RADIUS:
                    nearby_new.append(
                        (new_id, detection)
                    )

            if len(nearby_new) < 2:
                continue

            sufficiently_separated = False

            for i in range(len(nearby_new)):
                for j in range(i + 1, len(nearby_new)):
                    if (
                        point_distance(
                            nearby_new[i][1],
                            nearby_new[j][1],
                        )
                        >= DIVISION_MIN_SEPARATION
                    ):
                        sufficiently_separated = True
                        break
                if sufficiently_separated:
                    break

            if sufficiently_separated:
                division_parents.append(old_id)

        return {
            "new_ids": new_ids,
            "ended_ids": ended_ids,
            "matched_ids": list(matched_tracks),
            "division_parents": division_parents,
        }


# ============================================================
# FRAME METRICS
# ============================================================

def create_frame_metric(
    frame_number: int,
    fps: float,
    detections: list[dict],
    prediction: np.ndarray,
    tracker_event: dict,
):
    areas = [
        safe_float(d["area"])
        for d in detections
    ]

    circularities = [
        safe_float(d["circularity"])
        for d in detections
    ]

    solidities = [
        safe_float(d["solidity"])
        for d in detections
    ]

    eccentricities = [
        safe_float(d["eccentricity"])
        for d in detections
    ]

    perimeters = [
        safe_float(d["perimeter_px"])
        for d in detections
    ]

    intensities = [
        safe_float(d["mean_intensity"])
        for d in detections
    ]

    return {
        "frame": frame_number,
        "time_seconds": (
            frame_number / fps
            if fps > 0
            else 0.0
        ),
        "cell_count": len(detections),
        "new_tracks": len(
            tracker_event["new_ids"]
        ),
        "lost_tracks": len(
            tracker_event["ended_ids"]
        ),
        "candidate_division_events": len(
            tracker_event["division_parents"]
        ),
        "moving_cells": 0,
        "mean_speed_px_s": 0.0,
        "median_speed_px_s": 0.0,
        "max_speed_px_s": 0.0,
        "mean_displacement_px": 0.0,
        "total_displacement_px": 0.0,
        "mean_cell_area_px": (
            float(np.mean(areas))
            if areas else 0.0
        ),
        "mean_perimeter_px": (
            float(np.mean(perimeters))
            if perimeters else 0.0
        ),
        "mean_circularity": (
            float(np.mean(circularities))
            if circularities else 0.0
        ),
        "mean_solidity": (
            float(np.mean(solidities))
            if solidities else 0.0
        ),
        "mean_eccentricity": (
            float(np.mean(eccentricities))
            if eccentricities else 0.0
        ),
        "mean_intensity": (
            float(np.mean(intensities))
            if intensities else 0.0
        ),
        "interior_pixels": int(
            np.sum(prediction == 1)
        ),
        "boundary_pixels": int(
            np.sum(prediction == 2)
        ),
        "protrusion_area_rate_px2_s": 0.0,
        "retraction_area_rate_px2_s": 0.0,
        "protrusion_extent_rate_px_s": 0.0,
        "retraction_extent_rate_px_s": 0.0,
        "mean_shape_change_index": 0.0,
        "mean_area_change_rate_px2_s": 0.0,
        "candidate_apoptosis_like_events": 0,
    }


def add_motion_to_frame_metric(
    frame_metric: dict,
    current_rows: list[dict],
):
    speeds = [
        safe_float(row["speed_px_s"])
        for row in current_rows
        if safe_float(row["speed_px_s"]) > 0
    ]

    displacements = [
        safe_float(row["displacement_px"])
        for row in current_rows
        if safe_float(row["displacement_px"]) > 0
    ]

    protrusion = [
        safe_float(
            row["protrusion_area_rate_px2_s"]
        )
        for row in current_rows
    ]

    retraction = [
        safe_float(
            row["retraction_area_rate_px2_s"]
        )
        for row in current_rows
    ]

    protrusion_extent = [
        safe_float(
            row["protrusion_extent_rate_px_s"]
        )
        for row in current_rows
    ]

    retraction_extent = [
        safe_float(
            row["retraction_extent_rate_px_s"]
        )
        for row in current_rows
    ]

    shape_changes = [
        safe_float(
            row["shape_change_index"]
        )
        for row in current_rows
    ]

    area_rates = [
        safe_float(
            row["area_change_rate_px2_s"]
        )
        for row in current_rows
    ]

    frame_metric["moving_cells"] = sum(
        1
        for row in current_rows
        if safe_float(
            row["speed_px_s"]
        ) >= MOVEMENT_THRESHOLD_PX_S
    )

    frame_metric["mean_speed_px_s"] = (
        float(np.mean(speeds))
        if speeds else 0.0
    )

    frame_metric["median_speed_px_s"] = (
        float(np.median(speeds))
        if speeds else 0.0
    )

    frame_metric["max_speed_px_s"] = (
        float(np.max(speeds))
        if speeds else 0.0
    )

    frame_metric["mean_displacement_px"] = (
        float(np.mean(displacements))
        if displacements else 0.0
    )

    frame_metric["total_displacement_px"] = (
        float(np.sum(displacements))
        if displacements else 0.0
    )

    frame_metric["protrusion_area_rate_px2_s"] = (
        float(np.mean(protrusion))
        if protrusion else 0.0
    )

    frame_metric["retraction_area_rate_px2_s"] = (
        float(np.mean(retraction))
        if retraction else 0.0
    )

    frame_metric["protrusion_extent_rate_px_s"] = (
        float(np.mean(protrusion_extent))
        if protrusion_extent else 0.0
    )

    frame_metric["retraction_extent_rate_px_s"] = (
        float(np.mean(retraction_extent))
        if retraction_extent else 0.0
    )

    frame_metric["mean_shape_change_index"] = (
        float(np.mean(shape_changes))
        if shape_changes else 0.0
    )

    frame_metric["mean_area_change_rate_px2_s"] = (
        float(np.mean(area_rates))
        if area_rates else 0.0
    )


# ============================================================
# TRACK SUMMARY / BIOLOGICAL EVENT FEATURES
# ============================================================

def calculate_turning_angle_degrees(rows):
    vectors = []

    for previous, current in zip(
        rows[:-1],
        rows[1:],
    ):
        vx = float(current["x"]) - float(previous["x"])
        vy = float(current["y"]) - float(previous["y"])

        norm = math.sqrt(vx * vx + vy * vy)

        if norm > 1e-9:
            vectors.append(
                (vx / norm, vy / norm)
            )

    angles = []

    for a, b in zip(
        vectors[:-1],
        vectors[1:],
    ):
        dot = clamp(
            a[0] * b[0] + a[1] * b[1],
            -1.0,
            1.0,
        )
        angles.append(
            math.degrees(
                math.acos(dot)
            )
        )

    return (
        float(np.mean(angles))
        if angles
        else 0.0
    )


def evaluate_death_like_candidate(
    rows,
    width,
    height,
    final_analyzed_frame,
):
    if len(rows) < MIN_EVENT_TRACK_FRAMES:
        return 0.0, False

    first_area_values = [
        safe_float(row["area_px"])
        for row in rows[:3]
        if safe_float(row["area_px"]) > 0
    ]

    last_area_values = [
        safe_float(row["area_px"])
        for row in rows[-3:]
        if safe_float(row["area_px"]) > 0
    ]

    if not first_area_values or not last_area_values:
        return 0.0, False

    initial_area = float(
        np.median(first_area_values)
    )

    final_area = float(
        np.median(last_area_values)
    )

    if initial_area <= 0:
        return 0.0, False

    shrinkage = max(
        0.0,
        1.0 - final_area / initial_area,
    )

    initial_compactness = float(
        np.mean([
            safe_float(row["compactness"])
            for row in rows[:3]
        ])
    )

    final_compactness = float(
        np.mean([
            safe_float(row["compactness"])
            for row in rows[-3:]
        ])
    )

    compactness_increase = max(
        0.0,
        final_compactness - initial_compactness,
    )

    shrinkage_score = normalized_score(
        shrinkage,
        SHRINKAGE_SCORE_SCALE,
    )

    shape_irregularity_score = normalized_score(
        compactness_increase,
        COMPACTNESS_INCREASE_SCALE,
    )

    score = (
        0.70 * shrinkage_score
        + 0.30 * shape_irregularity_score
    )

    last = rows[-1]

    interior_end = not is_near_border(
        safe_float(last["x"]),
        safe_float(last["y"]),
        width,
        height,
    )

    ended_before_video_end = (
        rows[-1]["frame"] < final_analyzed_frame
    )

    candidate = (
        score >= DEATH_SCORE_THRESHOLD
        and interior_end
        and ended_before_video_end
    )

    return float(score), bool(candidate)


def build_track_summary(
    tracks: dict[int, Track],
    fps: float,
    width: int,
    height: int,
    final_analyzed_frame: int,
):
    summaries = []

    for track_id in sorted(tracks):
        track = tracks[track_id]
        rows = track.positions

        if not rows:
            continue

        speeds = np.asarray(
            [safe_float(row["speed_px_s"]) for row in rows],
            dtype=float,
        )

        distances = np.asarray(
            [safe_float(row["displacement_px"]) for row in rows],
            dtype=float,
        )

        areas = np.asarray(
            [safe_float(row["area_px"]) for row in rows],
            dtype=float,
        )

        perimeters = np.asarray(
            [safe_float(row["perimeter_px"]) for row in rows],
            dtype=float,
        )

        circularities = np.asarray(
            [safe_float(row["circularity"]) for row in rows],
            dtype=float,
        )

        solidities = np.asarray(
            [safe_float(row["solidity"]) for row in rows],
            dtype=float,
        )

        eccentricities = np.asarray(
            [safe_float(row["eccentricity"]) for row in rows],
            dtype=float,
        )

        intensities = np.asarray(
            [safe_float(row["mean_intensity"]) for row in rows],
            dtype=float,
        )

        protrusion_rates = np.asarray(
            [
                safe_float(
                    row[
                        "protrusion_area_rate_px2_s"
                    ]
                )
                for row in rows
            ],
            dtype=float,
        )

        retraction_rates = np.asarray(
            [
                safe_float(
                    row[
                        "retraction_area_rate_px2_s"
                    ]
                )
                for row in rows
            ],
            dtype=float,
        )

        shape_changes = np.asarray(
            [
                safe_float(
                    row["shape_change_index"]
                )
                for row in rows
            ],
            dtype=float,
        )

        first = rows[0]
        last = rows[-1]

        dx = (
            float(last["x"]) - float(first["x"])
        )
        dy = (
            float(last["y"]) - float(first["y"])
        )

        net_displacement = math.sqrt(
            dx * dx + dy * dy
        )

        total_distance = float(
            np.sum(distances[distances > 0])
        )

        frames_tracked = len(rows)

        duration_seconds = (
            (
                float(last["frame"])
                - float(first["frame"])
            ) / fps
            if fps > 0
            else 0.0
        )

        mean_speed = (
            float(
                np.mean(
                    speeds[speeds > 0]
                )
            )
            if np.any(speeds > 0)
            else 0.0
        )

        maximum_speed = (
            float(np.max(speeds))
            if len(speeds)
            else 0.0
        )

        confinement_ratio = (
            net_displacement /
            total_distance
            if total_distance > 0
            else 0.0
        )

        straight_line_speed = (
            net_displacement /
            duration_seconds
            if duration_seconds > 0
            else 0.0
        )

        linearity = (
            straight_line_speed /
            mean_speed
            if mean_speed > 0
            else 0.0
        )

        mean_turning_angle = (
            calculate_turning_angle_degrees(rows)
        )

        start_at_border = is_near_border(
            safe_float(first["x"]),
            safe_float(first["y"]),
            width,
            height,
        )

        end_at_border = is_near_border(
            safe_float(last["x"]),
            safe_float(last["y"]),
            width,
            height,
        )

        death_score, death_candidate = (
            evaluate_death_like_candidate(
                rows,
                width,
                height,
                final_analyzed_frame,
            )
        )

        first_area = (
            float(areas[0])
            if len(areas)
            else 0.0
        )

        last_area = (
            float(areas[-1])
            if len(areas)
            else 0.0
        )

        area_change_percent = (
            100.0 *
            (last_area - first_area) /
            first_area
            if first_area > 0
            else 0.0
        )

        summaries.append(
            {
                "track_id": int(track_id),
                "frames_tracked": int(frames_tracked),
                "duration_seconds": float(duration_seconds),
                "average_speed_px_s": float(mean_speed),
                "maximum_speed_px_s": float(maximum_speed),
                "total_distance_px": float(total_distance),
                "net_displacement_px": float(net_displacement),
                "confinement_ratio": float(confinement_ratio),
                "linearity_of_forward_progression": float(linearity),
                "mean_directional_change_deg": float(
                    mean_turning_angle
                ),
                "average_area_px": float(
                    np.mean(areas)
                    if len(areas)
                    else 0.0
                ),
                "maximum_area_px": float(
                    np.max(areas)
                    if len(areas)
                    else 0.0
                ),
                "area_change_percent": float(
                    area_change_percent
                ),
                "average_perimeter_px": float(
                    np.mean(perimeters)
                    if len(perimeters)
                    else 0.0
                ),
                "average_circularity": float(
                    np.mean(circularities)
                    if len(circularities)
                    else 0.0
                ),
                "average_solidity": float(
                    np.mean(solidities)
                    if len(solidities)
                    else 0.0
                ),
                "average_eccentricity": float(
                    np.mean(eccentricities)
                    if len(eccentricities)
                    else 0.0
                ),
                "average_intensity": float(
                    np.mean(intensities)
                    if len(intensities)
                    else 0.0
                ),
                "average_protrusion_area_rate_px2_s": float(
                    np.mean(protrusion_rates)
                    if len(protrusion_rates)
                    else 0.0
                ),
                "average_retraction_area_rate_px2_s": float(
                    np.mean(retraction_rates)
                    if len(retraction_rates)
                    else 0.0
                ),
                "morphodynamic_activity_index": float(
                    np.mean(shape_changes)
                    if len(shape_changes)
                    else 0.0
                ),
                "apoptosis_like_score": float(
                    death_score
                ),
                "apoptosis_like_candidate": bool(
                    death_candidate
                ),
                "start_at_border": bool(
                    start_at_border
                ),
                "end_at_border": bool(
                    end_at_border
                ),
                "start_x": float(first["x"]),
                "start_y": float(first["y"]),
                "end_x": float(last["x"]),
                "end_y": float(last["y"]),
            }
        )

    return summaries


# ============================================================
# POPULATION ANALYSIS
# ============================================================

def population_statistics(
    frame_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fps: float,
):
    if frame_df.empty:
        return {
            "initial_cells": 0,
            "final_cells": 0,
            "maximum_cells": 0,
            "minimum_cells": 0,
            "net_population_change": 0,
            "population_change_percent": 0.0,
            "apparent_growth_rate_cells_per_min": 0.0,
            "exponential_growth_rate_per_min": 0.0,
            "growth_percent_per_min": 0.0,
            "doubling_time_min": None,
            "appearance_events": 0,
            "loss_events": 0,
            "candidate_division_events": 0,
            "candidate_apoptosis_like_events": 0,
            "candidate_division_rate_per_min": 0.0,
            "candidate_apoptosis_like_rate_per_min": 0.0,
        }

    cell_counts = frame_df[
        "cell_count"
    ].to_numpy(dtype=float)

    times_min = (
        frame_df["time_seconds"].to_numpy(dtype=float)
        / 60.0
    )

    initial_cells = int(
        round(cell_counts[0])
    )

    final_cells = int(
        round(cell_counts[-1])
    )

    duration_min = (
        float(times_min[-1] - times_min[0])
        if len(times_min) > 1
        else 0.0
    )

    net_change = (
        final_cells - initial_cells
    )

    population_change_percent = (
        100.0 * net_change / initial_cells
        if initial_cells > 0
        else 0.0
    )

    apparent_growth_rate = safe_linear_slope(
        times_min,
        cell_counts,
    )

    positive_mask = (
        cell_counts > 0
    )

    exponential_growth_rate = 0.0

    if np.sum(positive_mask) >= 2:
        exponential_growth_rate = safe_linear_slope(
            times_min[positive_mask],
            np.log(
                cell_counts[positive_mask]
            ),
        )

    growth_percent_per_min = (
        100.0 *
        (math.exp(exponential_growth_rate) - 1.0)
        if exponential_growth_rate > -700
        else -100.0
    )

    doubling_time = (
        math.log(2.0) /
        exponential_growth_rate
        if exponential_growth_rate > 0
        else None
    )

    appearance_events = int(
        frame_df["new_tracks"].sum()
    )

    loss_events = int(
        frame_df["lost_tracks"].sum()
    )

    division_events = int(
        frame_df[
            "candidate_division_events"
        ].sum()
    )

    apoptosis_like_events = int(
        summary_df[
            "apoptosis_like_candidate"
        ].sum()
        if (
            not summary_df.empty
            and "apoptosis_like_candidate" in summary_df.columns
        )
        else 0
    )

    candidate_division_rate = (
        division_events / duration_min
        if duration_min > 0
        else 0.0
    )

    candidate_apoptosis_rate = (
        apoptosis_like_events / duration_min
        if duration_min > 0
        else 0.0
    )

    return {
        "initial_cells": initial_cells,
        "final_cells": final_cells,
        "maximum_cells": int(
            np.max(cell_counts)
        ),
        "minimum_cells": int(
            np.min(cell_counts)
        ),
        "net_population_change": int(
            net_change
        ),
        "population_change_percent": float(
            population_change_percent
        ),
        "apparent_growth_rate_cells_per_min": float(
            apparent_growth_rate
        ),
        "exponential_growth_rate_per_min": float(
            exponential_growth_rate
        ),
        "growth_percent_per_min": float(
            growth_percent_per_min
        ),
        "doubling_time_min": (
            float(doubling_time)
            if doubling_time is not None
            else None
        ),
        "appearance_events": appearance_events,
        "loss_events": loss_events,
        "candidate_division_events": division_events,
        "candidate_apoptosis_like_events": apoptosis_like_events,
        "candidate_division_rate_per_min": float(
            candidate_division_rate
        ),
        "candidate_apoptosis_like_rate_per_min": float(
            candidate_apoptosis_rate
        ),
        "interpretation_note": (
            "Population growth is an apparent imaging-derived measure. "
            "It can be affected by cells entering/leaving the field of view, "
            "segmentation errors and tracking errors."
        ),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.18)
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def generate_graphs(
    frame_df: pd.DataFrame,
    track_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
):
    graphs = []

    def line_plot(
        filename,
        title,
        x,
        y,
        xlabel,
        ylabel,
    ):
        plt.figure(figsize=(9, 5))
        plt.plot(
            x,
            y,
            linewidth=1.8,
        )
        save_plot(
            output_dir / filename,
            title,
            xlabel,
            ylabel,
        )
        graphs.append(filename)

    time_values = frame_df[
        "time_seconds"
    ]

    line_plot(
        "01_cell_count_vs_time.png",
        "Cell Count vs Time",
        time_values,
        frame_df["cell_count"],
        "Time (s)",
        "Cell Count",
    )

    line_plot(
        "02_moving_cells_vs_time.png",
        "Moving Cells vs Time",
        time_values,
        frame_df["moving_cells"],
        "Time (s)",
        "Moving Cells",
    )

    line_plot(
        "03_mean_speed_vs_time.png",
        "Mean Speed vs Time",
        time_values,
        frame_df["mean_speed_px_s"],
        "Time (s)",
        "Speed (px/s)",
    )

    line_plot(
        "04_median_speed_vs_time.png",
        "Median Speed vs Time",
        time_values,
        frame_df["median_speed_px_s"],
        "Time (s)",
        "Speed (px/s)",
    )

    line_plot(
        "05_max_speed_vs_time.png",
        "Maximum Speed vs Time",
        time_values,
        frame_df["max_speed_px_s"],
        "Time (s)",
        "Speed (px/s)",
    )

    line_plot(
        "06_mean_displacement_vs_time.png",
        "Mean Displacement vs Time",
        time_values,
        frame_df["mean_displacement_px"],
        "Time (s)",
        "Displacement (px)",
    )

    line_plot(
        "07_total_displacement_vs_time.png",
        "Total Displacement vs Time",
        time_values,
        frame_df["total_displacement_px"],
        "Time (s)",
        "Displacement (px)",
    )

    line_plot(
        "08_mean_cell_area_vs_time.png",
        "Mean Cell Area vs Time",
        time_values,
        frame_df["mean_cell_area_px"],
        "Time (s)",
        "Area (px²)",
    )

    line_plot(
        "09_mean_perimeter_vs_time.png",
        "Mean Perimeter vs Time",
        time_values,
        frame_df["mean_perimeter_px"],
        "Time (s)",
        "Perimeter (px)",
    )

    line_plot(
        "10_circularity_vs_time.png",
        "Cell Circularity vs Time",
        time_values,
        frame_df["mean_circularity"],
        "Time (s)",
        "Circularity",
    )

    line_plot(
        "11_solidity_vs_time.png",
        "Cell Solidity vs Time",
        time_values,
        frame_df["mean_solidity"],
        "Time (s)",
        "Solidity",
    )

    line_plot(
        "12_eccentricity_vs_time.png",
        "Cell Eccentricity vs Time",
        time_values,
        frame_df["mean_eccentricity"],
        "Time (s)",
        "Eccentricity",
    )

    line_plot(
        "13_morphodynamic_activity_vs_time.png",
        "Morphodynamic Activity vs Time",
        time_values,
        frame_df["mean_shape_change_index"],
        "Time (s)",
        "Shape Change Index",
    )

    line_plot(
        "14_protrusion_area_rate_vs_time.png",
        "Protrusion-Like Area Rate vs Time",
        time_values,
        frame_df["protrusion_area_rate_px2_s"],
        "Time (s)",
        "Protrusion-like area rate (px²/s)",
    )

    line_plot(
        "15_retraction_area_rate_vs_time.png",
        "Retraction-Like Area Rate vs Time",
        time_values,
        frame_df["retraction_area_rate_px2_s"],
        "Time (s)",
        "Retraction-like area rate (px²/s)",
    )

    line_plot(
        "16_mean_intensity_vs_time.png",
        "Mean Cell Intensity vs Time",
        time_values,
        frame_df["mean_intensity"],
        "Time (s)",
        "Mean intensity (0–255)",
    )

    # Population events
    plt.figure(figsize=(9, 5))
    plt.plot(
        time_values,
        frame_df["new_tracks"],
        label="Appearances",
        linewidth=1.7,
    )
    plt.plot(
        time_values,
        frame_df["lost_tracks"],
        label="Losses",
        linewidth=1.7,
    )
    plt.plot(
        time_values,
        frame_df["candidate_division_events"],
        label="Candidate divisions",
        linewidth=1.7,
    )
    plt.legend()
    save_plot(
        output_dir / "17_population_events_vs_time.png",
        "Population / Track Events vs Time",
        "Time (s)",
        "Events",
    )
    graphs.append(
        "17_population_events_vs_time.png"
    )

    if not summary_df.empty:
        plt.figure(figsize=(9, 5))
        plt.hist(
            summary_df["average_speed_px_s"].to_numpy(),
            bins=min(
                30,
                max(5, len(summary_df)),
            ),
        )
        save_plot(
            output_dir / "18_speed_distribution.png",
            "Speed Distribution",
            "Average Speed (px/s)",
            "Cells",
        )
        graphs.append(
            "18_speed_distribution.png"
        )

        plt.figure(figsize=(9, 5))
        plt.hist(
            summary_df["duration_seconds"].to_numpy(),
            bins=min(
                30,
                max(5, len(summary_df)),
            ),
        )
        save_plot(
            output_dir / "19_track_duration_distribution.png",
            "Track Duration Distribution",
            "Duration (s)",
            "Cells",
        )
        graphs.append(
            "19_track_duration_distribution.png"
        )

        plt.figure(figsize=(9, 5))
        population = frame_df["cell_count"].to_numpy()
        initial = population[0] if len(population) else 1
        relative = (
            population / initial
            if initial > 0
            else population
        )
        plt.plot(
            time_values,
            relative,
            linewidth=2,
        )
        save_plot(
            output_dir / "20_relative_population_vs_time.png",
            "Relative Cell Population vs Time",
            "Time (s)",
            "Population / Initial Population",
        )
        graphs.append(
            "20_relative_population_vs_time.png"
        )

        plt.figure(figsize=(9, 5))
        candidate_scores = summary_df[
            "apoptosis_like_score"
        ].to_numpy()
        plt.hist(
            candidate_scores,
            bins=10,
            range=(0, 1),
        )
        save_plot(
            output_dir / "21_apoptosis_like_score_distribution.png",
            "Apoptosis-Like Candidate Score Distribution",
            "Candidate score (0–1)",
            "Tracks",
        )
        graphs.append(
            "21_apoptosis_like_score_distribution.png"
        )

        top_distance = summary_df.nlargest(
            15,
            "total_distance_px",
        ).sort_values(
            "total_distance_px"
        )

        plt.figure(figsize=(10, 6))
        plt.barh(
            top_distance["track_id"].astype(str),
            top_distance["total_distance_px"],
        )
        save_plot(
            output_dir / "22_top_cells_total_distance.png",
            "Top Cells — Total Distance",
            "Cell",
            "Total Distance (px)",
        )
        graphs.append(
            "22_top_cells_total_distance.png"
        )

        top_speed = summary_df.nlargest(
            15,
            "average_speed_px_s",
        ).sort_values(
            "average_speed_px_s"
        )

        plt.figure(figsize=(10, 6))
        plt.barh(
            top_speed["track_id"].astype(str),
            top_speed["average_speed_px_s"],
        )
        save_plot(
            output_dir / "23_top_cells_average_speed.png",
            "Top Cells — Average Speed",
            "Cell",
            "Average Speed (px/s)",
        )
        graphs.append(
            "23_top_cells_average_speed.png"
        )

        # Full trajectories
        plt.figure(figsize=(10, 8))
        for track_id, group in track_df.groupby(
            "track_id"
        ):
            if len(group) > 1:
                plt.plot(
                    group["x"],
                    group["y"],
                    linewidth=0.7,
                    alpha=0.45,
                )

        plt.gca().invert_yaxis()
        save_plot(
            output_dir / "24_cell_trajectories.png",
            "Cell Trajectories",
            "X (px)",
            "Y (px)",
        )
        graphs.append(
            "24_cell_trajectories.png"
        )

        long_ids = summary_df.nlargest(
            20,
            "frames_tracked",
        )["track_id"].tolist()

        plt.figure(figsize=(10, 8))
        for track_id in long_ids:
            group = track_df[
                track_df["track_id"] == track_id
            ]
            if len(group) > 1:
                plt.plot(
                    group["x"],
                    group["y"],
                    linewidth=1.2,
                )

        plt.gca().invert_yaxis()
        save_plot(
            output_dir / "25_long_cell_trajectories.png",
            "Long Cell Trajectories",
            "X (px)",
            "Y (px)",
        )
        graphs.append(
            "25_long_cell_trajectories.png"
        )

        plt.figure(figsize=(9, 6))
        plt.scatter(
            summary_df["average_area_px"],
            summary_df["average_speed_px_s"],
            s=18,
            alpha=0.65,
        )
        save_plot(
            output_dir / "26_cell_area_vs_speed.png",
            "Cell Area vs Speed",
            "Average Area (px²)",
            "Average Speed (px/s)",
        )
        graphs.append(
            "26_cell_area_vs_speed.png"
        )

        plt.figure(figsize=(9, 6))
        plt.scatter(
            summary_df["morphodynamic_activity_index"],
            summary_df["average_speed_px_s"],
            s=18,
            alpha=0.65,
        )
        save_plot(
            output_dir / "27_morphodynamic_activity_vs_speed.png",
            "Morphodynamic Activity vs Cell Speed",
            "Morphodynamic activity index",
            "Average Speed (px/s)",
        )
        graphs.append(
            "27_morphodynamic_activity_vs_speed.png"
        )

        plt.figure(figsize=(9, 6))
        plt.scatter(
            summary_df["average_area_px"],
            summary_df["average_circularity"],
            s=18,
            alpha=0.65,
        )
        save_plot(
            output_dir / "28_area_vs_circularity.png",
            "Cell Area vs Circularity",
            "Average Area (px²)",
            "Average Circularity",
        )
        graphs.append(
            "28_area_vs_circularity.png"
        )

    return graphs


# ============================================================
# ANNOTATION
# ============================================================

def create_overlay(
    frame,
    prediction,
    tracker: CentroidTracker,
    detections,
):
    overlay = frame.copy()

    interior = prediction == 1
    boundary = prediction == 2

    green = np.zeros_like(frame)
    green[:] = (0, 255, 0)

    red = np.zeros_like(frame)
    red[:] = (0, 0, 255)

    if np.any(interior):
        overlay[interior] = cv2.addWeighted(
            overlay[interior],
            0.60,
            green[interior],
            0.40,
            0,
        )

    if np.any(boundary):
        overlay[boundary] = cv2.addWeighted(
            overlay[boundary],
            0.50,
            red[boundary],
            0.50,
            0,
        )

    for track_id in tracker.active_ids:
        track = tracker.tracks[track_id]

        if track.missed > 0:
            continue

        row = track.last_position
        x = int(round(row["x"]))
        y = int(round(row["y"]))

        cv2.circle(
            overlay,
            (x, y),
            5,
            (255, 255, 0),
            -1,
        )

        cv2.putText(
            overlay,
            f"#{track_id}",
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay


def create_track_only_overlay(
    frame,
    tracker: CentroidTracker,
):
    overlay = frame.copy()

    for track_id in tracker.active_ids:
        track = tracker.tracks[track_id]
        row = track.last_position

        x = int(round(row["x"]))
        y = int(round(row["y"]))

        cv2.circle(
            overlay,
            (x, y),
            4,
            (255, 255, 0),
            -1,
        )

        cv2.putText(
            overlay,
            f"#{track_id}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay


# ============================================================
# REPORT SUMMARY
# ============================================================

def build_summary(
    frame_df,
    track_df,
    summary_df,
    metadata,
    population,
):
    if frame_df.empty:
        return {
            "averageCells": 0,
            "maximumCells": 0,
            "averageMoving": 0,
            "meanSpeed": 0,
            "medianSpeed": 0,
            "maximumSpeed": 0,
            "meanDisplacement": 0,
            "totalDisplacement": 0,
            "meanArea": 0,
            "trackedObjects": 0,
            "population": population,
        }

    return {
        "averageCells": safe_float(
            frame_df["cell_count"].mean()
        ),
        "maximumCells": int(
            frame_df["cell_count"].max()
        ),
        "averageMoving": safe_float(
            frame_df["moving_cells"].mean()
        ),
        "meanSpeed": safe_float(
            frame_df["mean_speed_px_s"].mean()
        ),
        "medianSpeed": safe_float(
            frame_df["median_speed_px_s"].median()
        ),
        "maximumSpeed": safe_float(
            frame_df["max_speed_px_s"].max()
        ),
        "meanDisplacement": safe_float(
            frame_df["mean_displacement_px"].mean()
        ),
        "totalDisplacement": safe_float(
            frame_df["total_displacement_px"].sum()
        ),
        "meanArea": safe_float(
            frame_df["mean_cell_area_px"].mean()
        ),
        "meanPerimeter": safe_float(
            frame_df["mean_perimeter_px"].mean()
        ),
        "meanCircularity": safe_float(
            frame_df["mean_circularity"].mean()
        ),
        "meanSolidity": safe_float(
            frame_df["mean_solidity"].mean()
        ),
        "meanEccentricity": safe_float(
            frame_df["mean_eccentricity"].mean()
        ),
        "meanIntensity": safe_float(
            frame_df["mean_intensity"].mean()
        ),
        "meanMorphodynamicActivity": safe_float(
            frame_df["mean_shape_change_index"].mean()
        ),
        "meanProtrusionAreaRate": safe_float(
            frame_df[
                "protrusion_area_rate_px2_s"
            ].mean()
        ),
        "meanRetractionAreaRate": safe_float(
            frame_df[
                "retraction_area_rate_px2_s"
            ].mean()
        ),
        "trackedObjects": int(
            len(summary_df)
        ),
        "durationSeconds": safe_float(
            metadata["duration_seconds"]
        ),
        "frames": int(
            metadata["total_frames"]
        ),
        "fps": safe_float(
            metadata["fps"]
        ),
        "width": int(
            metadata["width"]
        ),
        "height": int(
            metadata["height"]
        ),
        "population": population,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_video(
    video_path: Path,
    output_dir: Path,
    job_id: str,
    process_every_n_frames: int,
    inference_batch_size: int,
    pixel_size_um: float | None,
):
    model = load_model()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = safe_float(
        cap.get(cv2.CAP_PROP_FPS),
        default=1.0,
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        fps = 1.0

    duration_seconds = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    stem = clean_stem(
        video_path.stem
    )

    metadata = {
        "filename": video_path.name,
        "experiment_name": stem,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "device": str(DEVICE),
        "model": "Instance U-Net",
        "segmentation": "Instance-aware",
        "classes": {
            "0": "background",
            "1": "cell interior",
            "2": "cell boundary",
        },
        "tracking": "Centroid nearest-neighbour",
        "cell_separation": "Interior/boundary connected components",
        "pixel_calibration_um_per_px": pixel_size_um,
        "analysis_frame_step": process_every_n_frames,
        "inference_batch_size": inference_batch_size,
        "speed_note": (
            "Inference is sampled every N source frames, while movement speed "
            "uses the actual source-video time interval between analyzed frames."
        ),
        "morphology_note": (
            "Shape metrics include area, perimeter, circularity, solidity, "
            "extent, eccentricity, major/minor axis, orientation and intensity."
        ),
        "protrusion_note": (
            "Protrusion/retraction metrics are boundary-remodeling proxies derived "
            "from aligned mask differences; they do not identify individual tentacles."
        ),
        "apoptosis_note": (
            "Apoptosis-like scores are morphology-based candidate events only. "
            "They are not definitive apoptosis diagnoses without appropriate biological validation."
        ),
        "proliferation_note": (
            "Population growth/doubling metrics are apparent imaging-derived measures "
            "and can be affected by field-of-view entry/exit and segmentation/tracking errors."
        ),
    }

    with jobs_lock:
        jobs[job_id].update(
            {
                "metadata": metadata,
                "progress": 0.0,
                "current_frame": 0,
                "total_frames": total_frames,
                "elapsed_seconds": 0.0,
                "frames_per_second": 0.0,
                "eta_seconds": None,
                "message": "Preparing video analysis...",
            }
        )

    # --------------------------------------------------------
    # MP4 writer
    # --------------------------------------------------------

    raw_mp4_path = (
        output_dir /
        f"{stem}_quantitative.mp4"
    )

    video_writer = None

    if width > 0 and height > 0:
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            str(raw_mp4_path),
            fourcc,
            fps,
            (width, height),
        )

        if not video_writer.isOpened():
            video_writer.release()
            video_writer = None

    tracker = CentroidTracker(
        fps=fps,
    )

    frame_records = []
    all_track_records = []

    frame_number = 0
    analysis_start = time.perf_counter()

    pending_items = []

    def process_batch(items):
        nonlocal frame_records
        nonlocal all_track_records

        inference_items = [
            item
            for item in items
            if item["should_process"]
        ]

        frames = [
            item["frame"]
            for item in inference_items
        ]

        predictions = predict_classes_batch(
            model,
            frames,
        )

        prediction_index = 0

        for item in items:
            source_frame_number = item[
                "frame_number"
            ]

            source_frame = item["frame"]

            if item["should_process"]:
                prediction = predictions[
                    prediction_index
                ]

                prediction_index += 1

                detections = detect_cells(
                    prediction,
                    source_frame,
                )

                tracker_event = tracker.update(
                    detections,
                    source_frame_number,
                    width,
                    height,
                )

                current_rows = []

                for track_id in tracker.active_ids:
                    track = tracker.tracks[track_id]

                    if (
                        track.last_frame
                        == source_frame_number
                    ):
                        current_rows.append(
                            track.last_position.copy()
                        )

                frame_metric = create_frame_metric(
                    source_frame_number,
                    fps,
                    detections,
                    prediction,
                    tracker_event,
                )

                add_motion_to_frame_metric(
                    frame_metric,
                    current_rows,
                )

                frame_records.append(
                    frame_metric
                )

                all_track_records.extend(
                    current_rows
                )

                if video_writer is not None:
                    annotated = create_overlay(
                        source_frame,
                        prediction,
                        tracker,
                        detections,
                    )

                    cv2.putText(
                        annotated,
                        f"Frame: {source_frame_number}/{total_frames}",
                        (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        annotated,
                        f"Cells: {len(detections)}",
                        (18, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        annotated,
                        f"Step: {process_every_n_frames}",
                        (18, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    video_writer.write(
                        annotated
                    )

            else:
                if video_writer is not None:
                    # Preserve the original video's temporal duration/FPS.
                    # Overlay only the most recent track coordinates.
                    annotated = create_track_only_overlay(
                        source_frame,
                        tracker,
                    )

                    video_writer.write(
                        annotated
                    )

        return predictions

    # --------------------------------------------------------
    # Read the source video in small batches.
    # This keeps memory bounded while enabling batched U-Net inference.
    # --------------------------------------------------------

    while True:
        success, frame = cap.read()

        if not success:
            break

        pending_items.append(
            {
                "frame_number": frame_number,
                "frame": frame,
                "should_process": (
                    frame_number % process_every_n_frames
                    == 0
                ),
            }
        )

        frame_number += 1

        has_enough_inference_frames = (
            sum(
                1
                for item in pending_items
                if item["should_process"]
            )
            >= inference_batch_size
        )

        if has_enough_inference_frames:
            process_batch(
                pending_items
            )
            pending_items = []

        # Status is updated frequently without waiting for completion.
        if total_frames:
            current = min(
                frame_number,
                total_frames,
            )

            elapsed = max(
                0.001,
                time.perf_counter()
                - analysis_start,
            )

            throughput = (
                current / elapsed
            )

            remaining = max(
                0,
                total_frames - current,
            )

            eta = (
                remaining / throughput
                if throughput > 0
                else None
            )

            progress = (
                100.0 * current /
                total_frames
            )

            with jobs_lock:
                jobs[job_id]["progress"] = min(
                    99.0,
                    progress,
                )
                jobs[job_id]["current_frame"] = current
                jobs[job_id]["elapsed_seconds"] = elapsed
                jobs[job_id]["frames_per_second"] = throughput
                jobs[job_id]["eta_seconds"] = eta
                jobs[job_id]["message"] = (
                    f"Analyzing frame {current:,} / {total_frames:,}"
                    f" · {progress:.0f}%"
                    f" · {throughput:.1f} source frames/s"
                )

    if pending_items:
        process_batch(
            pending_items
        )

    cap.release()

    if video_writer is not None:
        video_writer.release()

    # --------------------------------------------------------
    # Convert to H.264 if FFmpeg is installed.
    # --------------------------------------------------------

    annotated_path = raw_mp4_path
    ffmpeg = shutil.which("ffmpeg")

    if (
        ffmpeg
        and raw_mp4_path.exists()
        and raw_mp4_path.stat().st_size > 0
    ):
        h264_path = (
            output_dir /
            f"{stem}_quantitative_h264.mp4"
        )

        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_mp4_path),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(h264_path),
                ],
                check=True,
            )

            if (
                h264_path.exists()
                and h264_path.stat().st_size > 0
            ):
                raw_mp4_path.unlink(
                    missing_ok=True
                )
                annotated_path = h264_path

        except Exception as error:
            print(
                f"[CELL LAB] H.264 conversion failed: {error}"
            )
            h264_path.unlink(
                missing_ok=True
            )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    frame_df = pd.DataFrame(
        frame_records
    )

    track_columns = [
        "track_id",
        "frame",
        "time_seconds",
        "x",
        "y",
        "area_px",
        "perimeter_px",
        "circularity",
        "solidity",
        "extent",
        "eccentricity",
        "major_axis_length_px",
        "minor_axis_length_px",
        "orientation_deg",
        "equivalent_diameter_px",
        "compactness",
        "bbox_width_px",
        "bbox_height_px",
        "aspect_ratio",
        "mean_intensity",
        "intensity_std",
        "displacement_px",
        "speed_px_s",
        "area_change_rate_px2_s",
        "area_change_percent",
        "circularity_change",
        "eccentricity_change",
        "perimeter_change_px",
        "intensity_change",
        "protrusion_area_px2",
        "retraction_area_px2",
        "protrusion_area_rate_px2_s",
        "retraction_area_rate_px2_s",
        "protrusion_extent_px",
        "retraction_extent_px",
        "protrusion_extent_rate_px_s",
        "retraction_extent_rate_px_s",
        "shape_change_index",
    ]

    track_df = pd.DataFrame(
        all_track_records,
        columns=track_columns,
    )

    final_analyzed_frame = int(
        frame_df["frame"].max()
    ) if not frame_df.empty else 0

    summary_records = build_track_summary(
        tracker.tracks,
        fps,
        width,
        height,
        final_analyzed_frame,
    )

    summary_columns = [
        "track_id",
        "frames_tracked",
        "duration_seconds",
        "average_speed_px_s",
        "maximum_speed_px_s",
        "total_distance_px",
        "net_displacement_px",
        "confinement_ratio",
        "linearity_of_forward_progression",
        "mean_directional_change_deg",
        "average_area_px",
        "maximum_area_px",
        "area_change_percent",
        "average_perimeter_px",
        "average_circularity",
        "average_solidity",
        "average_eccentricity",
        "average_intensity",
        "average_protrusion_area_rate_px2_s",
        "average_retraction_area_rate_px2_s",
        "morphodynamic_activity_index",
        "apoptosis_like_score",
        "apoptosis_like_candidate",
        "start_at_border",
        "end_at_border",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]

    summary_df = pd.DataFrame(
        summary_records,
        columns=summary_columns,
    )

    # Add candidate death events back into the nearest ending frame.
    if (
        not summary_df.empty
        and not frame_df.empty
    ):
        ending_counts = defaultdict(int)

        last_frame_by_track = (
            track_df.groupby("track_id")["frame"].max().to_dict()
            if not track_df.empty
            else {}
        )

        for row in summary_df.itertuples():
            if row.apoptosis_like_candidate:
                ending_frame = int(
                    last_frame_by_track.get(
                        int(row.track_id),
                        int(frame_df["frame"].iloc[-1]),
                    )
                )
                ending_counts[ending_frame] += 1

        if ending_counts:
            frame_df["candidate_apoptosis_like_events"] = (
                frame_df["frame"].map(
                    ending_counts
                ).fillna(0).astype(int)
            )

    # --------------------------------------------------------
    # Population statistics
    # --------------------------------------------------------

    population = population_statistics(
        frame_df,
        summary_df,
        fps,
    )

    # --------------------------------------------------------
    # Save CSVs with the ORIGINAL VIDEO STEM.
    # --------------------------------------------------------

    frame_csv = output_dir / f"{stem}_frame_analysis.csv"
    track_csv = output_dir / f"{stem}_cell_tracks.csv"
    summary_csv = output_dir / f"{stem}_cell_summary.csv"

    frame_df.to_csv(
        frame_csv,
        index=False,
    )

    track_df.to_csv(
        track_csv,
        index=False,
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    graph_files = generate_graphs(
        frame_df,
        track_df,
        summary_df,
        output_dir,
    )

    # --------------------------------------------------------
    # Summary / report
    # --------------------------------------------------------

    summary = build_summary(
        frame_df,
        track_df,
        summary_df,
        metadata,
        population,
    )

    report = {
        "experiment": metadata,
        "summary": summary,
        "population_dynamics": population,
        "graphs": graph_files,
        "files": {
            "frame_analysis": frame_csv.name,
            "cell_tracks": track_csv.name,
            "cell_summary": summary_csv.name,
            "annotated_video": annotated_path.name,
            "report": f"{stem}_quantitative_report.txt",
        },
    }

    report_path = (
        output_dir /
        f"{stem}_quantitative_report.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "CELL LAB — QUANTITATIVE MICROSCOPY ANALYSIS\n"
        )
        file.write("=" * 70 + "\n\n")

        file.write("EXPERIMENT\n")
        file.write("-" * 70 + "\n")
        file.write(
            f"File: {metadata['filename']}\n"
        )
        file.write(
            f"Resolution: {width} x {height} px\n"
        )
        file.write(
            f"Frame rate: {fps:.3f} FPS\n"
        )
        file.write(
            f"Frames: {total_frames}\n"
        )
        file.write(
            f"Duration: {duration_seconds:.3f} seconds\n"
        )
        file.write(
            f"Device: {DEVICE}\n"
        )
        file.write(
            "Model: Instance U-Net\n"
        )
        file.write(
            f"Inference sampling: every {process_every_n_frames} source frames\n"
        )
        file.write(
            f"Inference batch size: {inference_batch_size}\n"
        )
        file.write(
            "Tracking: centroid nearest-neighbour\n"
        )
        file.write(
            "Cell segmentation classes: background / interior / boundary\n"
        )
        file.write(
            f"Pixel calibration: {pixel_size_um if pixel_size_um is not None else 'not provided'} um/px\n\n"
        )

        file.write("MOTILITY\n")
        file.write("-" * 70 + "\n")
        file.write(
            f"Mean speed: {summary['meanSpeed']:.4f} px/s\n"
        )
        file.write(
            f"Maximum speed: {summary['maximumSpeed']:.4f} px/s\n"
        )
        file.write(
            f"Mean displacement: {summary['meanDisplacement']:.4f} px\n"
        )
        file.write(
            f"Total displacement: {summary['totalDisplacement']:.4f} px\n\n"
        )

        file.write("MORPHOLOGY\n")
        file.write("-" * 70 + "\n")
        file.write(
            f"Mean area: {summary['meanArea']:.4f} px²\n"
        )
        file.write(
            f"Mean perimeter: {summary['meanPerimeter']:.4f} px\n"
        )
        file.write(
            f"Mean circularity: {summary['meanCircularity']:.4f}\n"
        )
        file.write(
            f"Mean solidity: {summary['meanSolidity']:.4f}\n"
        )
        file.write(
            f"Mean eccentricity: {summary['meanEccentricity']:.4f}\n"
        )
        file.write(
            f"Mean intensity: {summary['meanIntensity']:.4f}\n\n"
        )

        file.write("POPULATION DYNAMICS\n")
        file.write("-" * 70 + "\n")
        for key, value in population.items():
            if key not in {
                "interpretation_note"
            }:
                file.write(
                    f"{key}: {value}\n"
                )

        file.write("\nNOTES / LIMITATIONS\n")
        file.write("-" * 70 + "\n")
        file.write(
            metadata["protrusion_note"] + "\n"
        )
        file.write(
            metadata["apoptosis_note"] + "\n"
        )
        file.write(
            metadata["proliferation_note"] + "\n"
        )
        file.write(
            "Intensity measurements are image intensity metrics and should not be interpreted as direct metabolic measurements without calibration and an appropriate biochemical/fluorescence assay.\n"
        )

    # --------------------------------------------------------
    # Result JSON
    # --------------------------------------------------------

    result_path = (
        output_dir / "result.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    total_elapsed = max(
        0.001,
        time.perf_counter() - analysis_start,
    )

    with jobs_lock:
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["current_frame"] = total_frames
        jobs[job_id]["total_frames"] = total_frames
        jobs[job_id]["elapsed_seconds"] = total_elapsed
        jobs[job_id]["frames_per_second"] = (
            total_frames / total_elapsed
            if total_frames > 0
            else 0.0
        )
        jobs[job_id]["eta_seconds"] = 0.0
        jobs[job_id]["message"] = (
            "Analysis complete — results are ready."
        )

    return report


# ============================================================
# JOB WORKER
# ============================================================

def run_job(
    job_id: str,
    video_path: Path,
    output_dir: Path,
    process_every_n_frames: int,
    inference_batch_size: int,
    pixel_size_um: float | None,
):
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["message"] = (
                "Running Instance U-Net, morphology, tracking and population analysis..."
            )

        report = analyze_video(
            video_path=video_path,
            output_dir=output_dir,
            job_id=job_id,
            process_every_n_frames=process_every_n_frames,
            inference_batch_size=inference_batch_size,
            pixel_size_um=pixel_size_um,
        )

        with jobs_lock:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["message"] = (
                "Analysis complete."
            )
            jobs[job_id]["result"] = report

    except Exception as error:
        traceback.print_exc()

        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = str(error)
            jobs[job_id]["error"] = traceback.format_exc()


# ============================================================
# API: ANALYZE
# ============================================================

@app.post("/api/analyze")
def api_analyze():
    if "video" not in request.files:
        return jsonify(
            {
                "error": (
                    "No video was uploaded. Use multipart/form-data "
                    "with field name 'video'."
                )
            }
        ), 400

    uploaded = request.files["video"]

    if not uploaded.filename:
        return jsonify(
            {
                "error": "The uploaded file has no filename."
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
    }

    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in allowed_extensions:
        return jsonify(
            {
                "error": (
                    f"Unsupported video type: {extension}. "
                    f"Allowed: {', '.join(sorted(allowed_extensions))}"
                )
            }
        ), 400

    try:
        process_every_n_frames = int(
            request.form.get(
                "process_every_n_frames",
                DEFAULT_PROCESS_EVERY_N_FRAMES,
            )
        )
    except ValueError:
        process_every_n_frames = DEFAULT_PROCESS_EVERY_N_FRAMES

    process_every_n_frames = int(
        clamp(
            process_every_n_frames,
            1,
            10,
        )
    )

    try:
        inference_batch_size = int(
            request.form.get(
                "inference_batch_size",
                DEFAULT_INFERENCE_BATCH_SIZE,
            )
        )
    except ValueError:
        inference_batch_size = DEFAULT_INFERENCE_BATCH_SIZE

    inference_batch_size = int(
        clamp(
            inference_batch_size,
            1,
            8,
        )
    )

    pixel_size_raw = request.form.get(
        "pixel_size_um",
        "",
    ).strip()

    pixel_size_um = None

    if pixel_size_raw:
        try:
            pixel_size_um = float(
                pixel_size_raw
            )
            if pixel_size_um <= 0:
                pixel_size_um = None
        except ValueError:
            pixel_size_um = None

    job_id = uuid.uuid4().hex[:12]

    job_upload_dir = (
        UPLOAD_DIR / job_id
    )

    output_dir = (
        ANALYSIS_DIR / job_id
    )

    job_upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path = (
        job_upload_dir / original_name
    )

    uploaded.save(video_path)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_frame": 0,
            "total_frames": 0,
            "elapsed_seconds": 0.0,
            "frames_per_second": 0.0,
            "eta_seconds": None,
            "message": "Video uploaded. Waiting for analysis...",
            "filename": original_name,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=run_job,
        args=(
            job_id,
            video_path,
            output_dir,
            process_every_n_frames,
            inference_batch_size,
            pixel_size_um,
        ),
        daemon=True,
    )

    thread.start()

    return jsonify(
        {
            "job_id": job_id,
            "status": "queued",
            "filename": original_name,
            "process_every_n_frames": process_every_n_frames,
            "inference_batch_size": inference_batch_size,
            "message": "Analysis started.",
        }
    )


# ============================================================
# API: STATUS
# ============================================================

@app.get("/api/status/<job_id>")
def api_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            result_file = (
                ANALYSIS_DIR /
                job_id /
                "result.json"
            )

            if not result_file.exists():
                return jsonify(
                    {
                        "error": "Job not found."
                    }
                ), 404

            try:
                with result_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    result = json.load(file)
            except Exception:
                result = {}

            metadata = result.get(
                "experiment",
                {},
            )

            return jsonify(
                {
                    "id": job_id,
                    "status": "complete",
                    "progress": 100,
                    "current_frame": metadata.get(
                        "total_frames",
                        0,
                    ),
                    "total_frames": metadata.get(
                        "total_frames",
                        0,
                    ),
                    "elapsed_seconds": 0,
                    "frames_per_second": 0,
                    "eta_seconds": 0,
                    "message": "Analysis complete.",
                    "filename": metadata.get(
                        "filename",
                        "",
                    ),
                    "error": None,
                }
            )

        return jsonify(
            {
                "id": job["id"],
                "status": job["status"],
                "progress": job.get(
                    "progress",
                    0,
                ),
                "current_frame": job.get(
                    "current_frame",
                    0,
                ),
                "total_frames": job.get(
                    "total_frames",
                    0,
                ),
                "elapsed_seconds": job.get(
                    "elapsed_seconds",
                    0.0,
                ),
                "frames_per_second": job.get(
                    "frames_per_second",
                    0.0,
                ),
                "eta_seconds": job.get(
                    "eta_seconds"
                ),
                "message": job.get(
                    "message",
                    "",
                ),
                "filename": job.get(
                    "filename",
                    "",
                ),
                "error": job.get(
                    "error"
                ),
            }
        )


# ============================================================
# API: RESULTS
# ============================================================

@app.get("/api/results/<job_id>")
def api_results(job_id):
    result_path = (
        ANALYSIS_DIR /
        job_id /
        "result.json"
    )

    if result_path.exists():
        with result_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return jsonify(
                json.load(file)
            )

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return jsonify(
                {
                    "error": "Job not found."
                }
            ), 404

        return jsonify(
            {
                "status": job["status"],
                "progress": job.get(
                    "progress",
                    0,
                ),
                "message": job.get(
                    "message",
                    "",
                ),
                "error": job.get(
                    "error"
                ),
            }
        )


# ============================================================
# ANALYSIS FILES
# ============================================================

@app.get(
    "/analysis/<job_id>/<path:filename>"
)
def analysis_file(
    job_id,
    filename,
):
    directory = (
        ANALYSIS_DIR / job_id
    )

    requested = directory / filename

    if not requested.exists():
        # Backward compatibility for the previous frontend.
        # If it asks for a video_0 AVI/MP4, attempt to locate the
        # only quantitative video in the job directory.
        if filename.endswith(
            (".avi", ".mp4")
        ):
            matches = list(
                directory.glob(
                    "*_quantitative*.mp4"
                )
            )

            if len(matches) == 1:
                return send_from_directory(
                    directory,
                    matches[0].name,
                    as_attachment=False,
                )

    if not requested.exists():
        return jsonify(
            {
                "error": "Analysis file not found."
            }
        ), 404

    return send_from_directory(
        directory,
        filename,
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "model_exists": MODEL_PATH.exists(),
            "model": "Instance U-Net",
            "default_process_every_n_frames": DEFAULT_PROCESS_EVERY_N_FRAMES,
            "default_inference_batch_size": DEFAULT_INFERENCE_BATCH_SIZE,
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
        }
    )


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CELL LAB — QUANTITATIVE MICROSCOPY BACKEND")
    print("=" * 70)
    print(f"Base directory       : {BASE_DIR}")
    print(f"Model                : {MODEL_PATH}")
    print(f"Model exists         : {MODEL_PATH.exists()}")
    print(f"Device               : {DEVICE}")
    print(f"Inference step       : every {DEFAULT_PROCESS_EVERY_N_FRAMES} source frames")
    print(f"Inference batch size : {DEFAULT_INFERENCE_BATCH_SIZE}")
    print(f"FFmpeg available     : {shutil.which('ffmpeg') is not None}")
    print(f"Uploads              : {UPLOAD_DIR}")
    print(f"Analysis             : {ANALYSIS_DIR}")
    print("=" * 70)
    print("\nStarting server at http://127.0.0.1:5000\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )
