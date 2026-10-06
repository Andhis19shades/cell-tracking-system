import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects

from model_instance import InstanceUNet


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = Path(
    "/Users/abhyudaysingh/Downloads/videos 2/video_0.avi"
)

MODEL_PATH = Path(
    "/Users/abhyudaysingh/instance_unet_model.pth"
)

OUTPUT_DIR = Path(
    "/Users/abhyudaysingh/Downloads/quantitative_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_SIZE = 512

# Neural network thresholds
INTERIOR_THRESHOLD = 0.35
FOREGROUND_THRESHOLD = 0.30

# Object filtering
MIN_CELL_AREA = 150
MAX_CELL_AREA = 50000

# Watershed
MIN_PEAK_DISTANCE = 12
MIN_PEAK_HEIGHT = 2.0

# Tracking
MAX_MATCH_DISTANCE = 45.0
MAX_MISSING_FRAMES = 4

# Movement threshold
MOVEMENT_THRESHOLD = 1.5


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = InstanceUNet()

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model.to(DEVICE)
model.eval()

print("Model loaded.")


# ============================================================
# TRACK STRUCTURE
# ============================================================

tracks = {}

next_track_id = 1


# ============================================================
# CREATE TRACK
# ============================================================

def create_track(
    centroid,
    area,
    frame_number
):

    global next_track_id

    track_id = next_track_id

    next_track_id += 1

    tracks[track_id] = {

        "positions": [
            tuple(centroid)
        ],

        "areas": [
            area
        ],

        "frames": [
            frame_number
        ],

        "speeds": [],

        "displacements": [],

        "missing": 0,

        "active": True
    }

    return track_id


# ============================================================
# MATCH DETECTIONS
# ============================================================

def match_detections(
    detections,
    frame_number,
    fps
):

    active_tracks = []

    for track_id, track in tracks.items():

        if track["active"]:

            active_tracks.append(
                track_id
            )

    candidates = []

    for track_id in active_tracks:

        last_position = np.array(
            tracks[track_id]["positions"][-1],
            dtype=np.float32
        )

        for detection_index, detection in enumerate(detections):

            centroid = np.array(
                detection["centroid"],
                dtype=np.float32
            )

            distance = np.linalg.norm(
                centroid - last_position
            )

            if distance <= MAX_MATCH_DISTANCE:

                candidates.append(
                    (
                        distance,
                        track_id,
                        detection_index
                    )
                )

    candidates.sort(
        key=lambda x: x[0]
    )

    used_tracks = set()
    used_detections = set()

    assignments = []

    for distance, track_id, detection_index in candidates:

        if track_id in used_tracks:
            continue

        if detection_index in used_detections:
            continue

        used_tracks.add(
            track_id
        )

        used_detections.add(
            detection_index
        )

        assignments.append(
            (
                track_id,
                detection_index,
                distance
            )
        )

    # --------------------------------------------------------
    # UPDATE MATCHED TRACKS
    # --------------------------------------------------------

    for track_id, detection_index, distance in assignments:

        detection = detections[
            detection_index
        ]

        track = tracks[
            track_id
        ]

        previous_position = np.array(
            track["positions"][-1]
        )

        current_position = np.array(
            detection["centroid"]
        )

        displacement = float(
            np.linalg.norm(
                current_position -
                previous_position
            )
        )

        speed = displacement * fps

        track["positions"].append(
            tuple(current_position)
        )

        track["areas"].append(
            detection["area"]
        )

        track["frames"].append(
            frame_number
        )

        track["displacements"].append(
            displacement
        )

        track["speeds"].append(
            speed
        )

        track["missing"] = 0

        track["active"] = True

        detection["track_id"] = track_id

        detection["speed"] = speed

        detection["displacement"] = displacement

    # --------------------------------------------------------
    # UNMATCHED TRACKS
    # --------------------------------------------------------

    for track_id in active_tracks:

        if track_id not in used_tracks:

            tracks[track_id]["missing"] += 1

            if (
                tracks[track_id]["missing"]
                > MAX_MISSING_FRAMES
            ):

                tracks[track_id]["active"] = False

    # --------------------------------------------------------
    # CREATE NEW TRACKS
    # --------------------------------------------------------

    for detection_index, detection in enumerate(detections):

        if detection_index in used_detections:
            continue

        track_id = create_track(
            detection["centroid"],
            detection["area"],
            frame_number
        )

        detection["track_id"] = track_id

        detection["speed"] = 0.0

        detection["displacement"] = 0.0

    return detections


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_frame(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    original_height, original_width = gray.shape

    resized = cv2.resize(
        gray,
        (
            INPUT_SIZE,
            INPUT_SIZE
        )
    )

    image = resized.astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    interior_probability = (
        probabilities[1]
        .detach()
        .cpu()
        .numpy()
    )

    boundary_probability = (
        probabilities[2]
        .detach()
        .cpu()
        .numpy()
    )

    # Combined cell probability
    cell_probability = (
        interior_probability +
        boundary_probability
    )

    # --------------------------------------------------------
    # FOREGROUND
    # --------------------------------------------------------

    foreground = (
        cell_probability
        > FOREGROUND_THRESHOLD
    )

    # Remove small noise
    foreground = remove_small_objects(
        foreground,
        min_size=80
    )

    foreground = ndimage.binary_opening(
        foreground,
        structure=np.ones(
            (3, 3)
        )
    )

    foreground = ndimage.binary_closing(
        foreground,
        structure=np.ones(
            (5, 5)
        )
    )

    # --------------------------------------------------------
    # DISTANCE TRANSFORM
    # --------------------------------------------------------

    distance = ndimage.distance_transform_edt(
        foreground
    )

    # --------------------------------------------------------
    # FIND CELL CENTERS
    # --------------------------------------------------------

    coordinates = peak_local_max(

        distance,

        min_distance=MIN_PEAK_DISTANCE,

        threshold_abs=MIN_PEAK_HEIGHT,

        labels=foreground
    )

    markers = np.zeros_like(
        distance,
        dtype=np.int32
    )

    for marker_id, coordinate in enumerate(
        coordinates,
        start=1
    ):

        y, x = coordinate

        markers[
            y,
            x
        ] = marker_id

    # --------------------------------------------------------
    # WATERSHED
    # --------------------------------------------------------

    labels = watershed(
        -distance,
        markers,
        mask=foreground
    )

    # --------------------------------------------------------
    # CONVERT OBJECTS TO DETECTIONS
    # --------------------------------------------------------

    detections = []

    number_of_objects = labels.max()

    for object_id in range(
        1,
        number_of_objects + 1
    ):

        object_mask = (
            labels ==
            object_id
        )

        area = int(
            object_mask.sum()
        )

        if area < MIN_CELL_AREA:
            continue

        if area > MAX_CELL_AREA:
            continue

        moments = cv2.moments(
            object_mask.astype(
                np.uint8
            )
        )

        if moments["m00"] == 0:
            continue

        cx = (
            moments["m10"] /
            moments["m00"]
        )

        cy = (
            moments["m01"] /
            moments["m00"]
        )

        detections.append({

            "centroid": (
                cx,
                cy
            ),

            "area": area,

            "mask": object_mask

        })

    # --------------------------------------------------------
    # SCALE CENTROIDS TO ORIGINAL IMAGE
    # --------------------------------------------------------

    scale_x = (
        original_width /
        INPUT_SIZE
    )

    scale_y = (
        original_height /
        INPUT_SIZE
    )

    for detection in detections:

        x, y = detection[
            "centroid"
        ]

        detection[
            "centroid"
        ] = (

            x * scale_x,

            y * scale_y
        )

        detection[
            "area"
        ] = int(
            detection["area"] *
            scale_x *
            scale_y
        )

    return detections


# ============================================================
# DRAW ARROW
# ============================================================

def draw_motion_arrow(
    frame,
    centroid,
    previous_centroid
):

    if previous_centroid is None:
        return

    x1, y1 = map(
        int,
        previous_centroid
    )

    x2, y2 = map(
        int,
        centroid
    )

    distance = np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    if distance < MOVEMENT_THRESHOLD:
        return

    cv2.arrowedLine(

        frame,

        (
            x1,
            y1
        ),

        (
            x2,
            y2
        ),

        (0, 255, 255),

        2,

        tipLength=0.3
    )


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_panel(
    frame,
    frame_number,
    fps,
    detections
):

    height, width = frame.shape[:2]

    active_speeds = []

    for detection in detections:

        speed = detection.get(
            "speed",
            0
        )

        if speed > 0:

            active_speeds.append(
                speed
            )

    cell_count = len(
        detections
    )

    moving_count = sum(

        1

        for detection in detections

        if detection.get(
            "speed",
            0
        ) >= MOVEMENT_THRESHOLD
    )

    if active_speeds:

        mean_speed = np.mean(
            active_speeds
        )

        median_speed = np.median(
            active_speeds
        )

        maximum_speed = np.max(
            active_speeds
        )

    else:

        mean_speed = 0
        median_speed = 0
        maximum_speed = 0

    time_seconds = (
        frame_number /
        fps
    )

    # --------------------------------------------------------
    # PANEL
    # --------------------------------------------------------

    panel_height = 180

    overlay = frame.copy()

    cv2.rectangle(

        overlay,

        (
            0,
            0
        ),

        (
            width,
            panel_height
        ),

        (15, 15, 15),

        -1
    )

    frame[:] = cv2.addWeighted(

        overlay,

        0.80,

        frame,

        0.20,

        0
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [

        f"TIME: {time_seconds:.2f} s",

        f"FRAME: {frame_number}",

        f"CELLS: {cell_count}",

        f"MOVING: {moving_count}",

        f"MEAN SPEED: {mean_speed:.2f} px/s",

        f"MEDIAN SPEED: {median_speed:.2f} px/s",

        f"MAX SPEED: {maximum_speed:.2f} px/s"

    ]

    y = 28

    for line in lines:

        cv2.putText(

            frame,

            line,

            (
                15,
                y
            ),

            font,

            0.65,

            (255, 255, 255),

            2,

            cv2.LINE_AA
        )

        y += 24


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("Opening video...")

capture = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not capture.isOpened():

    raise RuntimeError(
        "Could not open video."
    )


fps = capture.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 8.0


total_frames = int(
    capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

width = int(
    capture.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

output_video = (
    OUTPUT_DIR /
    "video_0_quantitative.avi"
)

fourcc = cv2.VideoWriter_fourcc(
    *"MJPG"
)

writer = cv2.VideoWriter(

    str(output_video),

    fourcc,

    fps,

    (
        width,
        height
    )
)

if not writer.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# DATA STORAGE
# ============================================================

frame_results = []

detection_results = []

previous_centroids = {}


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

print()
print(
    f"Processing {total_frames} frames..."
)

while True:

    success, frame = capture.read()

    if not success:
        break

    frame_number += 1

    detections = predict_frame(
        frame
    )

    detections = match_detections(

        detections,

        frame_number,

        fps
    )

    # --------------------------------------------------------
    # CURRENT FRAME STATISTICS
    # --------------------------------------------------------

    speeds = [

        d.get(
            "speed",
            0
        )

        for d in detections

        if d.get(
            "speed",
            0
        ) > 0
    ]

    displacements = [

        d.get(
            "displacement",
            0
        )

        for d in detections
    ]

    moving_count = sum(

        1

        for d in detections

        if d.get(
            "speed",
            0
        ) >= MOVEMENT_THRESHOLD
    )

    if speeds:

        mean_speed = float(
            np.mean(speeds)
        )

        median_speed = float(
            np.median(speeds)
        )

        max_speed = float(
            np.max(speeds)
        )

    else:

        mean_speed = 0.0
        median_speed = 0.0
        max_speed = 0.0

    mean_displacement = (

        float(
            np.mean(
                displacements
            )
        )

        if displacements

        else 0.0
    )

    total_displacement = float(
        np.sum(
            displacements
        )
    )

    mean_area = (

        float(
            np.mean(
                [
                    d["area"]
                    for d in detections
                ]
            )
        )

        if detections

        else 0.0
    )

    # --------------------------------------------------------
    # SAVE FRAME STATISTICS
    # --------------------------------------------------------

    frame_results.append({

        "frame": frame_number,

        "time_seconds":
            frame_number / fps,

        "cell_count":
            len(detections),

        "moving_cells":
            moving_count,

        "mean_speed_px_s":
            mean_speed,

        "median_speed_px_s":
            median_speed,

        "max_speed_px_s":
            max_speed,

        "mean_displacement_px":
            mean_displacement,

        "total_displacement_px":
            total_displacement,

        "mean_cell_area_px":
            mean_area

    })

    # --------------------------------------------------------
    # DRAW CELLS
    # --------------------------------------------------------

    for detection in detections:

        x, y = detection[
            "centroid"
        ]

        x = int(x)
        y = int(y)

        track_id = detection[
            "track_id"
        ]

        speed = detection.get(
            "speed",
            0.0
        )

        displacement = detection.get(
            "displacement",
            0.0
        )

        area = detection[
            "area"
        ]

        # Previous position
        previous_position = None

        if track_id in tracks:

            positions = tracks[
                track_id
            ]["positions"]

            if len(positions) >= 2:

                previous_position = (
                    positions[-2]
                )

        # Draw trajectory segment
        draw_motion_arrow(

            frame,

            (
                x,
                y
            ),

            previous_position
        )

        # ----------------------------------------------------
        # CELL OUTLINE
        # ----------------------------------------------------

        mask = detection[
            "mask"
        ]

        # Resize mask to original resolution
        mask_original = cv2.resize(

            mask.astype(
                np.uint8
            ),

            (
                width,
                height
            ),

            interpolation=cv2.INTER_NEAREST
        )

        contours, _ = cv2.findContours(

            mask_original,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(

            frame,

            contours,

            -1,

            (0, 255, 0),

            2
        )

        # ----------------------------------------------------
        # CENTROID
        # ----------------------------------------------------

        cv2.circle(

            frame,

            (
                x,
                y
            ),

            4,

            (255, 255, 0),

            -1
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"#{track_id} "
            f"{speed:.1f}px/s"
        )

        cv2.putText(

            frame,

            label,

            (
                x + 6,
                y - 6
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # SAVE INDIVIDUAL DATA
        # ----------------------------------------------------

        detection_results.append({

            "frame":
                frame_number,

            "time_seconds":
                frame_number / fps,

            "track_id":
                track_id,

            "x":
                x,

            "y":
                y,

            "area_px":
                area,

            "displacement_px":
                displacement,

            "speed_px_s":
                speed

        })

    # --------------------------------------------------------
    # PANEL
    # --------------------------------------------------------

    draw_panel(

        frame,

        frame_number,

        fps,

        detections
    )

    writer.write(
        frame
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        frame_number % 25 == 0
        or frame_number == total_frames
    ):

        print(

            f"\rProcessed "
            f"{frame_number}/"
            f"{total_frames}",

            end="",
            flush=True
        )


# ============================================================
# RELEASE
# ============================================================

capture.release()
writer.release()

print()
print()
print("Video processing complete.")


# ============================================================
# SAVE FRAME DATA
# ============================================================

frame_df = pd.DataFrame(
    frame_results
)

frame_csv = (
    OUTPUT_DIR /
    "video_0_frame_analysis.csv"
)

frame_df.to_csv(
    frame_csv,
    index=False
)


# ============================================================
# SAVE DETECTION DATA
# ============================================================

detection_df = pd.DataFrame(
    detection_results
)

detection_csv = (
    OUTPUT_DIR /
    "video_0_cell_tracks.csv"
)

detection_df.to_csv(
    detection_csv,
    index=False
)


# ============================================================
# CELL SUMMARY
# ============================================================

summary_rows = []

for track_id, track in tracks.items():

    positions = track[
        "positions"
    ]

    speeds = track[
        "speeds"
    ]

    displacements = track[
        "displacements"
    ]

    areas = track[
        "areas"
    ]

    if len(positions) < 2:
        continue

    total_distance = float(
        np.sum(
            displacements
        )
    )

    average_speed = (

        float(
            np.mean(speeds)
        )

        if speeds

        else 0.0
    )

    maximum_speed = (

        float(
            np.max(speeds)
        )

        if speeds

        else 0.0
    )

    duration = (

        (
            track["frames"][-1]
            -
            track["frames"][0]
        )
        / fps
    )

    start_x, start_y = (
        positions[0]
    )

    end_x, end_y = (
        positions[-1]
    )

    net_displacement = float(

        np.sqrt(

            (
                end_x -
                start_x
            ) ** 2

            +

            (
                end_y -
                start_y
            ) ** 2
        )
    )

    summary_rows.append({

        "track_id":
            track_id,

        "start_frame":
            track["frames"][0],

        "end_frame":
            track["frames"][-1],

        "duration_seconds":
            duration,

        "frames_tracked":
            len(positions),

        "start_x":
            start_x,

        "start_y":
            start_y,

        "end_x":
            end_x,

        "end_y":
            end_y,

        "net_displacement_px":
            net_displacement,

        "total_distance_px":
            total_distance,

        "average_speed_px_s":
            average_speed,

        "maximum_speed_px_s":
            maximum_speed,

        "average_area_px":
            np.mean(areas),

        "maximum_area_px":
            np.max(areas)

    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_csv = (
    OUTPUT_DIR /
    "video_0_cell_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report_file = (
    OUTPUT_DIR /
    "video_0_quantitative_report.txt"
)

if len(detection_df) > 0:

    overall_mean_speed = (
        detection_df[
            "speed_px_s"
        ]
        .replace(
            0,
            np.nan
        )
        .mean()
    )

    overall_max_speed = (
        detection_df[
            "speed_px_s"
        ].max()
    )

else:

    overall_mean_speed = 0
    overall_max_speed = 0


with open(
    report_file,
    "w"
) as f:

    f.write(
        "QUANTITATIVE CELL MOTION ANALYSIS\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        f"Video: {VIDEO_PATH.name}\n"
    )

    f.write(
        f"Resolution: {width} x {height}\n"
    )

    f.write(
        f"FPS: {fps:.2f}\n"
    )

    f.write(
        f"Frames: {total_frames}\n"
    )

    f.write(
        f"Duration: "
        f"{total_frames / fps:.2f} seconds\n\n"
    )

    f.write(
        "DETECTION STATISTICS\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        f"Average cells/frame: "
        f"{frame_df['cell_count'].mean():.2f}\n"
    )

    f.write(
        f"Maximum cells/frame: "
        f"{frame_df['cell_count'].max()}\n"
    )

    f.write(
        f"Average moving cells/frame: "
        f"{frame_df['moving_cells'].mean():.2f}\n"
    )

    f.write(
        f"Average cell area: "
        f"{frame_df['mean_cell_area_px'].mean():.2f} px²\n\n"
    )

    f.write(
        "MOTION STATISTICS\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        f"Average speed: "
        f"{overall_mean_speed:.2f} px/s\n"
    )

    f.write(
        f"Maximum speed: "
        f"{overall_max_speed:.2f} px/s\n"
    )

    f.write(
        f"Tracked objects: "
        f"{len(summary_df)}\n\n"
    )

    f.write(
        "NOTE\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        "Speed is reported in pixels/second because "
        "microscope pixel-size calibration has not "
        "been provided.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("================================================")
print("QUANTITATIVE ANALYSIS COMPLETE")
print("================================================")

print()
print("Annotated video:")
print(output_video)

print()
print("Frame analysis:")
print(frame_csv)

print()
print("Cell tracks:")
print(detection_csv)

print()
print("Cell summary:")
print(summary_csv)

print()
print("Report:")
print(report_file)

print()
print("Results folder:")
print(OUTPUT_DIR)
