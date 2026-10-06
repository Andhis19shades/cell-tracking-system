import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

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

VIDEO_DIR = Path(
    "/Users/abhyudaysingh/Downloads/videos 2"
)

VIDEO_PATH = (
    VIDEO_DIR / "video_0.avi"
)

MODEL_PATH = (
    "/Users/abhyudaysingh/instance_unet_model.pth"
)

OUTPUT_DIR = Path(
    "/Users/abhyudaysingh/Downloads/"
    "instance_video_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

INPUT_SIZE = 512

MIN_CELL_AREA = 100

MAX_CELL_AREA = 100000

MAX_TRACK_DISTANCE = 80

MIN_PEAK_DISTANCE = 12

MIN_PEAK_HEIGHT = 3


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Instance U-Net...")

model = InstanceUNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# WATERSHED CELL SEPARATION
# ============================================================

def separate_cells(interior_mask):

    # Convert to boolean
    interior_mask = (
        interior_mask > 0
    )

    # Remove tiny noise
    interior_mask = ndimage.binary_opening(
        interior_mask,
        structure=np.ones((3, 3))
    )

    # Fill holes
    interior_mask = ndimage.binary_fill_holes(
        interior_mask
    )

    # Distance transform
    distance = distance_transform_edt(
        interior_mask
    )

    # Find cell centers
    coordinates = peak_local_max(
        distance,
        min_distance=MIN_PEAK_DISTANCE,
        threshold_abs=MIN_PEAK_HEIGHT,
        labels=interior_mask
    )

    # Create markers
    markers = np.zeros_like(
        distance,
        dtype=np.int32
    )

    for marker_id, coordinate in enumerate(
        coordinates,
        start=1
    ):

        y, x = coordinate

        markers[y, x] = marker_id

    # If no peaks were found,
    # use connected components
    if markers.max() == 0:

        labels = ndimage.label(
            interior_mask
        )[0]

        return labels

    # Watershed
    labels = watershed(
        -distance,
        markers,
        mask=interior_mask
    )

    # Clean small/huge regions
    cleaned = np.zeros_like(
        labels,
        dtype=np.int32
    )

    new_id = 1

    for cell_id in np.unique(labels):

        if cell_id == 0:
            continue

        area = np.sum(
            labels == cell_id
        )

        if (
            area >= MIN_CELL_AREA
            and
            area <= MAX_CELL_AREA
        ):

            cleaned[
                labels == cell_id
            ] = new_id

            new_id += 1

    return cleaned


# ============================================================
# TRACK MATCHING
# ============================================================

def match_tracks(
    previous_cells,
    current_cells,
    next_track_id
):

    candidates = []

    # Compare current cells with
    # previous-frame cells
    for current_id, current in (
        current_cells.items()
    ):

        cy, cx = current["centroid"]

        for previous_id, previous in (
            previous_cells.items()
        ):

            py, px = previous["centroid"]

            distance = np.sqrt(
                (cx - px) ** 2 +
                (cy - py) ** 2
            )

            if distance <= MAX_TRACK_DISTANCE:

                candidates.append(
                    (
                        distance,
                        current_id,
                        previous_id
                    )
                )

    # Closest pairs first
    candidates.sort(
        key=lambda x: x[0]
    )

    assignments = {}

    used_current = set()

    used_previous = set()

    for (
        distance,
        current_id,
        previous_id
    ) in candidates:

        if current_id in used_current:
            continue

        if previous_id in used_previous:
            continue

        assignments[current_id] = (
            previous_id
        )

        used_current.add(
            current_id
        )

        used_previous.add(
            previous_id
        )

    # Assign track IDs
    track_mapping = {}

    for current_id in current_cells:

        if current_id in assignments:

            previous_id = assignments[
                current_id
            ]

            track_mapping[current_id] = (
                previous_cells[
                    previous_id
                ]["track_id"]
            )

        else:

            track_mapping[current_id] = (
                next_track_id
            )

            next_track_id += 1

    return (
        track_mapping,
        next_track_id
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path):

    print()
    print("=" * 70)
    print("PROCESSING VIDEO")
    print("=" * 70)

    print(
        "Video:",
        video_path
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print()
        print("ERROR: Could not open video.")
        print(video_path)
        return

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

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

    print()
    print("Video Information")
    print("----------------------------")

    print(
        "Resolution:",
        width,
        "x",
        height
    )

    print(
        "FPS:",
        fps
    )

    print(
        "Total Frames:",
        total_frames
    )

    print(
        "Duration:",
        round(
            total_frames / fps,
            2
        ),
        "seconds"
    )
    
    # ========================================================
    # OUTPUT VIDEO
    # ========================================================#
    output_video = (
        OUTPUT_DIR /
        f"{video_path.stem}_tracked.avi"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print(
            "ERROR: Could not create output video."
        )
    # ========================================================
    # TRACKING VARIABLES
    # ========================================================

    previous_cells = {}

    next_track_id = 1

    records = []

    frame_number = 0


    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break


        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        gray = gray.astype(
            np.float32
        )

        min_value = gray.min()

        max_value = gray.max()

        if max_value > min_value:

            gray = (
                gray - min_value
            ) / (
                max_value - min_value
            )

        else:

            gray = np.zeros_like(
                gray
            )


        # ----------------------------------------------------
        # Resize to 512 × 512
        # ----------------------------------------------------

        resized = cv2.resize(
            gray,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            resized
        ).float()

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )


        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            )

            prediction = (
                prediction
                .squeeze(0)
                .cpu()
                .numpy()
            )


        # ----------------------------------------------------
        # Classes
        #
        # 0 = background
        # 1 = cell interior
        # 2 = cell boundary
        # ----------------------------------------------------

        interior = (
            prediction == 1
        )

        boundary = (
            prediction == 2
        )


        # ----------------------------------------------------
        # Resize masks back to video size
        # ----------------------------------------------------

        interior = cv2.resize(
            interior.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        boundary = cv2.resize(
            boundary.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)


        # ====================================================
        # WATERSHED
        # ====================================================

        labels = separate_cells(
            interior
        )


        # ====================================================
        # EXTRACT CELL INFORMATION
        # ====================================================

        current_cells = {}

        for cell_id in np.unique(labels):

            if cell_id == 0:
                continue

            ys, xs = np.where(
                labels == cell_id
            )

            if len(xs) == 0:
                continue

            area = len(xs)

            centroid_x = float(
                np.mean(xs)
            )

            centroid_y = float(
                np.mean(ys)
            )

            current_cells[cell_id] = {
                "centroid": (
                    centroid_y,
                    centroid_x
                ),
                "area": area
            }


        # ====================================================
        # TRACK CELLS
        # ====================================================

        (
            track_mapping,
            next_track_id
        ) = match_tracks(
            previous_cells,
            current_cells,
            next_track_id
        )


        # ====================================================
        # RECORD MOTION
        # ====================================================

        for cell_id, cell in (
            current_cells.items()
        ):

            track_id = track_mapping[
                cell_id
            ]

            cy, cx = cell[
                "centroid"
            ]

            displacement = 0.0

            speed = 0.0

            # Find previous position
            for previous_id, previous in (
                previous_cells.items()
            ):

                if (
                    previous["track_id"]
                    == track_id
                ):

                    py, px = previous[
                        "centroid"
                    ]

                    displacement = np.sqrt(
                        (cx - px) ** 2 +
                        (cy - py) ** 2
                    )

                    speed = (
                        displacement * fps
                    )

                    break


            records.append({
                "frame": frame_number,

                "time_seconds":
                    frame_number / fps,

                "track_id":
                    track_id,

                "area_pixels":
                    cell["area"],

                "centroid_x":
                    cx,

                "centroid_y":
                    cy,

                "displacement_pixels":
                    displacement,

                "speed_pixels_per_second":
                    speed
            })


        # ====================================================
        # UPDATE PREVIOUS FRAME
        # ====================================================

        previous_cells = {}

        for cell_id, cell in (
            current_cells.items()
        ):

            previous_cells[cell_id] = {

                "centroid":
                    cell["centroid"],

                "area":
                    cell["area"],

                "track_id":
                    track_mapping[cell_id]
            }


        # ====================================================
        # CREATE VISUALIZATION
        # ====================================================

        output_frame = frame.copy()


        # ----------------------------------------------------
        # Cell interior = green
        # ----------------------------------------------------

        output_frame[
            interior
        ] = (
            0.65 *
            output_frame[interior]
            +
            0.35 *
            np.array(
                [0, 255, 0]
            )
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Cell boundary = red
        # ----------------------------------------------------

        output_frame[
            boundary
        ] = (
            0.50 *
            output_frame[boundary]
            +
            0.50 *
            np.array(
                [0, 0, 255]
            )
        ).astype(
            np.uint8
        )


        # ====================================================
        # DRAW CELL CENTERS + IDs
        # ====================================================

        for cell_id, cell in (
            current_cells.items()
        ):

            track_id = track_mapping[
                cell_id
            ]

            cy, cx = cell[
                "centroid"
            ]

            cv2.circle(
                output_frame,
                (
                    int(cx),
                    int(cy)
                ),
                4,
                (255, 255, 0),
                -1
            )

            cv2.putText(
                output_frame,
                f"ID {track_id}",
                (
                    int(cx) + 6,
                    int(cy)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )


        # ====================================================
        # DISPLAY INFORMATION
        # ====================================================

        cv2.putText(
            output_frame,
            f"Frame: {frame_number}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output_frame,
            f"Cells: {len(current_cells)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output_frame,
            f"Tracks: {next_track_id - 1}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # WRITE OUTPUT FRAME
        # ====================================================

        writer.write(
            output_frame
        )


        frame_number += 1


        # ====================================================
        # PROGRESS
        # ====================================================

        if frame_number % 25 == 0:

            percentage = (
                100 *
                frame_number /
                total_frames
            )

            print(
                f"Progress: "
                f"{frame_number}/{total_frames} "
                f"({percentage:.1f}%)"
            )


    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    writer.release()


    # ========================================================
    # SAVE CSV
    # ========================================================

    dataframe = pd.DataFrame(
        records
    )

    csv_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_tracks.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

    if len(dataframe) > 0:

        cells_per_frame = (
            dataframe
            .groupby("frame")
            .size()
        )

        average_cells = (
            cells_per_frame.mean()
        )

        maximum_cells = (
            cells_per_frame.max()
        )

        average_area = (
            dataframe[
                "area_pixels"
            ].mean()
        )

        average_speed = (
            dataframe[
                "speed_pixels_per_second"
            ].mean()
        )

        maximum_speed = (
            dataframe[
                "speed_pixels_per_second"
            ].max()
        )

    else:

        average_cells = 0

        maximum_cells = 0

        average_area = 0

        average_speed = 0

        maximum_speed = 0


    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:

        file.write(
            "INSTANCE-AWARE CELL "
            "SEGMENTATION AND "
            "MOTION ANALYSIS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Video: {video_path.name}\n\n"
        )

        file.write(
            "VIDEO INFORMATION\n"
        )

        file.write(
            "-" * 40 + "\n"
        )

        file.write(
            f"Resolution: "
            f"{width} x {height}\n"
        )

        file.write(
            f"FPS: {fps:.2f}\n"
        )

        file.write(
            f"Total Frames: "
            f"{total_frames}\n"
        )

        file.write(
            f"Duration: "
            f"{total_frames / fps:.2f} seconds\n\n"
        )

        file.write(
            "CELL DETECTION\n"
        )

        file.write(
            "-" * 40 + "\n"
        )

        file.write(
            f"Average Cells / Frame: "
            f"{average_cells:.2f}\n"
        )

        file.write(
            f"Maximum Cells / Frame: "
            f"{maximum_cells}\n"
        )

        file.write(
            f"Total Track IDs: "
            f"{next_track_id - 1}\n"
        )

        file.write(
            f"Average Cell Area: "
            f"{average_area:.2f} pixels\n\n"
        )

        file.write(
            "MOTION ANALYSIS\n"
        )

        file.write(
            "-" * 40 + "\n"
        )

        file.write(
            f"Average Speed: "
            f"{average_speed:.2f} "
            f"pixels/second\n"
        )

        file.write(
            f"Maximum Speed: "
            f"{maximum_speed:.2f} "
            f"pixels/second\n\n"
        )

        file.write(
            "NOTE:\n"
        )

        file.write(
            "Speed is reported in pixels/second "
            "because microscope pixel-size "
            "calibration has not been provided.\n"
        )


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 70)

    print()
    print("Output Video:")
    print(output_video)

    print()
    print("CSV:")
    print(csv_path)

    print()
    print("Report:")
    print(report_path)


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("SINGLE VIDEO INSTANCE PIPELINE")
print("=" * 70)

if not VIDEO_PATH.exists():

    print()
    print("ERROR: Video does not exist:")
    print(VIDEO_PATH)

else:

    process_video(
        VIDEO_PATH
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print("Results folder:")
print(OUTPUT_DIR)
