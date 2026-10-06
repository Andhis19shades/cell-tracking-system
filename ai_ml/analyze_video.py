import os
import cv2
import torch
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

from pathlib import Path

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

VIDEO_DIR = Path(
    "/Users/abhyudaysingh/Downloads/videos 2"
)

MODEL_PATH = Path(
    "/Users/abhyudaysingh/unet_model.pth"
)

OUTPUT_DIR = Path(
    "/Users/abhyudaysingh/Downloads/video_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

print("Loading U-Net...")

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
# PARAMETERS
# ============================================================

INPUT_SIZE = 512

# Ignore tiny segmentation noise
MIN_OBJECT_AREA = 20

# Maximum distance an object can move between frames
MAX_MATCH_DISTANCE = 80


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_frame(frame):

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = frame

    gray = gray.astype(np.float32)

    # Normalize
    min_val = gray.min()
    max_val = gray.max()

    if max_val > min_val:
        gray = (
            gray - min_val
        ) / (
            max_val - min_val
        )

    # Resize to model input
    resized = cv2.resize(
        gray,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = torch.from_numpy(
        resized
    ).float()

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def get_prediction(frame):

    tensor = preprocess_frame(frame)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(output)

    probability = probability[0, 0]

    probability_np = (
        probability
        .detach()
        .cpu()
        .numpy()
    )

    # Binary segmentation
    binary = (
        probability_np > 0.5
    ).astype(np.uint8)

    return probability_np, binary


def get_objects(binary, probability):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    objects = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_OBJECT_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        cx = centroids[i][0]
        cy = centroids[i][1]

        component_mask = (
            labels == i
        )

        confidence = probability[
            component_mask
        ].mean()

        objects.append({
            "x": float(cx),
            "y": float(cy),
            "area": int(area),
            "width": int(w),
            "height": int(h),
            "confidence": float(confidence)
        })

    return objects


def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path):

    print("\n")
    print("=" * 70)
    print("ANALYZING:", video_path.name)
    print("=" * 70)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return None

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

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Total Frames: {total_frames}"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )


    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    tracks = {}

    next_track_id = 1

    frame_records = []

    all_speeds = []
    all_areas = []
    all_confidences = []

    object_counts = []

    previous_tracks = {}


    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        probability, binary = (
            get_prediction(frame)
        )

        objects = get_objects(
            binary,
            probability
        )

        object_counts.append(
            len(objects)
        )


        # Scale coordinates from 512
        # back to original resolution

        scale_x = width / INPUT_SIZE
        scale_y = height / INPUT_SIZE


        current_tracks = {}


        # ----------------------------------------------------
        # MATCH OBJECTS TO PREVIOUS FRAME
        # ----------------------------------------------------

        used_previous = set()

        for obj in objects:

            current_position = (
                obj["x"],
                obj["y"]
            )

            best_id = None
            best_distance = float("inf")

            for track_id, previous in (
                previous_tracks.items()
            ):

                if track_id in used_previous:
                    continue

                previous_position = (
                    previous["x"],
                    previous["y"]
                )

                d = distance(
                    current_position,
                    previous_position
                )

                if (
                    d < best_distance
                    and
                    d <= MAX_MATCH_DISTANCE
                ):

                    best_distance = d
                    best_id = track_id


            # New object
            if best_id is None:

                best_id = next_track_id

                next_track_id += 1

                displacement = 0.0
                speed = 0.0

            else:

                used_previous.add(
                    best_id
                )

                displacement = (
                    best_distance
                )

                time_interval = (
                    1.0 / fps
                    if fps > 0
                    else 0
                )

                if time_interval > 0:

                    speed = (
                        displacement
                        / time_interval
                    )

                else:

                    speed = 0.0


            # Convert position
            # back to original image scale

            original_x = (
                obj["x"] * scale_x
            )

            original_y = (
                obj["y"] * scale_y
            )


            current_tracks[best_id] = {

                "x": obj["x"],
                "y": obj["y"],

                "original_x":
                    original_x,

                "original_y":
                    original_y,

                "area":
                    obj["area"],

                "confidence":
                    obj["confidence"]
            }


            all_speeds.append(
                speed
            )

            all_areas.append(
                obj["area"]
            )

            all_confidences.append(
                obj["confidence"]
            )


            frame_records.append({

                "frame":
                    frame_number,

                "time_seconds":
                    (
                        (frame_number - 1)
                        / fps
                    ),

                "track_id":
                    best_id,

                "x":
                    original_x,

                "y":
                    original_y,

                "area":
                    obj["area"],

                "speed_pixels_per_second":
                    speed,

                "confidence":
                    obj["confidence"]
            })


        previous_tracks = (
            current_tracks
        )


        if frame_number % 50 == 0:

            print(
                f"Processed "
                f"{frame_number}/"
                f"{total_frames} frames"
            )


    cap.release()


    # ========================================================
    # STATISTICS
    # ========================================================

    average_objects = (
        np.mean(object_counts)
        if object_counts
        else 0
    )

    maximum_objects = (
        max(object_counts)
        if object_counts
        else 0
    )

    average_area = (
        np.mean(all_areas)
        if all_areas
        else 0
    )

    maximum_area = (
        max(all_areas)
        if all_areas
        else 0
    )

    minimum_area = (
        min(all_areas)
        if all_areas
        else 0
    )

    average_speed = (
        np.mean(all_speeds)
        if all_speeds
        else 0
    )

    maximum_speed = (
        max(all_speeds)
        if all_speeds
        else 0
    )

    average_confidence = (
        np.mean(all_confidences)
        if all_confidences
        else 0
    )


    # ========================================================
    # TRACK STATISTICS
    # ========================================================

    track_ids = set()

    for record in frame_records:

        track_ids.add(
            record["track_id"]
        )

    number_of_tracks = len(
        track_ids
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    base_name = (
        video_path.stem
    )

    csv_path = (
        OUTPUT_DIR
        /
        f"{base_name}_measurements.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "time_seconds",
                "track_id",
                "x",
                "y",
                "area",
                "speed_pixels_per_second",
                "confidence"
            ]
        )

        writer.writeheader()

        writer.writerows(
            frame_records
        )


    # ========================================================
    # GENERATE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        /
        f"{base_name}_analysis.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "CELL SEGMENTATION AND "
            "MOTION ANALYSIS REPORT\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Video: {video_path.name}\n\n"
        )

        f.write(
            "VIDEO INFORMATION\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"Resolution: "
            f"{width} x {height}\n"
        )

        f.write(
            f"FPS: {fps:.2f}\n"
        )

        f.write(
            f"Total Frames: "
            f"{total_frames}\n"
        )

        f.write(
            f"Duration: "
            f"{duration:.2f} seconds\n\n"
        )


        f.write(
            "SEGMENTATION RESULTS\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"Average Objects / Frame: "
            f"{average_objects:.2f}\n"
        )

        f.write(
            f"Maximum Objects / Frame: "
            f"{maximum_objects}\n"
        )

        f.write(
            f"Detected Tracks: "
            f"{number_of_tracks}\n"
        )

        f.write(
            f"Average Object Area: "
            f"{average_area:.2f} pixels\n"
        )

        f.write(
            f"Minimum Object Area: "
            f"{minimum_area:.2f} pixels\n"
        )

        f.write(
            f"Maximum Object Area: "
            f"{maximum_area:.2f} pixels\n"
        )

        f.write(
            f"Average Segmentation "
            f"Confidence: "
            f"{average_confidence:.4f}\n\n"
        )


        f.write(
            "MOVEMENT ANALYSIS\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"Average Speed: "
            f"{average_speed:.2f} "
            f"pixels/second\n"
        )

        f.write(
            f"Maximum Speed: "
            f"{maximum_speed:.2f} "
            f"pixels/second\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "NOTE:\n"
        )

        f.write(
            "Speed is currently reported in "
            "pixels/second because no microscope "
            "pixel-to-micrometer calibration has "
            "been provided.\n\n"
        )

        f.write(
            "The current model performs binary "
            "segmentation. Therefore connected "
            "regions are treated as detected "
            "objects and may contain multiple "
            "touching cells.\n"
        )


    # ========================================================
    # GRAPHS
    # ========================================================

    if frame_records:

        times = np.array([
            r["time_seconds"]
            for r in frame_records
        ])

        speeds = np.array([
            r["speed_pixels_per_second"]
            for r in frame_records
        ])

        areas = np.array([
            r["area"]
            for r in frame_records
        ])

        xs = np.array([
            r["x"]
            for r in frame_records
        ])

        ys = np.array([
            r["y"]
            for r in frame_records
        ])


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            times,
            speeds
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Speed (pixels/second)"
        )

        plt.title(
            f"Object Speed - {video_path.name}"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            /
            f"{base_name}_speed.png"
        )

        plt.close()


        # ----------------------------------------------------
        # AREA
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            times,
            areas
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Object Area (pixels)"
        )

        plt.title(
            f"Object Area - {video_path.name}"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            /
            f"{base_name}_area.png"
        )

        plt.close()


        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            times,
            xs,
            label="X position"
        )

        plt.plot(
            times,
            ys,
            label="Y position"
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Position (pixels)"
        )

        plt.title(
            f"Object Position - {video_path.name}"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            /
            f"{base_name}_position.png"
        )

        plt.close()


        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 7)
        )

        plt.plot(
            xs,
            ys
        )

        plt.scatter(
            xs[0],
            ys[0],
            label="Start"
        )

        plt.scatter(
            xs[-1],
            ys[-1],
            label="End"
        )

        plt.xlabel(
            "X position (pixels)"
        )

        plt.ylabel(
            "Y position (pixels)"
        )

        plt.title(
            f"Movement Trajectory - "
            f"{video_path.name}"
        )

        plt.legend()

        plt.grid(True)

        plt.gca().invert_yaxis()

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            /
            f"{base_name}_trajectory.png"
        )

        plt.close()


    # ========================================================
    # PRINT REPORT
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        f"ANALYSIS COMPLETE: "
        f"{video_path.name}"
    )

    print("=" * 60)

    print(
        f"Duration: "
        f"{duration:.2f} seconds"
    )

    print(
        f"Frames: "
        f"{total_frames}"
    )

    print(
        f"Average objects/frame: "
        f"{average_objects:.2f}"
    )

    print(
        f"Maximum objects/frame: "
        f"{maximum_objects}"
    )

    print(
        f"Average area: "
        f"{average_area:.2f} pixels"
    )

    print(
        f"Average speed: "
        f"{average_speed:.2f} pixels/sec"
    )

    print(
        f"Maximum speed: "
        f"{maximum_speed:.2f} pixels/sec"
    )

    print(
        f"Average confidence: "
        f"{average_confidence:.4f}"
    )

    print(
        f"Tracks detected: "
        f"{number_of_tracks}"
    )

    print("\nReport saved:")
    print(report_path)

    print("\nCSV saved:")
    print(csv_path)

    return {
        "video": video_path.name,
        "frames": total_frames,
        "fps": fps,
        "duration": duration,
        "average_objects": average_objects,
        "maximum_objects": maximum_objects,
        "average_area": average_area,
        "average_speed": average_speed,
        "maximum_speed": maximum_speed,
        "confidence": average_confidence,
        "tracks": number_of_tracks
    }


# ============================================================
# PROCESS ALL VIDEOS
# ============================================================

videos = sorted(
    VIDEO_DIR.glob("*.avi")
)

print("\nVideos found:", len(videos))

for video in videos:

    print("  ", video.name)


results = []

for video in videos:

    result = analyze_video(video)

    if result is not None:

        results.append(result)


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    /
    "overall_summary.csv"
)

with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video",
            "frames",
            "fps",
            "duration",
            "average_objects",
            "maximum_objects",
            "average_area",
            "average_speed",
            "maximum_speed",
            "confidence",
            "tracks"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print("\n")
print("=" * 70)
print("ALL VIDEO ANALYSIS COMPLETE")
print("=" * 70)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nEach video contains:"
)

print(
    "  - Analysis report (.txt)"
)

print(
    "  - Frame measurements (.csv)"
)

print(
    "  - Speed graph"
)

print(
    "  - Area graph"
)

print(
    "  - Position graph"
)

print(
    "  - Movement trajectory"
)

print(
    "\nOverall summary:"
)

print(
    summary_path
)
