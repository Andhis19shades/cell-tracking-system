import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path(
    "/Users/abhyudaysingh/Downloads/quantitative_results"
)

FRAME_FILE = RESULTS_DIR / "video_0_frame_analysis.csv"
TRACK_FILE = RESULTS_DIR / "video_0_cell_tracks.csv"
SUMMARY_FILE = RESULTS_DIR / "video_0_cell_summary.csv"

OUTPUT_DIR = RESULTS_DIR / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading quantitative data...")

frame_df = pd.read_csv(FRAME_FILE)
track_df = pd.read_csv(TRACK_FILE)
summary_df = pd.read_csv(SUMMARY_FILE)

print("Frame records:", len(frame_df))
print("Track records:", len(track_df))
print("Cell tracks:", len(summary_df))


# ============================================================
# 1. CELL COUNT VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["cell_count"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Cells detected")
plt.title("Cell Count vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_cell_count_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 2. MOVING CELLS VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["moving_cells"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Moving cells")
plt.title("Moving Cells vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_moving_cells_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 3. MEAN SPEED VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["mean_speed_px_s"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Mean speed (pixels/second)")
plt.title("Mean Cell Speed vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_mean_speed_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 4. MEDIAN SPEED VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["median_speed_px_s"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Median speed (pixels/second)")
plt.title("Median Cell Speed vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_median_speed_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 5. MAXIMUM SPEED VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["max_speed_px_s"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Maximum speed (pixels/second)")
plt.title("Maximum Cell Speed vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_max_speed_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 6. MEAN DISPLACEMENT VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["mean_displacement_px"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Mean displacement (pixels)")
plt.title("Mean Cell Displacement vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_mean_displacement_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 7. TOTAL DISPLACEMENT VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["total_displacement_px"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Total displacement (pixels)")
plt.title("Total Cell Movement vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_total_displacement_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 8. MEAN CELL AREA VS TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frame_df["time_seconds"],
    frame_df["mean_cell_area_px"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Mean cell area (pixels²)")
plt.title("Mean Cell Area vs Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_mean_cell_area_vs_time.png",
    dpi=200
)

plt.close()


# ============================================================
# 9. SPEED DISTRIBUTION
# ============================================================

speed_data = track_df[
    track_df["speed_px_s"] > 0
]["speed_px_s"]

plt.figure(figsize=(12, 6))

plt.hist(
    speed_data,
    bins=50
)

plt.xlabel("Speed (pixels/second)")
plt.ylabel("Frequency")
plt.title("Distribution of Cell Speeds")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_speed_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 10. TRACK DURATION DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 6))

plt.hist(
    summary_df["duration_seconds"],
    bins=40
)

plt.xlabel("Track duration (seconds)")
plt.ylabel("Number of tracks")
plt.title("Cell Track Duration Distribution")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "10_track_duration_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 11. TOTAL DISTANCE PER CELL
# ============================================================

top_distance = summary_df.sort_values(
    "total_distance_px",
    ascending=False
).head(30)

plt.figure(figsize=(14, 7))

plt.bar(
    top_distance["track_id"].astype(str),
    top_distance["total_distance_px"]
)

plt.xlabel("Track ID")
plt.ylabel("Total distance (pixels)")
plt.title("Top 30 Cells by Total Distance Travelled")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "11_top_cells_total_distance.png",
    dpi=200
)

plt.close()


# ============================================================
# 12. AVERAGE SPEED PER CELL
# ============================================================

top_speed = summary_df.sort_values(
    "average_speed_px_s",
    ascending=False
).head(30)

plt.figure(figsize=(14, 7))

plt.bar(
    top_speed["track_id"].astype(str),
    top_speed["average_speed_px_s"]
)

plt.xlabel("Track ID")
plt.ylabel("Average speed (pixels/second)")
plt.title("Top 30 Cells by Average Speed")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "12_top_cells_average_speed.png",
    dpi=200
)

plt.close()


# ============================================================
# 13. TRAJECTORY MAP
# ============================================================

plt.figure(figsize=(10, 10))

for track_id, group in track_df.groupby("track_id"):

    if len(group) < 5:
        continue

    plt.plot(
        group["x"],
        group["y"],
        linewidth=0.8,
        alpha=0.6
    )


plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")
plt.title("Cell Trajectories")

plt.gca().invert_yaxis()

plt.axis("equal")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "13_cell_trajectories.png",
    dpi=200
)

plt.close()


# ============================================================
# 14. TRAJECTORY MAP - LONG TRACKS ONLY
# ============================================================

plt.figure(figsize=(10, 10))

long_tracks = summary_df[
    summary_df["frames_tracked"] >= 20
]["track_id"]

for track_id in long_tracks:

    group = track_df[
        track_df["track_id"] == track_id
    ]

    plt.plot(
        group["x"],
        group["y"],
        linewidth=1.0,
        alpha=0.7
    )


plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")
plt.title("Cell Trajectories — Tracks ≥ 20 Frames")

plt.gca().invert_yaxis()

plt.axis("equal")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "14_long_cell_trajectories.png",
    dpi=200
)

plt.close()


# ============================================================
# 15. SPEED VS CELL AREA
# ============================================================

valid_summary = summary_df[
    (summary_df["average_speed_px_s"] > 0) &
    (summary_df["average_area_px"] > 0)
]

plt.figure(figsize=(10, 7))

plt.scatter(
    valid_summary["average_area_px"],
    valid_summary["average_speed_px_s"],
    alpha=0.5
)

plt.xlabel("Average cell area (pixels²)")
plt.ylabel("Average speed (pixels/second)")
plt.title("Cell Size vs Average Speed")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "15_cell_area_vs_speed.png",
    dpi=200
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("========================================")
print("GRAPH GENERATION COMPLETE")
print("========================================")
print()
print("Graphs saved to:")
print(OUTPUT_DIR)
print()
