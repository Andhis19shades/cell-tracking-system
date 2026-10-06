from pathlib import Path
import csv
import math

import cv2
import numpy as np
import tifffile

from skimage.morphology import skeletonize
from skan import Skeleton, summarize


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/Users/abhyudaysingh/microglia_dataset"
)

MASK_PATH = (
    BASE
    / "masks"
    / "microglia_example_labels_3frames.tif"
)

OUTPUT_DIR = (
    BASE
    / "morphology_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET CALIBRATION
# Public example metadata:
# 0.325 micrometers / pixel
# ============================================================

PIXEL_SIZE_UM = 0.325


# ============================================================
# HELPERS
# ============================================================

def circularity(
    area,
    perimeter
):
    if perimeter <= 0:
        return 0.0

    return (
        4.0 *
        math.pi *
        area /
        (perimeter ** 2)
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def find_soma_center(
    cell_mask
):
    """
    Approximate soma center using the maximum
    Euclidean distance inside the cell.

    The point with the largest distance from the
    boundary is used as a soma-center proxy.
    """

    distance = cv2.distanceTransform(
        cell_mask.astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    min_value, max_value, min_loc, max_loc = (
        cv2.minMaxLoc(distance)
    )

    if max_value <= 0:
        ys, xs = np.where(
            cell_mask
        )

        if len(xs) == 0:
            return (0.0, 0.0)

        return (
            float(xs.mean()),
            float(ys.mean())
        )

    return (
        float(max_loc[0]),
        float(max_loc[1])
    )


def skeleton_metrics(
    cell_mask
):
    """
    Analyze a single connected microglial mask.

    Returns:
        total skeleton length
        branch points
        endpoints
        soma center
        maximum process distance
    """

    binary = (
        cell_mask > 0
    )

    if binary.sum() < 10:
        return {
            "skeleton_length_px": 0.0,
            "branch_points": 0,
            "endpoints": 0,
            "soma_x_px": 0.0,
            "soma_y_px": 0.0,
            "max_process_distance_px": 0.0,
        }

    skeleton_image = skeletonize(
        binary
    )

    if skeleton_image.sum() < 2:
        soma_x, soma_y = find_soma_center(
            binary
        )

        return {
            "skeleton_length_px": 0.0,
            "branch_points": 0,
            "endpoints": 0,
            "soma_x_px": soma_x,
            "soma_y_px": soma_y,
            "max_process_distance_px": 0.0,
        }

    skel = Skeleton(
        skeleton_image
    )

    try:

        table = summarize(
            skel
        )

        if len(table) > 0:

            total_length = float(
                table[
                    "branch-distance"
                ].sum()
            )

        else:

            total_length = 0.0

    except Exception:

        total_length = 0.0

    # --------------------------------------------------------
    # Count pixels with 1 neighbour = endpoint
    # Count pixels with >=3 neighbours = branch point
    # --------------------------------------------------------

    skeleton_uint8 = (
        skeleton_image.astype(
            np.uint8
        )
    )

    neighbour_kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    neighbour_count = cv2.filter2D(
        skeleton_uint8,
        -1,
        neighbour_kernel
    )

    # The center pixel contributes 1 to the sum.
    neighbours = (
        neighbour_count -
        skeleton_uint8
    )

    endpoints = int(
        np.sum(
            skeleton_image &
            (neighbours == 1)
        )
    )

    branch_points = int(
        np.sum(
            skeleton_image &
            (neighbours >= 3)
        )
    )

    soma_x, soma_y = find_soma_center(
        binary
    )

    # --------------------------------------------------------
    # Maximum distance from soma to skeleton.
    # --------------------------------------------------------

    ys, xs = np.where(
        skeleton_image
    )

    if len(xs) > 0:

        distances = np.sqrt(
            (
                xs.astype(
                    np.float64
                ) -
                soma_x
            ) ** 2
            +
            (
                ys.astype(
                    np.float64
                ) -
                soma_y
            ) ** 2
        )

        max_distance = float(
            distances.max()
        )

    else:

        max_distance = 0.0

    return {
        "skeleton_length_px":
            total_length,

        "branch_points":
            branch_points,

        "endpoints":
            endpoints,

        "soma_x_px":
            soma_x,

        "soma_y_px":
            soma_y,

        "max_process_distance_px":
            max_distance,
    }


def sholl_analysis(
    cell_mask,
    soma_x,
    soma_y,
    pixel_size_um=PIXEL_SIZE_UM
):
    """
    Approximate Sholl analysis on a 2D skeleton.

    Concentric radii are tested around the soma center.
    """

    binary = (
        cell_mask > 0
    )

    skeleton_image = skeletonize(
        binary
    )

    ys, xs = np.where(
        skeleton_image
    )

    if len(xs) == 0:
        return {
            "sholl_max_intersections": 0,
            "sholl_critical_radius_um": 0.0,
        }

    dx = (
        xs.astype(
            np.float64
        ) -
        soma_x
    )

    dy = (
        ys.astype(
            np.float64
        ) -
        soma_y
    )

    distances_px = np.sqrt(
        dx * dx +
        dy * dy
    )

    max_distance_px = float(
        distances_px.max()
    )

    if max_distance_px < 2:
        return {
            "sholl_max_intersections": 0,
            "sholl_critical_radius_um": 0.0,
        }

    radii_px = np.arange(
        5.0,
        max_distance_px,
        5.0
    )

    intersections = []

    previous_mask = np.zeros(
        len(xs),
        dtype=bool
    )

    for radius in radii_px:

        tolerance = 1.75

        current = (
            np.abs(
                distances_px -
                radius
            ) <=
            tolerance
        )

        if not current.any():

            intersections.append(
                0
            )

            continue

        # Count groups of skeleton samples
        # around this annulus.
        indices = np.where(
            current
        )[0]

        points = np.column_stack(
            [
                xs[indices],
                ys[indices]
            ]
        )

        count = 0

        used = np.zeros(
            len(indices),
            dtype=bool
        )

        for i in range(
            len(indices)
        ):

            if used[i]:
                continue

            count += 1

            distances = np.sqrt(
                (
                    points[:, 0] -
                    points[i, 0]
                ) ** 2
                +
                (
                    points[:, 1] -
                    points[i, 1]
                ) ** 2
            )

            used |= (
                distances <= 2.5
            )

        intersections.append(
            count
        )

        previous_mask = current

    if not intersections:

        return {
            "sholl_max_intersections": 0,
            "sholl_critical_radius_um": 0.0,
        }

    maximum = max(
        intersections
    )

    max_index = intersections.index(
        maximum
    )

    critical_radius_um = (
        float(
            radii_px[max_index]
        ) *
        pixel_size_um
    )

    return {
        "sholl_max_intersections":
            int(maximum),

        "sholl_critical_radius_um":
            critical_radius_um,
    }


# ============================================================
# PROCESS ONE CELL
# ============================================================

def analyze_cell(
    full_instance_mask,
    instance_id,
    frame_index
):

    cell = (
        full_instance_mask ==
        instance_id
    ).astype(
        np.uint8
    )

    area_px = int(
        cell.sum()
    )

    if area_px < 20:
        return None

    # --------------------------------------------------------
    # CONTOUR
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        cell,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
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

    contour_area_px = float(
        cv2.contourArea(
            contour
        )
    )

    # --------------------------------------------------------
    # CIRCULARITY
    # --------------------------------------------------------

    cell_circularity = circularity(
        area_px,
        perimeter_px
    )

    # --------------------------------------------------------
    # CONVEX HULL
    # --------------------------------------------------------

    hull = cv2.convexHull(
        contour
    )

    hull_area_px = float(
        cv2.contourArea(
            hull
        )
    )

    if hull_area_px > 0:

        solidity = (
            area_px /
            hull_area_px
        )

    else:

        solidity = 0.0

    # --------------------------------------------------------
    # FIT ELLIPSE
    # --------------------------------------------------------

    eccentricity = 0.0

    aspect_ratio = 0.0

    if len(contour) >= 5:

        ellipse = cv2.fitEllipse(
            contour
        )

        major_axis = max(
            ellipse[1]
        )

        minor_axis = min(
            ellipse[1]
        )

        if major_axis > 0:

            aspect_ratio = (
                major_axis /
                max(
                    minor_axis,
                    1e-6
                )
            )

            ratio = (
                minor_axis /
                major_axis
            )

            ratio = max(
                0.0,
                min(
                    1.0,
                    ratio
                )
            )

            eccentricity = math.sqrt(
                max(
                    0.0,
                    1.0 -
                    ratio * ratio
                )
            )

    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    moments = cv2.moments(
        contour
    )

    if moments["m00"] != 0:

        centroid_x = (
            moments["m10"] /
            moments["m00"]
        )

        centroid_y = (
            moments["m01"] /
            moments["m00"]
        )

    else:

        centroid_x = 0.0
        centroid_y = 0.0

    # --------------------------------------------------------
    # SOMA PROXY
    # --------------------------------------------------------

    soma_x, soma_y = (
        find_soma_center(
            cell
        )
    )

    # --------------------------------------------------------
    # SKELETON
    # --------------------------------------------------------

    skeleton = skeleton_metrics(
        cell
    )

    skeleton_length_um = (
        skeleton[
            "skeleton_length_px"
        ]
        *
        PIXEL_SIZE_UM
    )

    max_process_distance_um = (
        skeleton[
            "max_process_distance_px"
        ]
        *
        PIXEL_SIZE_UM
    )

    # --------------------------------------------------------
    # SHOLL
    # --------------------------------------------------------

    sholl = sholl_analysis(
        cell,
        soma_x,
        soma_y
    )

    # --------------------------------------------------------
    # PHYSICAL UNITS
    # --------------------------------------------------------

    area_um2 = (
        area_px *
        PIXEL_SIZE_UM *
        PIXEL_SIZE_UM
    )

    perimeter_um = (
        perimeter_px *
        PIXEL_SIZE_UM
    )

    hull_area_um2 = (
        hull_area_px *
        PIXEL_SIZE_UM *
        PIXEL_SIZE_UM
    )

    # A useful morphology descriptor.
    process_to_area = (
        skeleton_length_um /
        max(
            area_um2,
            1e-6
        )
    )

    return {
        "frame": frame_index,

        "cell_id": instance_id,

        "area_px": area_px,

        "area_um2":
            area_um2,

        "perimeter_px":
            perimeter_px,

        "perimeter_um":
            perimeter_um,

        "circularity":
            cell_circularity,

        "solidity":
            solidity,

        "eccentricity":
            eccentricity,

        "aspect_ratio":
            aspect_ratio,

        "hull_area_um2":
            hull_area_um2,

        "centroid_x_px":
            centroid_x,

        "centroid_y_px":
            centroid_y,

        "soma_x_px":
            soma_x,

        "soma_y_px":
            soma_y,

        "skeleton_length_um":
            skeleton_length_um,

        "branch_points":
            skeleton[
                "branch_points"
            ],

        "endpoints":
            skeleton[
                "endpoints"
            ],

        "max_process_distance_um":
            max_process_distance_um,

        "process_area_ratio":
            process_to_area,

        "sholl_max_intersections":
            sholl[
                "sholl_max_intersections"
            ],

        "sholl_critical_radius_um":
            sholl[
                "sholl_critical_radius_um"
            ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 76)
    print("MICROGLIA MORPHOLOGY ENGINE")
    print("=" * 76)

    print(
        "Input:",
        MASK_PATH
    )

    print(
        "Pixel size:",
        PIXEL_SIZE_UM,
        "µm/pixel"
    )

    print("=" * 76)

    masks = tifffile.imread(
        MASK_PATH
    )

    print()
    print(
        "Mask stack:",
        masks.shape
    )

    rows = []

    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    for frame_index in range(
        masks.shape[0]
    ):

        frame = masks[
            frame_index
        ]

        instance_ids = np.unique(
            frame
        )

        instance_ids = (
            instance_ids[
                instance_ids != 0
            ]
        )

        print()
        print(
            f"Frame {frame_index}: "
            f"{len(instance_ids)} instances"
        )

        for instance_id in instance_ids:

            result = analyze_cell(
                frame,
                int(instance_id),
                frame_index
            )

            if result is None:
                continue

            rows.append(
                result
            )

    if not rows:

        raise RuntimeError(
            "No microglia were successfully analyzed."
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_DIR /
        "microglia_morphology.csv"
    )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_csv,
        "w",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("MICROGLIA MORPHOLOGY COMPLETE")
    print("=" * 76)

    print(
        "Cells analyzed:",
        len(rows)
    )

    areas = [
        row["area_um2"]
        for row in rows
    ]

    lengths = [
        row["skeleton_length_um"]
        for row in rows
    ]

    branches = [
        row["branch_points"]
        for row in rows
    ]

    endpoints = [
        row["endpoints"]
        for row in rows
    ]

    print(
        "Mean area:",
        f"{np.mean(areas):.2f} µm²"
    )

    print(
        "Mean process/skeleton length:",
        f"{np.mean(lengths):.2f} µm"
    )

    print(
        "Mean branch points:",
        f"{np.mean(branches):.2f}"
    )

    print(
        "Mean endpoints:",
        f"{np.mean(endpoints):.2f}"
    )

    print()
    print(
        "CSV:",
        output_csv
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
