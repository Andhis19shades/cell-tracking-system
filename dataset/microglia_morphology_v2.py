from pathlib import Path
import csv
import math

import cv2
import numpy as np
import tifffile
import pandas as pd

from skimage.morphology import skeletonize


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
    / "morphology_v2_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CALIBRATION
# ============================================================

PIXEL_SIZE_UM = 0.325


# ============================================================
# BASIC METRICS
# ============================================================

def circularity(
    area,
    perimeter
):

    if perimeter <= 0:
        return 0.0

    value = (
        4.0 *
        math.pi *
        area /
        (perimeter ** 2)
    )

    return float(
        min(
            max(
                value,
                0.0
            ),
            1.0
        )
    )


def connected_component_count(
    binary
):
    """
    Count connected regions in a binary image.
    """

    binary = (
        binary.astype(
            np.uint8
        )
        * 255
    )

    number, _, _, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    return max(
        0,
        number - 1
    )


# ============================================================
# SOMA CENTER
# ============================================================

def find_soma_center(
    cell_mask
):

    distance = cv2.distanceTransform(
        cell_mask.astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    _, max_value, _, max_location = (
        cv2.minMaxLoc(
            distance
        )
    )

    if max_value > 0:

        return (
            float(
                max_location[0]
            ),
            float(
                max_location[1]
            )
        )

    ys, xs = np.where(
        cell_mask
    )

    if len(xs) == 0:

        return (
            0.0,
            0.0
        )

    return (
        float(
            xs.mean()
        ),
        float(
            ys.mean()
        )
    )


# ============================================================
# SKELETON METRICS
# ============================================================

def skeleton_metrics(
    cell_mask
):

    binary = (
        cell_mask > 0
    )

    skeleton = skeletonize(
        binary
    )

    if skeleton.sum() < 2:

        soma_x, soma_y = (
            find_soma_center(
                binary
            )
        )

        return {
            "skeleton": skeleton,
            "skeleton_length_px": 0.0,
            "branch_points": 0,
            "endpoints": 0,
            "soma_x_px": soma_x,
            "soma_y_px": soma_y,
            "max_process_distance_px": 0.0,
        }

    # --------------------------------------------------------
    # Count neighbouring skeleton pixels.
    # --------------------------------------------------------

    skel_uint8 = (
        skeleton.astype(
            np.uint8
        )
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    neighbour_sum = cv2.filter2D(
        skel_uint8,
        -1,
        kernel
    )

    neighbours = (
        neighbour_sum -
        skel_uint8
    )

    # --------------------------------------------------------
    # Endpoint pixels
    # --------------------------------------------------------

    endpoint_pixels = (
        skeleton &
        (neighbours == 1)
    )

    # --------------------------------------------------------
    # Branch pixels
    # --------------------------------------------------------

    branch_pixels = (
        skeleton &
        (neighbours >= 3)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # A real branch junction can occupy several adjacent
    # skeleton pixels. Count connected clusters instead of
    # counting every branch pixel.
    # --------------------------------------------------------

    endpoints = connected_component_count(
        endpoint_pixels
    )

    branch_points = connected_component_count(
        branch_pixels
    )

    # --------------------------------------------------------
    # Approximate skeleton length.
    #
    # Horizontal / vertical edges = 1
    # Diagonal edges = sqrt(2)
    #
    # Each skeleton pixel is examined against previously
    # ordered neighbours to avoid double counting.
    # --------------------------------------------------------

    diagonal_length = math.sqrt(
        2.0
    )

    length_px = 0.0

    ys, xs = np.where(
        skeleton
    )

    skeleton_set = set(
        zip(
            xs.tolist(),
            ys.tolist()
        )
    )

    for x, y in skeleton_set:

        right = (
            x + 1,
            y
        )

        down = (
            x,
            y + 1
        )

        down_right = (
            x + 1,
            y + 1
        )

        down_left = (
            x - 1,
            y + 1
        )

        if right in skeleton_set:

            length_px += 1.0

        if down in skeleton_set:

            length_px += 1.0

        if down_right in skeleton_set:

            length_px += diagonal_length

        if down_left in skeleton_set:

            length_px += diagonal_length

    # --------------------------------------------------------
    # SOMA CENTER
    # --------------------------------------------------------

    soma_x, soma_y = (
        find_soma_center(
            binary
        )
    )

    # --------------------------------------------------------
    # MAXIMUM PROCESS REACH
    # --------------------------------------------------------

    if len(xs) > 0:

        distance = np.sqrt(
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
            distance.max()
        )

    else:

        max_distance = 0.0

    return {
        "skeleton": skeleton,
        "skeleton_length_px":
            length_px,

        "branch_points":
            int(branch_points),

        "endpoints":
            int(endpoints),

        "soma_x_px":
            soma_x,

        "soma_y_px":
            soma_y,

        "max_process_distance_px":
            max_distance,
    }


# ============================================================
# SHOLL ANALYSIS
# ============================================================

def sholl_analysis(
    skeleton,
    soma_x,
    soma_y
):

    ys, xs = np.where(
        skeleton
    )

    if len(xs) == 0:

        return {
            "max_intersections": 0,
            "critical_radius_px": 0.0,
            "profile": []
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

    radii = np.sqrt(
        dx * dx +
        dy * dy
    )

    max_radius = float(
        radii.max()
    )

    if max_radius < 3:

        return {
            "max_intersections": 0,
            "critical_radius_px": 0.0,
            "profile": []
        }

    # 5-pixel radial sampling.
    test_radii = np.arange(
        5.0,
        max_radius + 1.0,
        5.0
    )

    profile = []

    skeleton_points = np.column_stack(
        [
            xs,
            ys
        ]
    )

    for radius in test_radii:

        # Build a thin annulus.
        tolerance = 1.5

        annulus = (
            np.abs(
                radii -
                radius
            ) <=
            tolerance
        )

        points = skeleton_points[
            annulus
        ]

        if len(points) == 0:

            intersections = 0

        else:

            # Count spatial clusters of skeleton points
            # crossing the circle.
            used = np.zeros(
                len(points),
                dtype=bool
            )

            intersections = 0

            for i in range(
                len(points)
            ):

                if used[i]:
                    continue

                intersections += 1

                delta = (
                    points -
                    points[i]
                )

                distance = np.sqrt(
                    (
                        delta[:, 0] ** 2
                    ) +
                    (
                        delta[:, 1] ** 2
                    )
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
                        radius *
                        PIXEL_SIZE_UM
                    ),

                "intersections":
                    int(intersections)
            }
        )

    if not profile:

        return {
            "max_intersections": 0,
            "critical_radius_px": 0.0,
            "profile": []
        }

    max_intersections = max(
        row["intersections"]
        for row in profile
    )

    best_row = max(
        profile,
        key=lambda row:
        row["intersections"]
    )

    return {
        "max_intersections":
            int(max_intersections),

        "critical_radius_px":
            float(
                best_row[
                    "radius_px"
                ]
            ),

        "critical_radius_um":
            float(
                best_row[
                    "radius_um"
                ]
            ),

        "profile":
            profile
    }


# ============================================================
# ANALYZE CELL
# ============================================================

def analyze_cell(
    frame,
    instance_id,
    frame_index
):

    cell = (
        frame ==
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

    # --------------------------------------------------------
    # HULL
    # --------------------------------------------------------

    hull = cv2.convexHull(
        contour
    )

    hull_area_px = float(
        cv2.contourArea(
            hull
        )
    )

    solidity = (
        float(area_px) /
        hull_area_px
        if hull_area_px > 0
        else 0.0
    )

    # --------------------------------------------------------
    # ELLIPSE
    # --------------------------------------------------------

    eccentricity = 0.0
    aspect_ratio = 0.0

    if len(contour) >= 5:

        ellipse = cv2.fitEllipse(
            contour
        )

        major = max(
            ellipse[1]
        )

        minor = min(
            ellipse[1]
        )

        if major > 0:

            aspect_ratio = (
                major /
                max(
                    minor,
                    1e-6
                )
            )

            ratio = (
                minor /
                major
            )

            ratio = min(
                max(
                    ratio,
                    0.0
                ),
                1.0
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
    # SOMA + SKELETON
    # --------------------------------------------------------

    skeleton_data = skeleton_metrics(
        cell
    )

    skeleton = skeleton_data[
        "skeleton"
    ]

    soma_x = skeleton_data[
        "soma_x_px"
    ]

    soma_y = skeleton_data[
        "soma_y_px"
    ]

    # --------------------------------------------------------
    # SHOLL
    # --------------------------------------------------------

    sholl = sholl_analysis(
        skeleton,
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

    skeleton_length_um = (
        skeleton_data[
            "skeleton_length_px"
        ] *
        PIXEL_SIZE_UM
    )

    max_process_distance_um = (
        skeleton_data[
            "max_process_distance_px"
        ] *
        PIXEL_SIZE_UM
    )

    process_length_to_area = (
        skeleton_length_um /
        max(
            area_um2,
            1e-6
        )
    )

    # --------------------------------------------------------
    # RETURN MAIN METRICS
    # --------------------------------------------------------

    result = {
        "frame":
            int(frame_index),

        "cell_id":
            int(instance_id),

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
            float(solidity),

        "eccentricity":
            float(eccentricity),

        "aspect_ratio":
            float(aspect_ratio),

        "convex_hull_area_um2":
            float(hull_area_um2),

        "centroid_x_px":
            float(centroid_x),

        "centroid_y_px":
            float(centroid_y),

        "soma_x_px":
            float(soma_x),

        "soma_y_px":
            float(soma_y),

        "skeleton_length_um":
            float(
                skeleton_length_um
            ),

        "branch_points":
            int(
                skeleton_data[
                    "branch_points"
                ]
            ),

        "endpoints":
            int(
                skeleton_data[
                    "endpoints"
                ]
            ),

        "max_process_distance_um":
            float(
                max_process_distance_um
            ),

        "process_length_to_area":
            float(
                process_length_to_area
            ),

        "sholl_max_intersections":
            int(
                sholl[
                    "max_intersections"
                ]
            ),

        "sholl_critical_radius_um":
            float(
                sholl.get(
                    "critical_radius_um",
                    0.0
                )
            )
    }

    return (
        result,
        sholl["profile"],
        skeleton
    )


# ============================================================
# SAVE SKELETON VISUAL
# ============================================================

def save_skeleton_visual(
    frame,
    instance_id,
    skeleton,
    soma_x,
    soma_y,
    frame_index
):

    cell = (
        frame ==
        instance_id
    )

    visual = np.zeros(
        (
            frame.shape[0],
            frame.shape[1],
            3
        ),
        dtype=np.uint8
    )

    # Cell silhouette.
    visual[
        cell
    ] = (
        55,
        55,
        55
    )

    # Skeleton.
    visual[
        skeleton
    ] = (
        60,
        220,
        120
    )

    # Soma point.
    cv2.circle(
        visual,
        (
            int(
                round(
                    soma_x
                )
            ),
            int(
                round(
                    soma_y
                )
            )
        ),
        5,
        (
            80,
            120,
            255
        ),
        -1
    )

    output_path = (
        OUTPUT_DIR /
        f"frame_{frame_index:02d}_"
        f"cell_{instance_id:02d}_"
        "skeleton.png"
    )

    cv2.imwrite(
        str(output_path),
        visual
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 76)
    print("MICROGLIA MORPHOLOGY ENGINE V2")
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

    if not MASK_PATH.exists():

        raise FileNotFoundError(
            MASK_PATH
        )

    masks = tifffile.imread(
        MASK_PATH
    )

    print(
        "Mask stack:",
        masks.shape
    )

    cell_rows = []
    sholl_rows = []

    visual_counter = 0

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
            f"{len(instance_ids)} labelled cells"
        )

        for instance_id in instance_ids:

            result_data = analyze_cell(
                frame,
                int(instance_id),
                frame_index
            )

            if result_data is None:
                continue

            result, profile, skeleton = (
                result_data
            )

            cell_rows.append(
                result
            )

            for sholl in profile:

                sholl_rows.append(
                    {
                        "frame":
                            frame_index,

                        "cell_id":
                            int(instance_id),

                        "radius_px":
                            sholl[
                                "radius_px"
                            ],

                        "radius_um":
                            sholl[
                                "radius_um"
                            ],

                        "intersections":
                            sholl[
                                "intersections"
                            ]
                    }
                )

            # Save a few representative skeletons.
            if visual_counter < 10:

                save_skeleton_visual(
                    frame,
                    int(instance_id),
                    skeleton,
                    result[
                        "soma_x_px"
                    ],
                    result[
                        "soma_y_px"
                    ],
                    frame_index
                )

                visual_counter += 1

    if not cell_rows:

        raise RuntimeError(
            "No cells were successfully analyzed."
        )

    # --------------------------------------------------------
    # DATAFRAMES
    # --------------------------------------------------------

    cells_df = pd.DataFrame(
        cell_rows
    )

    sholl_df = pd.DataFrame(
        sholl_rows
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "Metric":
                    "Cells analyzed",

                "Value":
                    len(cells_df),

                "Unit":
                    "cell observations"
            },

            {
                "Metric":
                    "Mean cell area",

                "Value":
                    cells_df[
                        "area_um2"
                    ].mean(),

                "Unit":
                    "µm²"
            },

            {
                "Metric":
                    "Mean perimeter",

                "Value":
                    cells_df[
                        "perimeter_um"
                    ].mean(),

                "Unit":
                    "µm"
            },

            {
                "Metric":
                    "Mean circularity",

                "Value":
                    cells_df[
                        "circularity"
                    ].mean(),

                "Unit":
                    "dimensionless"
            },

            {
                "Metric":
                    "Mean solidity",

                "Value":
                    cells_df[
                        "solidity"
                    ].mean(),

                "Unit":
                    "dimensionless"
            },

            {
                "Metric":
                    "Mean skeleton/process length",

                "Value":
                    cells_df[
                        "skeleton_length_um"
                    ].mean(),

                "Unit":
                    "µm"
            },

            {
                "Metric":
                    "Mean branch points",

                "Value":
                    cells_df[
                        "branch_points"
                    ].mean(),

                "Unit":
                    "junctions/cell"
            },

            {
                "Metric":
                    "Mean endpoints",

                "Value":
                    cells_df[
                        "endpoints"
                    ].mean(),

                "Unit":
                    "endpoints/cell"
            },

            {
                "Metric":
                    "Mean maximum process reach",

                "Value":
                    cells_df[
                        "max_process_distance_um"
                    ].mean(),

                "Unit":
                    "µm"
            },

            {
                "Metric":
                    "Mean Sholl maximum",

                "Value":
                    cells_df[
                        "sholl_max_intersections"
                    ].mean(),

                "Unit":
                    "intersections"
            }
        ]
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    cells_csv = (
        OUTPUT_DIR /
        "microglia_cells_v2.csv"
    )

    sholl_csv = (
        OUTPUT_DIR /
        "microglia_sholl_v2.csv"
    )

    summary_csv = (
        OUTPUT_DIR /
        "microglia_summary_v2.csv"
    )

    cells_df.to_csv(
        cells_csv,
        index=False
    )

    sholl_df.to_csv(
        sholl_csv,
        index=False
    )

    summary.to_csv(
        summary_csv,
        index=False
    )

    # --------------------------------------------------------
    # SAVE EXCEL
    # --------------------------------------------------------

    excel_path = (
        OUTPUT_DIR /
        "microglia_morphology_v2.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        cells_df.to_excel(
            writer,
            sheet_name="Cell Metrics",
            index=False
        )

        sholl_df.to_excel(
            writer,
            sheet_name="Sholl Profile",
            index=False
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("MICROGLIA MORPHOLOGY V2 COMPLETE")
    print("=" * 76)

    print(
        "Cell observations:",
        len(cells_df)
    )

    print(
        "Mean area:",
        f"{cells_df['area_um2'].mean():.2f} µm²"
    )

    print(
        "Mean skeleton/process length:",
        f"{cells_df['skeleton_length_um'].mean():.2f} µm"
    )

    print(
        "Mean branch points:",
        f"{cells_df['branch_points'].mean():.2f}"
    )

    print(
        "Mean endpoints:",
        f"{cells_df['endpoints'].mean():.2f}"
    )

    print(
        "Mean Sholl maximum:",
        f"{cells_df['sholl_max_intersections'].mean():.2f}"
    )

    print()
    print(
        "Excel:",
        excel_path
    )

    print(
        "Cell CSV:",
        cells_csv
    )

    print(
        "Sholl CSV:",
        sholl_csv
    )

    print(
        "Summary CSV:",
        summary_csv
    )

    print()
    print(
        "Representative skeletons:",
        OUTPUT_DIR
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
