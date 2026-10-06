from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import tifffile

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
    / "tracking_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SPATIAL CALIBRATION
# ============================================================

PIXEL_SIZE_UM = 0.325


# ============================================================
# TEMPORAL CALIBRATION
#
# We intentionally do NOT invent a frame interval.
#
# Enter the actual interval between frames here when known.
# Example:
#
# FRAME_INTERVAL_SECONDS = 10.0
#
# Set to None if unknown.
# ============================================================

FRAME_INTERVAL_SECONDS = None


# ============================================================
# TRACKING PARAMETERS
# ============================================================

MAX_TIP_MATCH_DISTANCE_PX = 40.0

MIN_CELL_AREA_PX = 20

MIN_TIP_DISTANCE_FROM_SOMA_PX = 8.0


# ============================================================
# HELPERS
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
            float(max_location[0]),
            float(max_location[1]),
            float(max_value)
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
        float(xs.mean()),
        float(ys.mean()),
        0.0
    )


def get_skeleton_and_endpoints(
    cell_mask
):

    binary = (
        cell_mask > 0
    )

    skeleton = skeletonize(
        binary
    )

    if skeleton.sum() == 0:

        return (
            skeleton,
            []
        )

    skel = (
        skeleton.astype(
            np.uint8
        )
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    neighbour_sum = cv2.filter2D(
        skel,
        -1,
        kernel
    )

    neighbours = (
        neighbour_sum -
        skel
    )

    endpoint_pixels = (
        skeleton &
        (neighbours == 1)
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

    for label_id in range(
        1,
        number
    ):

        x = float(
            centroids[
                label_id,
                0
            ]
        )

        y = float(
            centroids[
                label_id,
                1
            ]
        )

        endpoints.append(
            (
                x,
                y
            )
        )

    return (
        skeleton,
        endpoints
    )


def extract_cell_geometry(
    frame,
    cell_id
):

    cell = (
        frame ==
        cell_id
    ).astype(
        np.uint8
    )

    area_px = int(
        cell.sum()
    )

    if area_px < MIN_CELL_AREA_PX:

        return None

    moments = cv2.moments(
        cell
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

    (
        soma_x,
        soma_y,
        soma_radius_px
    ) = find_soma_center(
        cell
    )

    (
        skeleton,
        raw_endpoints
    ) = get_skeleton_and_endpoints(
        cell
    )

    # --------------------------------------------------------
    # Remove endpoints too close to the soma.
    #
    # This prevents the soma-side skeleton endpoint from being
    # interpreted as a process tip in small/simple cells.
    # --------------------------------------------------------

    tips = []

    for x, y in raw_endpoints:

        distance = math.sqrt(
            (
                x -
                soma_x
            ) ** 2
            +
            (
                y -
                soma_y
            ) ** 2
        )

        if distance >= (
            max(
                MIN_TIP_DISTANCE_FROM_SOMA_PX,
                soma_radius_px * 1.25
            )
        ):

            tips.append(
                {
                    "x": x,
                    "y": y,
                    "distance_from_soma_px":
                        distance
                }
            )

    return {
        "cell_mask": cell,
        "skeleton": skeleton,
        "area_px": area_px,
        "centroid_x_px": float(
            centroid_x
        ),
        "centroid_y_px": float(
            centroid_y
        ),
        "soma_x_px": float(
            soma_x
        ),
        "soma_y_px": float(
            soma_y
        ),
        "soma_radius_px": float(
            soma_radius_px
        ),
        "tips": tips
    }


def euclidean_distance(
    x1,
    y1,
    x2,
    y2
):

    return math.sqrt(
        (
            x2 -
            x1
        ) ** 2
        +
        (
            y2 -
            y1
        ) ** 2
    )


def match_tips(
    previous_tips,
    current_tips
):

    """
    Greedy nearest-neighbour matching.

    Returns:
        list of matches

    Each match contains:
        previous tip
        current tip
        displacement
    """

    if not previous_tips:

        return []

    if not current_tips:

        return []

    candidates = []

    for i, previous in enumerate(
        previous_tips
    ):

        for j, current in enumerate(
            current_tips
        ):

            distance = euclidean_distance(
                previous["x"],
                previous["y"],
                current["x"],
                current["y"]
            )

            if distance <= (
                MAX_TIP_MATCH_DISTANCE_PX
            ):

                candidates.append(
                    (
                        distance,
                        i,
                        j
                    )
                )

    candidates.sort(
        key=lambda item:
        item[0]
    )

    used_previous = set()
    used_current = set()

    matches = []

    for (
        distance,
        previous_index,
        current_index
    ) in candidates:

        if previous_index in used_previous:
            continue

        if current_index in used_current:
            continue

        used_previous.add(
            previous_index
        )

        used_current.add(
            current_index
        )

        matches.append(
            {
                "previous":
                    previous_tips[
                        previous_index
                    ],

                "current":
                    current_tips[
                        current_index
                    ],

                "displacement_px":
                    float(
                        distance
                    )
            }
        )

    return matches


def tip_dynamics(
    previous,
    current,
    soma_x,
    soma_y
):

    previous_radial = euclidean_distance(
        previous["x"],
        previous["y"],
        soma_x,
        soma_y
    )

    current_radial = euclidean_distance(
        current["x"],
        current["y"],
        soma_x,
        soma_y
    )

    radial_change = (
        current_radial -
        previous_radial
    )

    if radial_change > 0.5:

        state = "extension"

    elif radial_change < -0.5:

        state = "retraction"

    else:

        state = "stable"

    return (
        radial_change,
        state
    )


def save_tracking_visual(
    image_shape,
    previous_geometry,
    current_geometry,
    frame_index,
    cell_id,
    matches
):

    visual = np.zeros(
        (
            image_shape[0],
            image_shape[1],
            3
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Previous skeleton
    # --------------------------------------------------------

    previous_skeleton = (
        previous_geometry[
            "skeleton"
        ]
    )

    visual[
        previous_skeleton
    ] = (
        90,
        90,
        200
    )

    # --------------------------------------------------------
    # Current skeleton
    # --------------------------------------------------------

    current_skeleton = (
        current_geometry[
            "skeleton"
        ]
    )

    visual[
        current_skeleton
    ] = (
        70,
        210,
        90
    )

    # --------------------------------------------------------
    # SOMA
    # --------------------------------------------------------

    cv2.circle(
        visual,
        (
            int(
                round(
                    current_geometry[
                        "soma_x_px"
                    ]
                )
            ),
            int(
                round(
                    current_geometry[
                        "soma_y_px"
                    ]
                )
            )
        ),
        5,
        (
            80,
            180,
            250
        ),
        -1
    )

    # --------------------------------------------------------
    # TIP MATCH LINES
    # --------------------------------------------------------

    for match in matches:

        previous_tip = match[
            "previous"
        ]

        current_tip = match[
            "current"
        ]

        cv2.line(
            visual,
            (
                int(
                    round(
                        previous_tip[
                            "x"
                        ]
                    )
                ),
                int(
                    round(
                        previous_tip[
                            "y"
                        ]
                    )
                )
            ),
            (
                int(
                    round(
                        current_tip[
                            "x"
                        ]
                    )
                ),
                int(
                    round(
                        current_tip[
                            "y"
                        ]
                    )
                )
            ),
            (
                220,
                180,
                70
            ),
            1
        )

        cv2.circle(
            visual,
            (
                int(
                    round(
                        current_tip[
                            "x"
                        ]
                    )
                ),
                int(
                    round(
                        current_tip[
                            "y"
                        ]
                    )
                )
            ),
            3,
            (
                220,
                180,
                70
            ),
            -1
        )

    output_path = (
        OUTPUT_DIR
        / (
            f"frame_{frame_index:02d}_"
            f"cell_{cell_id:03d}_"
            "tip_motion.png"
        )
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
    print("=" * 78)
    print("MICROGLIA TEMPORAL MOVEMENT ENGINE")
    print("=" * 78)

    print(
        "Input:",
        MASK_PATH
    )

    print(
        "Pixel size:",
        PIXEL_SIZE_UM,
        "µm/pixel"
    )

    if FRAME_INTERVAL_SECONDS is None:

        print(
            "Frame interval: UNKNOWN"
        )

    else:

        print(
            "Frame interval:",
            FRAME_INTERVAL_SECONDS,
            "seconds"
        )

    print("=" * 78)

    if not MASK_PATH.exists():

        raise FileNotFoundError(
            MASK_PATH
        )

    masks = tifffile.imread(
        MASK_PATH
    )

    print(
        "Frames:",
        masks.shape[0]
    )

    print(
        "Resolution:",
        masks.shape[1],
        "x",
        masks.shape[2]
    )

    # --------------------------------------------------------
    # BUILD GEOMETRY
    # --------------------------------------------------------

    frame_geometry = {}

    for frame_index in range(
        masks.shape[0]
    ):

        frame = masks[
            frame_index
        ]

        ids = np.unique(
            frame
        )

        ids = ids[
            ids != 0
        ]

        frame_geometry[
            frame_index
        ] = {}

        print()
        print(
            f"Frame {frame_index}: "
            f"{len(ids)} cells"
        )

        for cell_id in ids:

            geometry = extract_cell_geometry(
                frame,
                int(cell_id)
            )

            if geometry is None:
                continue

            frame_geometry[
                frame_index
            ][
                int(cell_id)
            ] = geometry

    # --------------------------------------------------------
    # CELL-LEVEL TRACKING
    # --------------------------------------------------------

    cell_rows = []

    # --------------------------------------------------------
    # PROCESS-TIP TRACKING
    # --------------------------------------------------------

    tip_rows = []

    visual_count = 0

    frame_indices = sorted(
        frame_geometry.keys()
    )

    for frame_index in frame_indices:

        current_cells = frame_geometry[
            frame_index
        ]

        # ----------------------------------------------------
        # Basic state for each cell
        # ----------------------------------------------------

        for cell_id, geometry in (
            current_cells.items()
        ):

            cell_rows.append(
                {
                    "frame":
                        frame_index,

                    "cell_id":
                        cell_id,

                    "centroid_x_px":
                        geometry[
                            "centroid_x_px"
                        ],

                    "centroid_y_px":
                        geometry[
                            "centroid_y_px"
                        ],

                    "centroid_x_um":
                        geometry[
                            "centroid_x_px"
                        ] *
                        PIXEL_SIZE_UM,

                    "centroid_y_um":
                        geometry[
                            "centroid_y_px"
                        ] *
                        PIXEL_SIZE_UM,

                    "soma_x_px":
                        geometry[
                            "soma_x_px"
                        ],

                    "soma_y_px":
                        geometry[
                            "soma_y_px"
                        ],

                    "soma_x_um":
                        geometry[
                            "soma_x_px"
                        ] *
                        PIXEL_SIZE_UM,

                    "soma_y_um":
                        geometry[
                            "soma_y_px"
                        ] *
                        PIXEL_SIZE_UM,

                    "cell_area_um2":
                        geometry[
                            "area_px"
                        ] *
                        PIXEL_SIZE_UM *
                        PIXEL_SIZE_UM,

                    "soma_radius_um":
                        geometry[
                            "soma_radius_px"
                        ] *
                        PIXEL_SIZE_UM,

                    "process_tips":
                        len(
                            geometry[
                                "tips"
                            ]
                        )
                }
            )

    # --------------------------------------------------------
    # FRAME-TO-FRAME MOVEMENT
    # --------------------------------------------------------

    for frame_index in frame_indices:

        next_frame = (
            frame_index + 1
        )

        if next_frame not in frame_geometry:

            continue

        current_cells = (
            frame_geometry[
                frame_index
            ]
        )

        next_cells = (
            frame_geometry[
                next_frame
            ]
        )

        common_ids = sorted(
            set(
                current_cells.keys()
            )
            &
            set(
                next_cells.keys()
            )
        )

        for cell_id in common_ids:

            previous = current_cells[
                cell_id
            ]

            current = next_cells[
                cell_id
            ]

            # ------------------------------------------------
            # CENTROID MOVEMENT
            # ------------------------------------------------

            centroid_displacement_px = (
                euclidean_distance(
                    previous[
                        "centroid_x_px"
                    ],
                    previous[
                        "centroid_y_px"
                    ],
                    current[
                        "centroid_x_px"
                    ],
                    current[
                        "centroid_y_px"
                    ]
                )
            )

            centroid_displacement_um = (
                centroid_displacement_px
                *
                PIXEL_SIZE_UM
            )

            # ------------------------------------------------
            # SOMA MOVEMENT
            # ------------------------------------------------

            soma_displacement_px = (
                euclidean_distance(
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
            )

            soma_displacement_um = (
                soma_displacement_px
                *
                PIXEL_SIZE_UM
            )

            # ------------------------------------------------
            # VELOCITIES
            # ------------------------------------------------

            if FRAME_INTERVAL_SECONDS is not None:

                dt_min = (
                    FRAME_INTERVAL_SECONDS
                    /
                    60.0
                )

                centroid_velocity = (
                    centroid_displacement_um
                    /
                    dt_min
                )

                soma_velocity = (
                    soma_displacement_um
                    /
                    dt_min
                )

            else:

                dt_min = None

                centroid_velocity = None

                soma_velocity = None

            # ------------------------------------------------
            # TIP MATCHING
            # ------------------------------------------------

            matches = match_tips(
                previous[
                    "tips"
                ],
                current[
                    "tips"
                ]
            )

            # ------------------------------------------------
            # TIP ROWS
            # ------------------------------------------------

            for tip_index, match in enumerate(
                matches,
                start=1
            ):

                previous_tip = match[
                    "previous"
                ]

                current_tip = match[
                    "current"
                ]

                displacement_px = (
                    match[
                        "displacement_px"
                    ]
                )

                displacement_um = (
                    displacement_px
                    *
                    PIXEL_SIZE_UM
                )

                (
                    radial_change_px,
                    state
                ) = tip_dynamics(
                    previous_tip,
                    current_tip,
                    current[
                        "soma_x_px"
                    ],
                    current[
                        "soma_y_px"
                    ]
                )

                radial_change_um = (
                    radial_change_px
                    *
                    PIXEL_SIZE_UM
                )

                if FRAME_INTERVAL_SECONDS is not None:

                    tip_velocity = (
                        displacement_um
                        /
                        dt_min
                    )

                else:

                    tip_velocity = None

                tip_rows.append(
                    {
                        "frame_from":
                            frame_index,

                        "frame_to":
                            next_frame,

                        "cell_id":
                            cell_id,

                        "tip_pair":
                            tip_index,

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

                        "displacement_px":
                            displacement_px,

                        "displacement_um":
                            displacement_um,

                        "radial_change_px":
                            radial_change_px,

                        "radial_change_um":
                            radial_change_um,

                        "state":
                            state,

                        "speed_um_per_min":
                            tip_velocity
                    }
                )

            # ------------------------------------------------
            # SAVE VISUAL EXAMPLES
            # ------------------------------------------------

            if visual_count < 10:

                save_tracking_visual(
                    masks.shape[1:],
                    previous,
                    current,
                    frame_index,
                    cell_id,
                    matches
                )

                visual_count += 1

            # ------------------------------------------------
            # CELL MOTION ROW
            # ------------------------------------------------

            cell_rows.append(
                {
                    "frame_from":
                        frame_index,

                    "frame_to":
                        next_frame,

                    "cell_id":
                        cell_id,

                    "centroid_displacement_um":
                        centroid_displacement_um,

                    "centroid_velocity_um_per_min":
                        centroid_velocity,

                    "soma_displacement_um":
                        soma_displacement_um,

                    "soma_velocity_um_per_min":
                        soma_velocity,

                    "previous_process_tips":
                        len(
                            previous[
                                "tips"
                            ]
                        ),

                    "current_process_tips":
                        len(
                            current[
                                "tips"
                            ]
                        ),

                    "matched_process_tips":
                        len(
                            matches
                        )
                }
            )

    # --------------------------------------------------------
    # DATAFRAMES
    # --------------------------------------------------------

    cell_df = pd.DataFrame(
        cell_rows
    )

    tip_df = pd.DataFrame(
        tip_rows
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    cell_csv = (
        OUTPUT_DIR /
        "microglia_cell_motion.csv"
    )

    tip_csv = (
        OUTPUT_DIR /
        "microglia_process_tip_motion.csv"
    )

    cell_df.to_csv(
        cell_csv,
        index=False
    )

    tip_df.to_csv(
        tip_csv,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    extension_count = 0
    retraction_count = 0
    stable_count = 0

    if not tip_df.empty:

        extension_count = int(
            (
                tip_df["state"]
                == "extension"
            ).sum()
        )

        retraction_count = int(
            (
                tip_df["state"]
                == "retraction"
            ).sum()
        )

        stable_count = int(
            (
                tip_df["state"]
                == "stable"
            ).sum()
        )

    summary_rows = []

    summary_rows.append(
        {
            "Metric":
                "Cell frame observations",

            "Value":
                len(cell_df)
        }
    )

    if not cell_df.empty:

        movement_df = cell_df[
            cell_df.get(
                "centroid_displacement_um",
                pd.Series(dtype=float)
            ).notna()
        ]

        if not movement_df.empty:

            summary_rows.append(
                {
                    "Metric":
                        "Mean centroid displacement",

                    "Value":
                        movement_df[
                            "centroid_displacement_um"
                        ].mean()
                }
            )

            summary_rows.append(
                {
                    "Metric":
                        "Mean soma displacement",

                    "Value":
                        movement_df[
                            "soma_displacement_um"
                        ].mean()
                }
            )

            if (
                "centroid_velocity_um_per_min"
                in movement_df.columns
            ):

                velocity_values = (
                    movement_df[
                        "centroid_velocity_um_per_min"
                    ]
                    .dropna()
                )

                if not velocity_values.empty:

                    summary_rows.append(
                        {
                            "Metric":
                                "Mean centroid velocity",

                            "Value":
                                velocity_values.mean()
                        }
                    )

            if (
                "soma_velocity_um_per_min"
                in movement_df.columns
            ):

                soma_velocity_values = (
                    movement_df[
                        "soma_velocity_um_per_min"
                    ]
                    .dropna()
                )

                if not soma_velocity_values.empty:

                    summary_rows.append(
                        {
                            "Metric":
                                "Mean soma velocity",

                            "Value":
                                soma_velocity_values.mean()
                        }
                    )

    summary_rows.extend(
        [
            {
                "Metric":
                    "Matched process-tip transitions",

                "Value":
                    len(tip_df)
            },

            {
                "Metric":
                    "Process-tip extensions",

                "Value":
                    extension_count
            },

            {
                "Metric":
                    "Process-tip retractions",

                "Value":
                    retraction_count
            },

            {
                "Metric":
                    "Process-tip stable transitions",

                "Value":
                    stable_count
            }
        ]
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        OUTPUT_DIR /
        "microglia_motion_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    # --------------------------------------------------------
    # EXCEL
    # --------------------------------------------------------

    excel_path = (
        OUTPUT_DIR /
        "microglia_motion_analysis.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        cell_df.to_excel(
            writer,
            sheet_name="Cell Motion",
            index=False
        )

        tip_df.to_excel(
            writer,
            sheet_name="Process Tip Motion",
            index=False
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("MICROGLIA MOVEMENT ANALYSIS COMPLETE")
    print("=" * 78)

    print(
        "Cell motion rows:",
        len(cell_df)
    )

    print(
        "Process-tip transitions:",
        len(tip_df)
    )

    print(
        "Extensions:",
        extension_count
    )

    print(
        "Retractions:",
        retraction_count
    )

    print(
        "Stable:",
        stable_count
    )

    if FRAME_INTERVAL_SECONDS is None:

        print()
        print(
            "NOTE: Frame interval is unknown."
        )

        print(
            "Speeds are therefore not reported "
            "in µm/min."
        )

        print(
            "The CSV still contains displacement "
            "in µm per frame."
        )

    else:

        print()
        print(
            "Frame interval:",
            FRAME_INTERVAL_SECONDS,
            "seconds"
        )

    print()
    print(
        "Excel:",
        excel_path
    )

    print(
        "Cell CSV:",
        cell_csv
    )

    print(
        "Tip CSV:",
        tip_csv
    )

    print(
        "Summary CSV:",
        summary_csv
    )

    print()
    print(
        "Tracking visuals:",
        OUTPUT_DIR
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
