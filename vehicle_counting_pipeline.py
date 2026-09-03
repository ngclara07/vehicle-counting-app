from __future__ import annotations

import csv
import json
import math
import subprocess
import time

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch

from rfdetr.assets.coco_classes import COCO_CLASSES
from trackers import ByteTrackTracker


# ============================================================
# APPROACH LABELS
# ============================================================


APPROACH_LABELS = {
    "left": "City centre / left -> Downtown",
    "bottom": "Bottom origin -> Downtown",
    "right": "Right-side approach -> Downtown",
    "upper": "Downtown outbound / rejected",
}


APPROACH_SHORT_LABELS = {
    "left": "LEFT",
    "bottom": "BOTTOM",
    "right": "RIGHT",
    "upper": "OUTBOUND",
}


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class PipelineConfig:
    """
    Exercise 1.2 - vehicle counting towards Downtown.

    Direction labels represent the PHYSICAL SOURCE of the
    vehicle in the recorded camera scene.

    Required scene interpretation:

        LEFT SIDE
            -> City centre / left -> Downtown

        RIGHT SIDE
            -> Right-side approach -> Downtown

        BOTTOM-RIGHT CORNER
            -> Bottom origin -> Downtown

    Direction is origin-locked. A vehicle does not change
    source label when it turns near Downtown.
    """

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    processing_width: int = 576

    # --------------------------------------------------------
    # RF-DETR
    # --------------------------------------------------------

    confidence_threshold: float = 0.45

    allowed_vehicle_classes: tuple[str, ...] = (
        "car",
        "truck",
        "bus",
    )

    # --------------------------------------------------------
    # REFERENCE CAMERA GEOMETRY
    # --------------------------------------------------------

    source_reference_width: int = 1040
    source_reference_height: int = 600

    downtown_zone_source: tuple[int, int, int, int] = (
        560,
        170,
        1035,
        430,
    )

    # --------------------------------------------------------
    # SOURCE / ENTRY REGIONS
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # These regions intentionally encode the roadway semantics
    # visible in the coursework video.
    #
    # LEFT:
    #   Main Street / City-centre traffic.
    #
    # BOTTOM:
    #   Bottom-right corner entry lane.
    #
    # RIGHT:
    #   Right-hand edge of the frame ABOVE the bottom-entry
    #   corridor.
    #
    # Consequently BOTTOM and RIGHT do not compete for the same
    # lower-right source location.
    # --------------------------------------------------------

    approach_zones_source: dict[
        str,
        tuple[int, int, int, int],
    ] = field(
        default_factory=lambda: {
            "left": (
                0,
                330,
                730,
                599,
            ),

            "bottom": (
                720,
                430,
                1039,
                599,
            ),

            "right": (
                875,
                180,
                1039,
                430,
            ),

            "upper": (
                550,
                0,
                1039,
                205,
            ),
        }
    )

    downtown_inner_margin_source: int = 10

    show_approach_zones: bool = False

    # --------------------------------------------------------
    # ORIGIN CLASSIFICATION
    # --------------------------------------------------------

    approach_lock_minimum_updates: int = 3

    approach_classification_window_updates: int = 12

    approach_lock_minimum_displacement_pixels: float = 6.0

    approach_axis_dominance_ratio: float = 0.80

    lock_approach_once_identified: bool = True

    # --------------------------------------------------------
    # NORMALISED CAMERA SOURCE BOUNDARIES
    # --------------------------------------------------------
    #
    # These are deliberately explicit.
    #
    # x <= 0.70:
    #     City-centre / left source.
    #
    # x >= 0.69 and y >= 0.715:
    #     bottom-right source.
    #
    # x >= 0.84 and y < 0.715:
    #     right-side source.
    #
    # The slight geometric tolerance is intentional because
    # RF-DETR bounding boxes jitter from frame to frame.
    # --------------------------------------------------------

    left_source_max_x_ratio: float = 0.70

    bottom_source_min_x_ratio: float = 0.69

    bottom_source_min_y_ratio: float = 0.715

    right_source_min_x_ratio: float = 0.84

    right_source_max_y_ratio: float = 0.715

    upper_source_max_y_ratio: float = 0.34

    minimum_source_motion_ratio: float = 0.50

    # --------------------------------------------------------
    # FRAME DIFFERENCING
    # --------------------------------------------------------

    frame_difference_threshold: int = 16

    gaussian_blur_size: tuple[int, int] = (
        5,
        5,
    )

    frame_difference_dilation_iterations: int = 2

    # --------------------------------------------------------
    # MOG2
    # --------------------------------------------------------

    background_history: int = 500

    background_variance_threshold: int = 28

    background_detect_shadows: bool = False

    background_warmup_frames: int = 40

    background_warmup_learning_rate: float = 0.04

    background_learning_rate: float = 0.001

    suppress_detections_during_warmup: bool = True

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    open_kernel_size: tuple[int, int] = (
        3,
        3,
    )

    close_kernel_size: tuple[int, int] = (
        7,
        5,
    )

    morph_open_iterations: int = 1

    morph_close_iterations: int = 1

    morph_dilate_iterations: int = 1

    # --------------------------------------------------------
    # MOTION VALIDATION
    # --------------------------------------------------------

    minimum_box_area: int = 220

    box_inset_ratio: float = 0.12

    minimum_frame_diff_occupancy: float = 0.008

    minimum_background_occupancy: float = 0.012

    minimum_combined_occupancy: float = 0.004

    minimum_combined_pixels: int = 10

    minimum_largest_component_area: int = 8

    motion_grid_rows: int = 2

    motion_grid_columns: int = 3

    minimum_active_grid_cells: int = 2

    minimum_grid_cell_occupancy: float = 0.008

    # --------------------------------------------------------
    # BYTETRACK
    # --------------------------------------------------------

    tracker_lost_buffer: int = 20

    tracker_activation_threshold: float = 0.25

    tracker_minimum_consecutive_frames: int = 1

    tracker_minimum_iou_threshold: float = 0.10

    tracker_high_confidence_threshold: float = 0.60

    # --------------------------------------------------------
    # TRAJECTORY / MOVEMENT
    # --------------------------------------------------------

    track_history_length: int = 14

    minimum_track_observations: int = 3

    minimum_net_displacement_pixels: float = 4.0

    minimum_net_displacement_ratio: float = 0.025

    minimum_speed_pixels_per_update: float = 0.35

    minimum_speed_ratio_per_update: float = 0.0025

    minimum_direction_consistency: float = 0.35

    movement_confirmation_updates: int = 1

    stop_confirmation_updates: int = 2

    track_state_timeout_frames: int = 60

    # --------------------------------------------------------
    # DOWNTOWN VALIDATION
    # --------------------------------------------------------

    minimum_distance_reduction_pixels: float = 4.0

    minimum_inside_zone_motion_pixels: float = 5.0

    minimum_valid_approach_displacement_pixels: float = 5.0

    allow_tracks_first_seen_inside_downtown: bool = True

    minimum_track_updates_before_count: int = 3

    minimum_frames_in_downtown_zone: int = 2

    # --------------------------------------------------------
    # DUPLICATE / ID-SWITCH PROTECTION
    # --------------------------------------------------------

    duplicate_count_memory_frames: int = 90

    duplicate_count_distance_pixels: float = 65.0

    duplicate_count_iou_threshold: float = 0.10


# ============================================================
# RESULT DATA STRUCTURES
# ============================================================


@dataclass
class TrackObservation:
    tracker_id: int
    class_name: str
    confidence: float

    box: np.ndarray
    position: tuple[float, float]

    displacement: float

    is_moving: bool
    inside_downtown: bool

    approach_side: str | None
    approach_locked: bool

    counted: bool


@dataclass
class CountEvent:
    event_number: int

    frame_number: int

    time_seconds: float

    tracker_id: int

    class_name: str

    confidence: float

    approach_side: str

    approach: str

    label_source: str

    origin_x: float
    origin_y: float

    count_reason: str

    total_count: int


@dataclass
class CountingFrameResult:
    frame_number: int
    total_frames: int

    annotated_frame: np.ndarray

    frame_difference_mask: np.ndarray

    background_mask: np.ndarray

    combined_motion_mask: np.ndarray

    confirmed_tracks: list[TrackObservation]

    new_events: list[CountEvent]

    active_vehicle_count: int

    unique_track_count: int

    total_count: int

    approach_counts: dict[str, int]

    class_counts: dict[str, int]

    vehicles_per_minute: float

    processing_fps: float

    warmup: bool

    progress: float

    downtown_zone: tuple[int, int, int, int]

    downtown_inner_zone: tuple[int, int, int, int]


# ============================================================
# BASIC HELPERS
# ============================================================


def resize_to_max_width(
    image: np.ndarray,
    maximum_width: int,
) -> np.ndarray:

    if image.shape[1] <= maximum_width:
        return image

    scale = maximum_width / image.shape[1]

    new_height = max(
        1,
        int(
            round(
                image.shape[0] * scale
            )
        ),
    )

    return cv2.resize(
        image,
        (
            int(maximum_width),
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def empty_detections(
    detections: sv.Detections,
) -> sv.Detections:

    if len(detections) == 0:
        return detections

    mask = np.zeros(
        len(detections),
        dtype=bool,
    )

    return detections[
        mask
    ]


def get_class_name(
    class_id: int | None,
) -> str:

    if class_id is None:
        return "unknown"

    class_id = int(
        class_id
    )

    if isinstance(
        COCO_CLASSES,
        dict,
    ):
        value = COCO_CLASSES.get(
            class_id
        )

        if value is None:
            value = COCO_CLASSES.get(
                str(class_id)
            )

        if value is None:
            return f"class_{class_id}"

        return str(
            value
        ).strip().lower()

    try:
        return str(
            COCO_CLASSES[
                class_id
            ]
        ).strip().lower()

    except (
        IndexError,
        KeyError,
        TypeError,
    ):
        return f"class_{class_id}"


# ============================================================
# VIDEO METADATA
# ============================================================


def get_video_metadata(
    video_path: Path | str,
) -> dict[str, Any]:

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "OpenCV could not open:\n"
            f"{video_path}"
        )

    try:
        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

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

    finally:
        capture.release()

    if (
        not math.isfinite(fps)
        or
        fps <= 0
    ):
        fps = 30.0

    duration_seconds = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


# ============================================================
# GEOMETRY
# ============================================================


def scale_rectangle(
    rectangle: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:

    x1, y1, x2, y2 = rectangle

    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def shrink_rectangle(
    rectangle: tuple[int, int, int, int],
    margin_x: int,
    margin_y: int,
) -> tuple[int, int, int, int]:

    x1, y1, x2, y2 = rectangle

    return (
        x1 + margin_x,
        y1 + margin_y,
        x2 - margin_x,
        y2 - margin_y,
    )


def get_scaled_downtown_geometry(
    frame_width: int,
    frame_height: int,
    config: PipelineConfig,
):

    scale_x = (
        frame_width
        / config.source_reference_width
    )

    scale_y = (
        frame_height
        / config.source_reference_height
    )

    downtown_zone = scale_rectangle(
        config.downtown_zone_source,
        scale_x,
        scale_y,
    )

    margin_x = max(
        1,
        int(
            round(
                config.downtown_inner_margin_source
                * scale_x
            )
        ),
    )

    margin_y = max(
        1,
        int(
            round(
                config.downtown_inner_margin_source
                * scale_y
            )
        ),
    )

    downtown_inner_zone = shrink_rectangle(
        downtown_zone,
        margin_x,
        margin_y,
    )

    approach_zones = {
        side: scale_rectangle(
            rectangle,
            scale_x,
            scale_y,
        )
        for (
            side,
            rectangle,
        ) in config.approach_zones_source.items()
    }

    return (
        downtown_zone,
        downtown_inner_zone,
        approach_zones,
    )


def point_in_rectangle(
    point: tuple[float, float],
    rectangle: tuple[int, int, int, int],
) -> bool:

    px, py = point

    x1, y1, x2, y2 = rectangle

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def distance_between_points(
    point_a,
    point_b,
) -> float:

    return math.hypot(
        float(point_a[0])
        - float(point_b[0]),
        float(point_a[1])
        - float(point_b[1]),
    )


def distance_point_to_rectangle(
    point,
    rectangle,
) -> float:

    px, py = point

    x1, y1, x2, y2 = rectangle

    dx = max(
        x1 - px,
        0,
        px - x2,
    )

    dy = max(
        y1 - py,
        0,
        py - y2,
    )

    return math.hypot(
        dx,
        dy,
    )


def infer_scene_dimensions(
    approach_zones,
) -> tuple[float, float]:

    maximum_x = max(
        rectangle[2]
        for rectangle in approach_zones.values()
    )

    maximum_y = max(
        rectangle[3]
        for rectangle in approach_zones.values()
    )

    return (
        float(maximum_x + 1),
        float(maximum_y + 1),
    )


# ============================================================
# FRAME DIFFERENCING / MOG2
# ============================================================


def preprocess_grayscale(
    frame_bgr: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:

    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.GaussianBlur(
        gray,
        config.gaussian_blur_size,
        0,
    )


def calculate_frame_difference(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:

    frame_delta = cv2.absdiff(
        previous_gray,
        current_gray,
    )

    _, difference_mask = cv2.threshold(
        frame_delta,
        config.frame_difference_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            5,
            5,
        ),
    )

    return cv2.dilate(
        difference_mask,
        kernel,
        iterations=(
            config.frame_difference_dilation_iterations
        ),
    )


def refine_motion_mask(
    mask: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        config.open_kernel_size,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        config.close_kernel_size,
    )

    refined = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=(
            config.morph_open_iterations
        ),
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=(
            config.morph_close_iterations
        ),
    )

    refined = cv2.dilate(
        refined,
        open_kernel,
        iterations=(
            config.morph_dilate_iterations
        ),
    )

    return refined


# ============================================================
# DETECTION / MOTION VALIDATION
# ============================================================


def detection_bottom_centre(
    box: np.ndarray,
) -> tuple[float, float]:

    x1, _, x2, y2 = map(
        float,
        box,
    )

    return (
        (x1 + x2) / 2.0,
        y2,
    )


def filter_vehicle_classes(
    detections: sv.Detections,
    config: PipelineConfig,
) -> sv.Detections:

    if len(detections) == 0:
        return detections

    if detections.class_id is None:
        return empty_detections(
            detections
        )

    allowed = {
        class_name.lower()
        for class_name
        in config.allowed_vehicle_classes
    }

    keep = np.asarray(
        [
            get_class_name(
                class_id
            )
            in allowed
            for class_id in detections.class_id
        ],
        dtype=bool,
    )

    return detections[
        keep
    ]


def crop_box_region(
    mask: np.ndarray,
    box: np.ndarray,
    inset_ratio: float,
) -> np.ndarray | None:

    height, width = mask.shape[:2]

    x1, y1, x2, y2 = (
        np.round(
            box
        ).astype(int)
    )

    box_width = max(
        1,
        x2 - x1,
    )

    box_height = max(
        1,
        y2 - y1,
    )

    inset_x = int(
        box_width
        * inset_ratio
    )

    inset_y = int(
        box_height
        * inset_ratio
    )

    x1 = max(
        0,
        x1 + inset_x,
    )

    y1 = max(
        0,
        y1 + inset_y,
    )

    x2 = min(
        width,
        x2 - inset_x,
    )

    y2 = min(
        height,
        y2 - inset_y,
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        return None

    return mask[
        y1:y2,
        x1:x2,
    ]


def mask_occupancy(
    region: np.ndarray | None,
) -> float:

    if (
        region is None
        or
        region.size == 0
    ):
        return 0.0

    return (
        float(
            cv2.countNonZero(
                region
            )
        )
        / float(
            region.size
        )
    )


def largest_component_area(
    region: np.ndarray | None,
) -> int:

    if (
        region is None
        or
        region.size == 0
    ):
        return 0

    (
        number_labels,
        _,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        region,
        connectivity=8,
    )

    if number_labels <= 1:
        return 0

    return int(
        stats[
            1:,
            cv2.CC_STAT_AREA,
        ].max()
    )


def active_grid_cells(
    region: np.ndarray | None,
    rows: int,
    columns: int,
    minimum_occupancy: float,
) -> int:

    if (
        region is None
        or
        region.size == 0
    ):
        return 0

    height, width = region.shape[:2]

    active = 0

    for row in range(
        rows
    ):
        y1 = int(
            round(
                row * height / rows
            )
        )

        y2 = int(
            round(
                (row + 1)
                * height
                / rows
            )
        )

        for column in range(
            columns
        ):
            x1 = int(
                round(
                    column
                    * width
                    / columns
                )
            )

            x2 = int(
                round(
                    (column + 1)
                    * width
                    / columns
                )
            )

            cell = region[
                y1:y2,
                x1:x2,
            ]

            if (
                mask_occupancy(
                    cell
                )
                >= minimum_occupancy
            ):
                active += 1

    return active


def filter_motion_validated_vehicles(
    detections: sv.Detections,
    frame_difference_mask: np.ndarray,
    background_mask: np.ndarray,
    combined_motion_mask: np.ndarray,
    config: PipelineConfig,
) -> sv.Detections:

    if len(detections) == 0:
        return detections

    keep: list[bool] = []

    for box in detections.xyxy:
        x1, y1, x2, y2 = map(
            float,
            box,
        )

        box_area = max(
            0.0,
            (x2 - x1)
            * (y2 - y1),
        )

        if (
            box_area
            < config.minimum_box_area
        ):
            keep.append(
                False
            )

            continue

        diff_region = crop_box_region(
            frame_difference_mask,
            box,
            config.box_inset_ratio,
        )

        bg_region = crop_box_region(
            background_mask,
            box,
            config.box_inset_ratio,
        )

        fused_region = crop_box_region(
            combined_motion_mask,
            box,
            config.box_inset_ratio,
        )

        diff_occupancy = mask_occupancy(
            diff_region
        )

        bg_occupancy = mask_occupancy(
            bg_region
        )

        fused_occupancy = mask_occupancy(
            fused_region
        )

        if (
            fused_region is not None
            and
            fused_region.size > 0
        ):
            fused_pixels = cv2.countNonZero(
                fused_region
            )

        else:
            fused_pixels = 0

        component_area = largest_component_area(
            fused_region
        )

        grid_cells = active_grid_cells(
            fused_region,
            config.motion_grid_rows,
            config.motion_grid_columns,
            config.minimum_grid_cell_occupancy,
        )

        valid_motion = (
            diff_occupancy
            >= config.minimum_frame_diff_occupancy

            and
            bg_occupancy
            >= config.minimum_background_occupancy

            and
            fused_occupancy
            >= config.minimum_combined_occupancy

            and
            fused_pixels
            >= config.minimum_combined_pixels

            and
            component_area
            >= config.minimum_largest_component_area

            and
            grid_cells
            >= config.minimum_active_grid_cells
        )

        keep.append(
            valid_motion
        )

    return detections[
        np.asarray(
            keep,
            dtype=bool,
        )
    ]


# ============================================================
# TRAJECTORY MOVEMENT
# ============================================================


def calculate_trajectory_metrics(
    history,
    box,
    config: PipelineConfig,
    pixel_scale: float,
) -> dict[str, float | bool]:

    if (
        len(history)
        < config.minimum_track_observations
    ):
        return {
            "enough_history": False,
            "net_displacement": 0.0,
            "speed": 0.0,
            "consistency": 0.0,
            "required_displacement": 0.0,
            "required_speed": 0.0,
        }

    points = np.asarray(
        history,
        dtype=np.float32,
    )

    step_vectors = np.diff(
        points,
        axis=0,
    )

    step_lengths = np.linalg.norm(
        step_vectors,
        axis=1,
    )

    total_path_length = float(
        np.sum(
            step_lengths
        )
    )

    net_vector = (
        points[-1]
        - points[0]
    )

    net_displacement = float(
        np.linalg.norm(
            net_vector
        )
    )

    indices = np.arange(
        len(points),
        dtype=np.float32,
    )

    slope_x = float(
        np.polyfit(
            indices,
            points[:, 0],
            1,
        )[0]
    )

    slope_y = float(
        np.polyfit(
            indices,
            points[:, 1],
            1,
        )[0]
    )

    speed = math.hypot(
        slope_x,
        slope_y,
    )

    if total_path_length > 1e-6:
        consistency = (
            net_displacement
            / total_path_length
        )

    else:
        consistency = 0.0

    x1, y1, x2, y2 = map(
        float,
        box,
    )

    diagonal = math.hypot(
        max(
            1.0,
            x2 - x1,
        ),
        max(
            1.0,
            y2 - y1,
        ),
    )

    required_displacement = max(
        config.minimum_net_displacement_pixels
        * pixel_scale,
        diagonal
        * config.minimum_net_displacement_ratio,
    )

    required_speed = max(
        config.minimum_speed_pixels_per_update
        * pixel_scale,
        diagonal
        * config.minimum_speed_ratio_per_update,
    )

    return {
        "enough_history": True,
        "net_displacement": net_displacement,
        "speed": speed,
        "consistency": consistency,
        "required_displacement": required_displacement,
        "required_speed": required_speed,
    }


def trajectory_confirms_movement(
    history,
    box,
    config: PipelineConfig,
    pixel_scale: float,
) -> tuple[bool, float]:

    metrics = calculate_trajectory_metrics(
        history,
        box,
        config,
        pixel_scale,
    )

    if not metrics[
        "enough_history"
    ]:
        return (
            False,
            0.0,
        )

    moving = (
        metrics[
            "net_displacement"
        ]
        >= metrics[
            "required_displacement"
        ]

        and
        metrics[
            "speed"
        ]
        >= metrics[
            "required_speed"
        ]

        and
        metrics[
            "consistency"
        ]
        >= config.minimum_direction_consistency
    )

    return (
        bool(
            moving
        ),
        float(
            metrics[
                "net_displacement"
            ]
        ),
    )


# ============================================================
# ORIGIN CLASSIFICATION
# ============================================================


def approach_side_to_text(
    approach_side: str | None,
) -> str:

    return APPROACH_LABELS.get(
        approach_side,
        "Unknown approach",
    )


def stable_early_origin(
    origin_samples,
) -> tuple[float, float]:

    if not origin_samples:
        return (
            0.0,
            0.0,
        )

    points = np.asarray(
        origin_samples[
            :min(
                3,
                len(origin_samples),
            )
        ],
        dtype=np.float32,
    )

    origin = np.median(
        points,
        axis=0,
    )

    return (
        float(
            origin[0]
        ),
        float(
            origin[1]
        ),
    )


def robust_early_motion(
    origin_samples,
) -> tuple[float, float]:
    """
    Regression across the preserved EARLY trajectory.

    This is more stable than comparing only two detections.
    """

    if len(origin_samples) < 2:
        return (
            0.0,
            0.0,
        )

    points = np.asarray(
        origin_samples,
        dtype=np.float32,
    )

    indices = np.arange(
        len(points),
        dtype=np.float32,
    )

    if len(points) >= 3:
        slope_x = float(
            np.polyfit(
                indices,
                points[:, 0],
                1,
            )[0]
        )

        slope_y = float(
            np.polyfit(
                indices,
                points[:, 1],
                1,
            )[0]
        )

        scale = max(
            1,
            len(points) - 1,
        )

        return (
            slope_x * scale,
            slope_y * scale,
        )

    return (
        float(
            points[-1, 0]
            - points[0, 0]
        ),
        float(
            points[-1, 1]
            - points[0, 1]
        ),
    )


def estimate_backward_entry_edge(
    origin_position,
    dx: float,
    dy: float,
    scene_width: float,
    scene_height: float,
) -> str | None:
    """
    Extend the early trajectory backwards and determine which
    image boundary it most plausibly came through.

    This is a FALLBACK only.

    Explicit scene source zones take precedence.
    """

    ox, oy = map(
        float,
        origin_position,
    )

    maximum_x = (
        scene_width - 1.0
    )

    maximum_y = (
        scene_height - 1.0
    )

    epsilon = 1e-6

    candidates: list[
        tuple[
            float,
            str,
        ]
    ] = []

    # Reverse of rightward motion -> left edge.
    if dx > epsilon:
        t = (
            ox / dx
        )

        y = (
            oy
            - t * dy
        )

        if (
            t >= 0
            and
            0 <= y <= maximum_y
        ):
            candidates.append(
                (
                    t,
                    "left",
                )
            )

    # Reverse of leftward motion -> right edge.
    if dx < -epsilon:
        t = (
            (
                maximum_x - ox
            )
            / (-dx)
        )

        y = (
            oy
            - t * dy
        )

        if (
            t >= 0
            and
            0 <= y <= maximum_y
        ):
            candidates.append(
                (
                    t,
                    "right",
                )
            )

    # Reverse of downward motion -> upper edge.
    if dy > epsilon:
        t = (
            oy / dy
        )

        x = (
            ox
            - t * dx
        )

        if (
            t >= 0
            and
            0 <= x <= maximum_x
        ):
            candidates.append(
                (
                    t,
                    "upper",
                )
            )

    # Reverse of upward motion -> bottom edge.
    if dy < -epsilon:
        t = (
            (
                maximum_y - oy
            )
            / (-dy)
        )

        x = (
            ox
            - t * dx
        )

        if (
            t >= 0
            and
            0 <= x <= maximum_x
        ):
            candidates.append(
                (
                    t,
                    "bottom",
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[
        0
    ][1]


def classify_origin_approach(
    origin_position,
    origin_samples,
    approach_zones,
    downtown_zone,
    config: PipelineConfig,
    pixel_scale: float,
) -> tuple[str | None, str]:
    """
    Classify the vehicle's SOURCE.

    Correct scene interpretation:

        left side
            -> LEFT

        bottom-right corner
            -> BOTTOM

        right side above bottom-right corridor
            -> RIGHT

    Hard source geometry is used before trajectory-based
    fallbacks. This prevents turning direction from corrupting
    source identity.
    """

    if (
        len(origin_samples)
        < config.approach_lock_minimum_updates
    ):
        return (
            None,
            "waiting for stable source observations",
        )

    stable_origin = stable_early_origin(
        origin_samples
    )

    ox, oy = stable_origin

    dx, dy = robust_early_motion(
        origin_samples
    )

    (
        scene_width,
        scene_height,
    ) = infer_scene_dimensions(
        approach_zones
    )

    x_ratio = (
        ox
        / max(
            scene_width - 1.0,
            1.0,
        )
    )

    y_ratio = (
        oy
        / max(
            scene_height - 1.0,
            1.0,
        )
    )

    threshold = (
        config.approach_lock_minimum_displacement_pixels
        * pixel_scale
    )

    movement = math.hypot(
        dx,
        dy,
    )

    minimum_source_motion = (
        threshold
        * config.minimum_source_motion_ratio
    )

    if (
        movement
        < minimum_source_motion
        and
        len(origin_samples)
        < config.approach_classification_window_updates
    ):
        return (
            None,
            "waiting for sufficient early source movement",
        )

    # ========================================================
    # HARD SOURCE RULE 1:
    # BOTTOM-RIGHT CORNER -> BOTTOM
    # ========================================================
    #
    # This is deliberately evaluated first.
    #
    # A bottom-right vehicle may immediately turn left/up.
    # Its leftward component must NOT convert it into RIGHT.
    # ========================================================

    if (
        x_ratio
        >= config.bottom_source_min_x_ratio
        and
        y_ratio
        >= config.bottom_source_min_y_ratio
    ):
        return (
            "bottom",
            (
                "BOTTOM origin locked: vehicle first "
                "observed in calibrated bottom-right "
                "entry corridor"
            ),
        )

    # ========================================================
    # HARD SOURCE RULE 2:
    # LEFT / CITY-CENTRE ROADWAY -> LEFT
    # ========================================================
    #
    # A City-centre vehicle can be very low in the image.
    # Its low y coordinate must NOT make it a bottom vehicle.
    # ========================================================

    if (
        x_ratio
        <= config.left_source_max_x_ratio
    ):
        return (
            "left",
            (
                "City-centre/LEFT origin locked: vehicle "
                "first observed on left/Main-Street "
                "source side"
            ),
        )

    # ========================================================
    # HARD SOURCE RULE 3:
    # RIGHT EDGE ABOVE BOTTOM CORRIDOR -> RIGHT
    # ========================================================

    if (
        x_ratio
        >= config.right_source_min_x_ratio
        and
        y_ratio
        < config.right_source_max_y_ratio
    ):
        return (
            "right",
            (
                "RIGHT-side origin locked: vehicle first "
                "observed on calibrated right-side "
                "entry corridor"
            ),
        )

    # ========================================================
    # UPPER / DOWNTOWN-ORIGIN
    # ========================================================

    if (
        y_ratio
        <= config.upper_source_max_y_ratio
        and
        dy > 0
    ):
        return (
            "upper",
            (
                "Downtown/outbound origin locked "
                "from upper source"
            ),
        )

    # ========================================================
    # TRAJECTORY FALLBACK
    # ========================================================

    if (
        len(origin_samples)
        < config.approach_classification_window_updates
    ):
        return (
            None,
            (
                "waiting for full early trajectory "
                "classification window"
            ),
        )

    entry_edge = estimate_backward_entry_edge(
        stable_origin,
        dx,
        dy,
        scene_width,
        scene_height,
    )

    # Bottom takes priority for lower-right ambiguous cases.
    if (
        entry_edge == "bottom"
        and
        y_ratio
        >= (
            config.bottom_source_min_y_ratio
            - 0.08
        )
        and
        x_ratio
        >= (
            config.bottom_source_min_x_ratio
            - 0.08
        )
    ):
        return (
            "bottom",
            (
                "BOTTOM origin locked: early trajectory "
                "back-projects to bottom boundary"
            ),
        )

    if entry_edge == "left":
        return (
            "left",
            (
                "City-centre/LEFT origin locked: early "
                "trajectory back-projects to left boundary"
            ),
        )

    if entry_edge == "right":
        # Do not permit RIGHT fallback in bottom-right corridor.
        if (
            y_ratio
            >= config.bottom_source_min_y_ratio
        ):
            return (
                "bottom",
                (
                    "BOTTOM origin locked: lower-right "
                    "source overrides right-edge "
                    "trajectory ambiguity"
                ),
            )

        return (
            "right",
            (
                "RIGHT-side origin locked: early trajectory "
                "back-projects to right boundary"
            ),
        )

    if entry_edge == "upper":
        return (
            "upper",
            (
                "Downtown/outbound origin locked: early "
                "trajectory back-projects to upper boundary"
            ),
        )

    # ========================================================
    # LAST GEOMETRIC / MOTION FALLBACK
    # ========================================================

    x1, y1, x2, y2 = downtown_zone

    horizontal = abs(
        dx
    )

    vertical = abs(
        dy
    )

    # Lower-right + any genuine upward progress => bottom.
    if (
        x_ratio
        >= (
            config.bottom_source_min_x_ratio
            - 0.08
        )
        and
        y_ratio
        >= (
            config.bottom_source_min_y_ratio
            - 0.08
        )
        and
        dy < 0
    ):
        return (
            "bottom",
            (
                "BOTTOM origin locked from lower-right "
                "location + upward progress"
            ),
        )

    # Left of Downtown + rightward progress => left.
    if (
        ox < x1
        and
        dx > 0
    ):
        return (
            "left",
            (
                "City-centre/LEFT origin locked from "
                "left-of-Downtown + rightward progress"
            ),
        )

    # Genuine right-side position + leftward progress.
    if (
        ox > x2
        and
        y_ratio
        < config.right_source_max_y_ratio
        and
        dx < 0
    ):
        return (
            "right",
            (
                "RIGHT-side origin locked from "
                "right-of-Downtown + leftward progress"
            ),
        )

    # Bottom + upward movement.
    if (
        oy > y2
        and
        dy < 0
        and
        vertical
        >= horizontal
        * 0.25
    ):
        return (
            "bottom",
            (
                "BOTTOM origin locked from "
                "below-Downtown + upward progress"
            ),
        )

    return (
        None,
        (
            "source remains ambiguous; "
            "direction not locked"
        ),
    )


# ============================================================
# FIRST ACQUIRED INSIDE DOWNTOWN
# ============================================================


def infer_inside_acquisition_side(
    origin_position,
    current_position,
    downtown_zone,
    config: PipelineConfig,
    pixel_scale: float,
) -> tuple[str | None, str]:

    ox, oy = origin_position

    cx, cy = current_position

    dx = (
        cx - ox
    )

    dy = (
        cy - oy
    )

    threshold = (
        config.approach_lock_minimum_displacement_pixels
        * pixel_scale
    )

    if (
        abs(dx) < threshold
        and
        abs(dy) < threshold
    ):
        return (
            None,
            (
                "insufficient inside-zone "
                "movement"
            ),
        )

    x1, y1, x2, y2 = downtown_zone

    distances = {
        "left": abs(
            ox - x1
        ),
        "right": abs(
            ox - x2
        ),
        "bottom": abs(
            oy - y2
        ),
        "upper": abs(
            oy - y1
        ),
    }

    nearest = min(
        distances,
        key=distances.get,
    )

    if (
        nearest == "bottom"
        and
        dy < 0
    ):
        return (
            "bottom",
            (
                "inside acquisition near "
                "bottom inbound boundary"
            ),
        )

    if (
        nearest == "left"
        and
        dx > 0
    ):
        return (
            "left",
            (
                "inside acquisition near "
                "left inbound boundary"
            ),
        )

    if (
        nearest == "right"
        and
        dx < 0
    ):
        return (
            "right",
            (
                "inside acquisition near "
                "right inbound boundary"
            ),
        )

    if nearest == "upper":
        return (
            "upper",
            (
                "inside acquisition near "
                "upper/outbound boundary"
            ),
        )

    return (
        None,
        (
            "inside acquisition "
            "direction unresolved"
        ),
    )


# ============================================================
# DOWNTOWN TRAJECTORY VALIDATION
# ============================================================


def direction_matches_locked_approach(
    approach_side: str | None,
    origin_position,
    current_position,
    config: PipelineConfig,
    pixel_scale: float,
) -> bool:

    if approach_side is None:
        return False

    dx = (
        float(
            current_position[0]
        )
        - float(
            origin_position[0]
        )
    )

    dy = (
        float(
            current_position[1]
        )
        - float(
            origin_position[1]
        )
    )

    threshold = (
        config.minimum_valid_approach_displacement_pixels
        * pixel_scale
    )

    if approach_side == "left":
        return (
            dx >= threshold
        )

    if approach_side == "bottom":
        return (
            dy <= -threshold
        )

    if approach_side == "right":
        return (
            dx <= -threshold
        )

    return False


def validate_locked_trajectory_to_downtown(
    origin_position,
    current_position,
    target_zone,
    approach_side: str | None,
    config: PipelineConfig,
    pixel_scale: float,
) -> tuple[bool, str]:

    if approach_side is None:
        return (
            False,
            (
                "origin direction "
                "not locked yet"
            ),
        )

    if approach_side == "upper":
        return (
            False,
            (
                "rejected: vehicle originated "
                "from Downtown/outbound side"
            ),
        )

    if not direction_matches_locked_approach(
        approach_side,
        origin_position,
        current_position,
        config,
        pixel_scale,
    ):
        return (
            False,
            (
                "locked origin has not yet "
                "shown sufficient inbound progress"
            ),
        )

    origin_distance = distance_point_to_rectangle(
        origin_position,
        target_zone,
    )

    current_distance = distance_point_to_rectangle(
        current_position,
        target_zone,
    )

    inside = point_in_rectangle(
        current_position,
        target_zone,
    )

    minimum_reduction = (
        config.minimum_distance_reduction_pixels
        * pixel_scale
    )

    if (
        origin_distance > 0
        and
        inside
    ):
        if (
            (
                origin_distance
                - current_distance
            )
            >= minimum_reduction
            or
            current_distance == 0
        ):
            return (
                True,
                (
                    "origin-locked inbound "
                    "trajectory entered Downtown"
                ),
            )

    if (
        config.allow_tracks_first_seen_inside_downtown
        and
        origin_distance == 0
        and
        inside
        and
        distance_between_points(
            origin_position,
            current_position,
        )
        >= (
            config.minimum_inside_zone_motion_pixels
            * pixel_scale
        )
    ):
        return (
            True,
            (
                "origin-locked track first "
                "acquired inside Downtown"
            ),
        )

    return (
        False,
        (
            "vehicle has not completed "
            "the Downtown inbound condition"
        ),
    )


# ============================================================
# MOVEMENT STATE
# ============================================================


def create_movement_state() -> dict[str, Any]:

    return {
        "positive_updates": 0,
        "negative_updates": 0,
        "is_moving": False,
    }


def update_movement_state(
    state: dict[str, Any],
    movement_confirmed: bool,
    config: PipelineConfig,
) -> bool:

    if movement_confirmed:
        state[
            "positive_updates"
        ] += 1

        state[
            "negative_updates"
        ] = 0

        if (
            state[
                "positive_updates"
            ]
            >= config.movement_confirmation_updates
        ):
            state[
                "is_moving"
            ] = True

    else:
        state[
            "negative_updates"
        ] += 1

        state[
            "positive_updates"
        ] = 0

        if (
            state[
                "negative_updates"
            ]
            >= config.stop_confirmation_updates
        ):
            state[
                "is_moving"
            ] = False

    return bool(
        state[
            "is_moving"
        ]
    )


def create_counting_state(
    class_name: str,
) -> dict[str, Any]:

    return {
        "origin_position": None,

        "classification_origin_position": None,

        "origin_samples": [],

        "first_inside_downtown": None,

        "seen_outside_downtown": False,

        "frames_in_downtown": 0,

        "track_updates": 0,

        "locked_approach_side": None,

        "approach_lock_frame": None,

        "approach_label_source": None,

        "counted": False,

        "class_name": class_name,

        "count_reason": None,
    }


# ============================================================
# DUPLICATE / ID-SWITCH PROTECTION
# ============================================================


def calculate_box_iou(
    box_a,
    box_b,
) -> float:

    ax1, ay1, ax2, ay2 = map(
        float,
        box_a,
    )

    bx1, by1, bx2, by2 = map(
        float,
        box_b,
    )

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    width = max(
        0.0,
        ix2 - ix1,
    )

    height = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        width
        * height
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        * max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        * max(
            0.0,
            by2 - by1,
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


def remove_old_count_memory(
    memory: list[dict[str, Any]],
    frame_number: int,
    config: PipelineConfig,
) -> list[dict[str, Any]]:

    return [
        item
        for item in memory
        if (
            frame_number
            - item[
                "frame"
            ]
            <= config.duplicate_count_memory_frames
        )
    ]


def update_existing_count_memory(
    memory: list[dict[str, Any]],
    tracker_id: int,
    frame_number: int,
    position,
    box,
    class_name: str,
) -> None:

    for item in memory:
        if (
            item[
                "tracker_id"
            ]
            != tracker_id
        ):
            continue

        item[
            "frame"
        ] = frame_number

        item[
            "position"
        ] = tuple(
            map(
                float,
                position,
            )
        )

        item[
            "box"
        ] = tuple(
            map(
                float,
                box,
            )
        )

        item[
            "class_name"
        ] = class_name

        return


def recently_counted_same_vehicle(
    tracker_id: int,
    position,
    box,
    frame_number: int,
    memory: list[dict[str, Any]],
    config: PipelineConfig,
    pixel_scale: float,
) -> bool:

    maximum_distance = (
        config.duplicate_count_distance_pixels
        * pixel_scale
    )

    for item in memory:
        if (
            item[
                "tracker_id"
            ]
            == tracker_id
        ):
            continue

        frame_gap = (
            frame_number
            - item[
                "frame"
            ]
        )

        if (
            frame_gap < 0
            or
            frame_gap
            > config.duplicate_count_memory_frames
        ):
            continue

        position_distance = distance_between_points(
            position,
            item[
                "position"
            ],
        )

        iou = calculate_box_iou(
            box,
            item[
                "box"
            ],
        )

        if (
            position_distance
            <= maximum_distance
            or
            iou
            >= config.duplicate_count_iou_threshold
        ):
            return True

    return False


# ============================================================
# DRAWING
# ============================================================


def draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 14,
    gap_length: int = 8,
) -> None:

    start_vector = np.asarray(
        start,
        dtype=np.float32,
    )

    end_vector = np.asarray(
        end,
        dtype=np.float32,
    )

    vector = (
        end_vector
        - start_vector
    )

    length = float(
        np.linalg.norm(
            vector
        )
    )

    if length <= 0:
        return

    direction = (
        vector
        / length
    )

    distance = 0.0

    while distance < length:
        segment_start = (
            start_vector
            + direction
            * distance
        )

        segment_end = (
            start_vector
            + direction
            * min(
                distance
                + dash_length,
                length,
            )
        )

        cv2.line(
            image,
            tuple(
                np.round(
                    segment_start
                ).astype(int)
            ),
            tuple(
                np.round(
                    segment_end
                ).astype(int)
            ),
            colour,
            thickness,
            cv2.LINE_AA,
        )

        distance += (
            dash_length
            + gap_length
        )


def draw_downtown_geometry(
    image: np.ndarray,
    downtown_zone,
    downtown_inner_zone,
    approach_zones,
    show_approach_zones: bool,
) -> None:

    destination_colour = (
        255,
        205,
        40,
    )

    inbound_colour = (
        60,
        230,
        120,
    )

    outbound_colour = (
        60,
        80,
        245,
    )

    x1, y1, x2, y2 = downtown_zone

    ix1, iy1, ix2, iy2 = (
        downtown_inner_zone
    )

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            30,
            80,
            80,
        ),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.12,
        image,
        0.88,
        0,
        image,
    )

    cv2.rectangle(
        image,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        destination_colour,
        2,
    )

    cv2.rectangle(
        image,
        (
            ix1,
            iy1,
        ),
        (
            ix2,
            iy2,
        ),
        (
            0,
            175,
            255,
        ),
        1,
    )

    cv2.putText(
        image,
        "DOWNTOWN DESTINATION",
        (
            x1 + 8,
            y1 + 22,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        destination_colour,
        2,
        cv2.LINE_AA,
    )

    middle_y = int(
        (
            iy1 + iy2
        )
        / 2
    )

    middle_x = int(
        (
            ix1 + ix2
        )
        / 2
    )

    # LEFT -> Downtown
    cv2.line(
        image,
        (
            ix1,
            iy1,
        ),
        (
            ix1,
            iy2,
        ),
        inbound_colour,
        4,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        image,
        (
            max(
                0,
                ix1 - 32,
            ),
            middle_y,
        ),
        (
            ix1 + 16,
            middle_y,
        ),
        inbound_colour,
        2,
        cv2.LINE_AA,
        tipLength=0.30,
    )

    # BOTTOM -> Downtown
    cv2.line(
        image,
        (
            ix1,
            iy2,
        ),
        (
            ix2,
            iy2,
        ),
        inbound_colour,
        4,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        image,
        (
            middle_x,
            min(
                image.shape[0] - 1,
                iy2 + 32,
            ),
        ),
        (
            middle_x,
            iy2 - 16,
        ),
        inbound_colour,
        2,
        cv2.LINE_AA,
        tipLength=0.30,
    )

    # RIGHT -> Downtown
    cv2.line(
        image,
        (
            ix2,
            iy1,
        ),
        (
            ix2,
            iy2,
        ),
        inbound_colour,
        4,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        image,
        (
            min(
                image.shape[1] - 1,
                ix2 + 32,
            ),
            middle_y,
        ),
        (
            ix2 - 16,
            middle_y,
        ),
        inbound_colour,
        2,
        cv2.LINE_AA,
        tipLength=0.30,
    )

    # Upper / outbound
    draw_dashed_line(
        image,
        (
            ix1,
            iy1,
        ),
        (
            ix2,
            iy1,
        ),
        outbound_colour,
        2,
    )

    cv2.putText(
        image,
        "OUTBOUND - NOT COUNTED",
        (
            ix1 + 4,
            max(
                15,
                iy1 - 7,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        outbound_colour,
        1,
        cv2.LINE_AA,
    )

    if not show_approach_zones:
        return

    colours = {
        "left": (
            0,
            230,
            255,
        ),
        "bottom": (
            255,
            160,
            20,
        ),
        "right": (
            220,
            80,
            255,
        ),
        "upper": (
            60,
            80,
            245,
        ),
    }

    for (
        side,
        rectangle,
    ) in approach_zones.items():
        ax1, ay1, ax2, ay2 = rectangle

        colour = colours.get(
            side,
            (
                180,
                180,
                180,
            ),
        )

        cv2.rectangle(
            image,
            (
                ax1,
                ay1,
            ),
            (
                ax2,
                ay2,
            ),
            colour,
            1,
        )

        cv2.putText(
            image,
            (
                "SOURCE: "
                + APPROACH_SHORT_LABELS.get(
                    side,
                    side.upper(),
                )
            ),
            (
                ax1 + 4,
                min(
                    image.shape[0] - 5,
                    ay1 + 16,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            colour,
            1,
            cv2.LINE_AA,
        )


def draw_trajectory(
    image: np.ndarray,
    history,
) -> None:

    points = list(
        history
    )

    if len(points) < 2:
        return

    for index in range(
        1,
        len(points),
    ):
        first = tuple(
            np.round(
                points[
                    index - 1
                ]
            ).astype(int)
        )

        second = tuple(
            np.round(
                points[
                    index
                ]
            ).astype(int)
        )

        cv2.line(
            image,
            first,
            second,
            (
                255,
                190,
                40,
            ),
            2,
            cv2.LINE_AA,
        )


def draw_detection(
    image: np.ndarray,
    track: TrackObservation,
) -> None:

    x1, y1, x2, y2 = (
        np.round(
            track.box
        ).astype(int)
    )

    if track.counted:
        colour = (
            255,
            70,
            255,
        )

    elif track.approach_locked:
        colour = (
            60,
            230,
            120,
        )

    else:
        colour = (
            80,
            190,
            245,
        )

    cv2.rectangle(
        image,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        colour,
        2,
    )

    px, py = (
        np.round(
            track.position
        ).astype(int)
    )

    cv2.circle(
        image,
        (
            px,
            py,
        ),
        4,
        colour,
        -1,
    )

    label = (
        f"{track.class_name.upper()} "
        f"#{track.tracker_id:03d} "
        f"{track.confidence:.2f}"
    )

    if (
        track.approach_side
        is not None
    ):
        label += (
            " SRC:"
            + APPROACH_SHORT_LABELS.get(
                track.approach_side,
                "?",
            )
        )

    if track.counted:
        label += " COUNTED"

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        1,
    )

    label_top = max(
        0,
        y1
        - text_height
        - baseline
        - 8,
    )

    cv2.rectangle(
        image,
        (
            x1,
            label_top,
        ),
        (
            min(
                image.shape[1] - 1,
                x1
                + text_width
                + 10,
            ),
            y1,
        ),
        (
            5,
            17,
            26,
        ),
        -1,
    )

    cv2.rectangle(
        image,
        (
            x1,
            label_top,
        ),
        (
            min(
                image.shape[1] - 1,
                x1
                + text_width
                + 10,
            ),
            y1,
        ),
        colour,
        1,
    )

    cv2.putText(
        image,
        label,
        (
            x1 + 5,
            y1 - 5,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        colour,
        1,
        cv2.LINE_AA,
    )


def draw_count_panel(
    image: np.ndarray,
    total_count: int,
    approach_counts: dict[str, int],
) -> None:

    panel_x = 15
    panel_y = 15

    panel_width = 300
    panel_height = 132

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x,
            panel_y,
        ),
        (
            panel_x
            + panel_width,
            panel_y
            + panel_height,
        ),
        (
            5,
            15,
            24,
        ),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.84,
        image,
        0.16,
        0,
        image,
    )

    cv2.rectangle(
        image,
        (
            panel_x,
            panel_y,
        ),
        (
            panel_x
            + panel_width,
            panel_y
            + panel_height,
        ),
        (
            55,
            100,
            120,
        ),
        1,
    )

    cv2.putText(
        image,
        (
            "TO DOWNTOWN: "
            f"{total_count}"
        ),
        (
            panel_x + 10,
            panel_y + 26,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (
            100,
            240,
            150,
        ),
        2,
        cv2.LINE_AA,
    )

    rows = [
        (
            "City centre / left",
            approach_counts.get(
                "left",
                0,
            ),
        ),
        (
            "Bottom origin",
            approach_counts.get(
                "bottom",
                0,
            ),
        ),
        (
            "Right origin",
            approach_counts.get(
                "right",
                0,
            ),
        ),
    ]

    for (
        index,
        (
            text,
            value,
        ),
    ) in enumerate(
        rows
    ):
        cv2.putText(
            image,
            f"{text}: {value}",
            (
                panel_x + 10,
                panel_y
                + 56
                + index * 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (
                235,
                220,
                90,
            ),
            1,
            cv2.LINE_AA,
        )


def create_diagnostic_view(
    frame_difference_mask: np.ndarray,
    background_mask: np.ndarray,
    combined_motion_mask: np.ndarray,
) -> np.ndarray:

    return np.hstack(
        [
            cv2.cvtColor(
                frame_difference_mask,
                cv2.COLOR_GRAY2BGR,
            ),
            cv2.cvtColor(
                background_mask,
                cv2.COLOR_GRAY2BGR,
            ),
            cv2.cvtColor(
                combined_motion_mask,
                cv2.COLOR_GRAY2BGR,
            ),
        ]
    )


# ============================================================
# OUTPUT VIDEO
# ============================================================


def transcode_h264(
    input_path: Path,
) -> Path:

    try:
        import imageio_ffmpeg

    except ImportError:
        return input_path

    if not input_path.is_file():
        return input_path

    output_path = input_path.with_name(
        input_path.stem
        + "_browser.mp4"
    )

    try:
        ffmpeg = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if (
            output_path.is_file()
            and
            output_path.stat().st_size > 0
        ):
            return output_path

    except Exception:
        pass

    return input_path


# ============================================================
# MAIN VEHICLE COUNTING PIPELINE
# ============================================================


class VehicleCountingPipeline:

    def __init__(
        self,
        model,
        config: PipelineConfig,
    ) -> None:

        self.model = model

        self.config = config

        self.track_histories = defaultdict(
            lambda: deque(
                maxlen=(
                    self.config.track_history_length
                )
            )
        )

        self.movement_states: dict[
            int,
            dict[str, Any],
        ] = {}

        self.counting_states: dict[
            int,
            dict[str, Any],
        ] = {}

        self.last_seen_frame: dict[
            int,
            int,
        ] = {}

        self.unique_track_ids: set[int] = set()

        self.counted_tracker_ids: set[int] = set()

        self.counted_vehicle_memory: list[
            dict[str, Any]
        ] = []

        self.events: list[
            CountEvent
        ] = []

        self.approach_counts = Counter()

        self.class_counts = Counter()

        self.history: list[
            dict[str, Any]
        ] = []

        self.evidence_paths: list[
            Path
        ] = []

        self.diagnostic_paths: list[
            Path
        ] = []

        self.output_video_path: Path | None = None

        self.raw_output_video_path: Path | None = None

        self.event_csv_path: Path | None = None

        self.summary: dict[
            str,
            Any,
        ] = {}

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset_run_state(
        self,
    ) -> None:

        self.track_histories.clear()

        self.movement_states.clear()

        self.counting_states.clear()

        self.last_seen_frame.clear()

        self.unique_track_ids.clear()

        self.counted_tracker_ids.clear()

        self.counted_vehicle_memory.clear()

        self.events.clear()

        self.approach_counts.clear()

        self.class_counts.clear()

        self.history.clear()

        self.evidence_paths.clear()

        self.diagnostic_paths.clear()

        self.output_video_path = None

        self.raw_output_video_path = None

        self.event_csv_path = None

        self.summary = {}

    # --------------------------------------------------------
    # MOG2
    # --------------------------------------------------------

    def create_background_subtractor(
        self,
    ):

        return cv2.createBackgroundSubtractorMOG2(
            history=(
                self.config.background_history
            ),
            varThreshold=(
                self.config.background_variance_threshold
            ),
            detectShadows=(
                self.config.background_detect_shadows
            ),
        )

    # --------------------------------------------------------
    # BYTETRACK
    # --------------------------------------------------------

    def create_tracker(
        self,
        source_fps: float,
    ):

        try:
            return ByteTrackTracker(
                lost_track_buffer=(
                    self.config.tracker_lost_buffer
                ),
                frame_rate=max(
                    1.0,
                    source_fps,
                ),
                track_activation_threshold=(
                    self.config.tracker_activation_threshold
                ),
                minimum_consecutive_frames=(
                    self.config
                    .tracker_minimum_consecutive_frames
                ),
                minimum_iou_threshold=(
                    self.config
                    .tracker_minimum_iou_threshold
                ),
                high_conf_det_threshold=(
                    self.config
                    .tracker_high_confidence_threshold
                ),
            )

        except TypeError:
            try:
                return ByteTrackTracker(
                    lost_track_buffer=(
                        self.config.tracker_lost_buffer
                    ),
                    frame_rate=max(
                        1.0,
                        source_fps,
                    ),
                    track_activation_threshold=(
                        self.config
                        .tracker_activation_threshold
                    ),
                )

            except TypeError:
                return ByteTrackTracker(
                    frame_rate=max(
                        1.0,
                        source_fps,
                    )
                )

    @staticmethod
    def update_tracker(
        tracker,
        detections: sv.Detections,
    ) -> sv.Detections:

        if hasattr(
            tracker,
            "update_with_detections",
        ):
            return tracker.update_with_detections(
                detections
            )

        if hasattr(
            tracker,
            "update",
        ):
            return tracker.update(
                detections
            )

        raise RuntimeError(
            "No compatible ByteTrack update method found."
        )

    # --------------------------------------------------------
    # STATE CLEANUP
    # --------------------------------------------------------

    def remove_stale_states(
        self,
        frame_number: int,
    ) -> None:

        stale_ids = [
            tracker_id
            for (
                tracker_id,
                last_frame,
            ) in self.last_seen_frame.items()
            if (
                frame_number
                - last_frame
                > self.config.track_state_timeout_frames
            )
        ]

        for tracker_id in stale_ids:
            self.track_histories.pop(
                tracker_id,
                None,
            )

            self.movement_states.pop(
                tracker_id,
                None,
            )

            self.counting_states.pop(
                tracker_id,
                None,
            )

            self.last_seen_frame.pop(
                tracker_id,
                None,
            )

    # --------------------------------------------------------
    # TRACK EVALUATION
    # --------------------------------------------------------

    def evaluate_tracks_and_count(
        self,
        tracked_detections: sv.Detections,
        frame_number: int,
        source_fps: float,
        downtown_inner_zone,
        approach_zones,
        pixel_scale: float,
    ) -> tuple[
        list[TrackObservation],
        list[CountEvent],
    ]:

        observations: list[
            TrackObservation
        ] = []

        new_events: list[
            CountEvent
        ] = []

        if (
            len(tracked_detections) == 0
            or
            tracked_detections.tracker_id is None
        ):
            self.remove_stale_states(
                frame_number
            )

            return (
                observations,
                new_events,
            )

        if (
            tracked_detections.class_id
            is not None
        ):
            class_ids = (
                tracked_detections.class_id
            )

        else:
            class_ids = [
                None
            ] * len(
                tracked_detections
            )

        if (
            tracked_detections.confidence
            is not None
        ):
            confidences = (
                tracked_detections.confidence
            )

        else:
            confidences = np.ones(
                len(
                    tracked_detections
                ),
                dtype=np.float32,
            )

        self.counted_vehicle_memory = (
            remove_old_count_memory(
                self.counted_vehicle_memory,
                frame_number,
                self.config,
            )
        )

        for (
            box,
            class_id,
            confidence,
            tracker_id,
        ) in zip(
            tracked_detections.xyxy,
            class_ids,
            confidences,
            tracked_detections.tracker_id,
        ):
            if tracker_id is None:
                continue

            tracker_id = int(
                tracker_id
            )

            if tracker_id < 0:
                continue

            box = np.asarray(
                box,
                dtype=np.float32,
            )

            class_name = get_class_name(
                class_id
            )

            position = detection_bottom_centre(
                box
            )

            self.unique_track_ids.add(
                tracker_id
            )

            self.last_seen_frame[
                tracker_id
            ] = frame_number

            history = self.track_histories[
                tracker_id
            ]

            history.append(
                position
            )

            state = self.counting_states.setdefault(
                tracker_id,
                create_counting_state(
                    class_name
                ),
            )

            state[
                "track_updates"
            ] += 1

            inside_downtown = point_in_rectangle(
                position,
                downtown_inner_zone,
            )

            # ------------------------------------------------
            # PRESERVE SOURCE
            # ------------------------------------------------

            if (
                state[
                    "origin_position"
                ]
                is None
            ):
                state[
                    "origin_position"
                ] = tuple(
                    map(
                        float,
                        position,
                    )
                )

                state[
                    "first_inside_downtown"
                ] = inside_downtown

            if (
                state[
                    "locked_approach_side"
                ]
                is None
                and
                len(
                    state[
                        "origin_samples"
                    ]
                )
                < self.config
                .approach_classification_window_updates
            ):
                state[
                    "origin_samples"
                ].append(
                    tuple(
                        map(
                            float,
                            position,
                        )
                    )
                )

            # ------------------------------------------------
            # DOWNTOWN OCCUPANCY
            # ------------------------------------------------

            if not inside_downtown:
                state[
                    "seen_outside_downtown"
                ] = True

                state[
                    "frames_in_downtown"
                ] = 0

            else:
                state[
                    "frames_in_downtown"
                ] += 1

            # ------------------------------------------------
            # MOVEMENT
            # ------------------------------------------------

            (
                movement_confirmed,
                displacement,
            ) = trajectory_confirms_movement(
                history,
                box,
                self.config,
                pixel_scale,
            )

            movement_state = self.movement_states.setdefault(
                tracker_id,
                create_movement_state(),
            )

            is_moving = update_movement_state(
                movement_state,
                movement_confirmed,
                self.config,
            )

            # ------------------------------------------------
            # SOURCE LOCK
            # ------------------------------------------------

            locked_side = state[
                "locked_approach_side"
            ]

            if (
                locked_side is None
                and
                state[
                    "track_updates"
                ]
                >= self.config.approach_lock_minimum_updates
            ):
                if not state[
                    "first_inside_downtown"
                ]:
                    (
                        candidate_side,
                        label_source,
                    ) = classify_origin_approach(
                        origin_position=(
                            state[
                                "origin_position"
                            ]
                        ),
                        origin_samples=(
                            state[
                                "origin_samples"
                            ]
                        ),
                        approach_zones=(
                            approach_zones
                        ),
                        downtown_zone=(
                            downtown_inner_zone
                        ),
                        config=(
                            self.config
                        ),
                        pixel_scale=(
                            pixel_scale
                        ),
                    )

                else:
                    (
                        candidate_side,
                        label_source,
                    ) = infer_inside_acquisition_side(
                        origin_position=(
                            state[
                                "origin_position"
                            ]
                        ),
                        current_position=(
                            position
                        ),
                        downtown_zone=(
                            downtown_inner_zone
                        ),
                        config=(
                            self.config
                        ),
                        pixel_scale=(
                            pixel_scale
                        ),
                    )

                if (
                    candidate_side
                    is not None
                ):
                    state[
                        "locked_approach_side"
                    ] = candidate_side

                    state[
                        "approach_lock_frame"
                    ] = frame_number

                    state[
                        "approach_label_source"
                    ] = label_source

                    if state[
                        "origin_samples"
                    ]:
                        state[
                            "classification_origin_position"
                        ] = stable_early_origin(
                            state[
                                "origin_samples"
                            ]
                        )

                    else:
                        state[
                            "classification_origin_position"
                        ] = state[
                            "origin_position"
                        ]

            locked_side = state[
                "locked_approach_side"
            ]

            classification_origin = (
                state[
                    "classification_origin_position"
                ]
                or
                state[
                    "origin_position"
                ]
            )

            # ------------------------------------------------
            # TRAJECTORY VALIDATION
            # ------------------------------------------------

            (
                trajectory_valid,
                trajectory_reason,
            ) = validate_locked_trajectory_to_downtown(
                origin_position=(
                    classification_origin
                ),
                current_position=(
                    position
                ),
                target_zone=(
                    downtown_inner_zone
                ),
                approach_side=(
                    locked_side
                ),
                config=(
                    self.config
                ),
                pixel_scale=(
                    pixel_scale
                ),
            )

            counted = (
                tracker_id
                in self.counted_tracker_ids
            )

            if counted:
                update_existing_count_memory(
                    self.counted_vehicle_memory,
                    tracker_id,
                    frame_number,
                    position,
                    box,
                    class_name,
                )

            duplicate_recent_count = False

            if (
                not counted
                and
                trajectory_valid
            ):
                duplicate_recent_count = (
                    recently_counted_same_vehicle(
                        tracker_id=(
                            tracker_id
                        ),
                        position=(
                            position
                        ),
                        box=(
                            box
                        ),
                        frame_number=(
                            frame_number
                        ),
                        memory=(
                            self.counted_vehicle_memory
                        ),
                        config=(
                            self.config
                        ),
                        pixel_scale=(
                            pixel_scale
                        ),
                    )
                )

            # ------------------------------------------------
            # COUNT
            # ------------------------------------------------

            should_count = (
                not state[
                    "counted"
                ]

                and
                tracker_id
                not in self.counted_tracker_ids

                and
                not duplicate_recent_count

                and
                inside_downtown

                and
                state[
                    "frames_in_downtown"
                ]
                >= self.config
                .minimum_frames_in_downtown_zone

                and
                state[
                    "track_updates"
                ]
                >= self.config
                .minimum_track_updates_before_count

                and
                trajectory_valid

                and
                locked_side
                in {
                    "left",
                    "bottom",
                    "right",
                }

                and
                is_moving
            )

            if should_count:
                state[
                    "counted"
                ] = True

                state[
                    "count_reason"
                ] = trajectory_reason

                self.counted_tracker_ids.add(
                    tracker_id
                )

                self.approach_counts[
                    locked_side
                ] += 1

                self.class_counts[
                    class_name
                ] += 1

                running_total = (
                    len(
                        self.events
                    )
                    + 1
                )

                event = CountEvent(
                    event_number=(
                        running_total
                    ),
                    frame_number=(
                        frame_number
                    ),
                    time_seconds=(
                        frame_number
                        / max(
                            source_fps,
                            1e-9,
                        )
                    ),
                    tracker_id=(
                        tracker_id
                    ),
                    class_name=(
                        class_name
                    ),
                    confidence=float(
                        confidence
                    ),
                    approach_side=(
                        locked_side
                    ),
                    approach=(
                        approach_side_to_text(
                            locked_side
                        )
                    ),
                    label_source=(
                        state[
                            "approach_label_source"
                        ]
                        or
                        "origin locked"
                    ),
                    origin_x=float(
                        classification_origin[
                            0
                        ]
                    ),
                    origin_y=float(
                        classification_origin[
                            1
                        ]
                    ),
                    count_reason=(
                        trajectory_reason
                    ),
                    total_count=(
                        running_total
                    ),
                )

                self.events.append(
                    event
                )

                new_events.append(
                    event
                )

                self.counted_vehicle_memory.append(
                    {
                        "frame":
                            frame_number,

                        "tracker_id":
                            tracker_id,

                        "position":
                            tuple(
                                map(
                                    float,
                                    position,
                                )
                            ),

                        "box":
                            tuple(
                                map(
                                    float,
                                    box,
                                )
                            ),

                        "class_name":
                            class_name,

                        "approach_side":
                            locked_side,
                    }
                )

                counted = True

            observations.append(
                TrackObservation(
                    tracker_id=(
                        tracker_id
                    ),
                    class_name=(
                        class_name
                    ),
                    confidence=float(
                        confidence
                    ),
                    box=(
                        box
                    ),
                    position=(
                        position
                    ),
                    displacement=(
                        displacement
                    ),
                    is_moving=(
                        is_moving
                    ),
                    inside_downtown=(
                        inside_downtown
                    ),
                    approach_side=(
                        locked_side
                    ),
                    approach_locked=(
                        locked_side
                        is not None
                    ),
                    counted=(
                        counted
                    ),
                )
            )

        self.remove_stale_states(
            frame_number
        )

        return (
            observations,
            new_events,
        )

    # --------------------------------------------------------
    # ANNOTATION
    # --------------------------------------------------------

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        tracks: list[TrackObservation],
        downtown_zone,
        downtown_inner_zone,
        approach_zones,
    ) -> np.ndarray:

        annotated = frame_bgr.copy()

        draw_downtown_geometry(
            annotated,
            downtown_zone,
            downtown_inner_zone,
            approach_zones,
            self.config.show_approach_zones,
        )

        for track in tracks:
            draw_trajectory(
                annotated,
                self.track_histories[
                    track.tracker_id
                ],
            )

            draw_detection(
                annotated,
                track,
            )

        draw_count_panel(
            annotated,
            len(
                self.events
            ),
            {
                "left": int(
                    self.approach_counts.get(
                        "left",
                        0,
                    )
                ),
                "bottom": int(
                    self.approach_counts.get(
                        "bottom",
                        0,
                    )
                ),
                "right": int(
                    self.approach_counts.get(
                        "right",
                        0,
                    )
                ),
            },
        )

        return annotated

    # --------------------------------------------------------
    # EVENT CSV
    # --------------------------------------------------------

    def save_events_csv(
        self,
        output_directory: Path,
    ) -> Path:

        path = (
            output_directory
            / "vehicle_counting_events.csv"
        )

        fieldnames = [
            "event_number",
            "frame_number",
            "time_seconds",
            "tracker_id",
            "class_name",
            "confidence",
            "approach_side",
            "approach",
            "label_source",
            "origin_x",
            "origin_y",
            "count_reason",
            "total_count",
        ]

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for event in self.events:
                writer.writerow(
                    asdict(
                        event
                    )
                )

        return path

    # --------------------------------------------------------
    # VIDEO PROCESSING
    # --------------------------------------------------------

    def process_video(
        self,
        video_path: Path | str,
        output_directory: Path | str,
        save_output_video: bool = True,
        save_evidence: bool = True,
    ):

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.reset_run_state()

        with open(
            output_directory
            / "run_config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(
                    self.config
                ),
                file,
                indent=2,
            )

        metadata = get_video_metadata(
            video_path
        )

        source_fps = float(
            metadata[
                "fps"
            ]
        )

        total_frames = int(
            metadata[
                "total_frames"
            ]
        )

        duration_seconds = float(
            metadata[
                "duration_seconds"
            ]
        )

        duration_minutes = (
            duration_seconds
            / 60.0
        )

        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )

        if not capture.isOpened():
            raise RuntimeError(
                "Could not open:\n"
                f"{video_path}"
            )

        background_subtractor = (
            self.create_background_subtractor()
        )

        tracker = self.create_tracker(
            source_fps
        )

        raw_output_path = (
            output_directory
            / "exercise_1_2_vehicle_counting.mp4"
        )

        self.raw_output_video_path = (
            raw_output_path
        )

        evidence_frames = {
            max(
                2,
                int(
                    total_frames
                    * 0.25
                ),
            ),
            max(
                2,
                int(
                    total_frames
                    * 0.50
                ),
            ),
            max(
                2,
                int(
                    total_frames
                    * 0.75
                ),
            ),
        }

        video_writer = None

        frames_processed = 0

        processing_started = (
            time.perf_counter()
        )

        try:
            success, first_frame = (
                capture.read()
            )

            if not success:
                raise RuntimeError(
                    "Could not read first video frame."
                )

            first_frame = resize_to_max_width(
                first_frame,
                self.config.processing_width,
            )

            previous_gray = preprocess_grayscale(
                first_frame,
                self.config,
            )

            (
                frame_height,
                frame_width,
            ) = first_frame.shape[:2]

            (
                downtown_zone,
                downtown_inner_zone,
                approach_zones,
            ) = get_scaled_downtown_geometry(
                frame_width,
                frame_height,
                self.config,
            )

            pixel_scale = (
                frame_width
                / 576.0
            )

            background_subtractor.apply(
                first_frame,
                learningRate=1.0,
            )

            if save_output_video:
                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                video_writer = cv2.VideoWriter(
                    str(
                        raw_output_path
                    ),
                    fourcc,
                    source_fps,
                    (
                        frame_width,
                        frame_height,
                    ),
                )

                if not video_writer.isOpened():
                    raise RuntimeError(
                        "Could not create output video."
                    )

            frame_number = 1

            while True:
                success, frame_bgr = (
                    capture.read()
                )

                if not success:
                    break

                frame_started = (
                    time.perf_counter()
                )

                frame_number += 1

                frames_processed += 1

                frame_bgr = resize_to_max_width(
                    frame_bgr,
                    self.config.processing_width,
                )

                current_gray = preprocess_grayscale(
                    frame_bgr,
                    self.config,
                )

                # ============================================
                # I. FRAME DIFFERENCING
                # ============================================

                frame_difference_mask = (
                    calculate_frame_difference(
                        previous_gray,
                        current_gray,
                        self.config,
                    )
                )

                # ============================================
                # II. MOG2
                # ============================================

                warmup = (
                    frame_number
                    <= self.config
                    .background_warmup_frames
                )

                if warmup:
                    learning_rate = (
                        self.config
                        .background_warmup_learning_rate
                    )

                else:
                    learning_rate = (
                        self.config
                        .background_learning_rate
                    )

                background_mask = (
                    background_subtractor.apply(
                        frame_bgr,
                        learningRate=(
                            learning_rate
                        ),
                    )
                )

                _, background_mask = cv2.threshold(
                    background_mask,
                    200,
                    255,
                    cv2.THRESH_BINARY,
                )

                # ============================================
                # III. MOTION FUSION
                # ============================================

                combined_motion_mask = cv2.bitwise_and(
                    frame_difference_mask,
                    background_mask,
                )

                combined_motion_mask = refine_motion_mask(
                    combined_motion_mask,
                    self.config,
                )

                # ============================================
                # IV. RF-DETR
                # ============================================

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                with torch.inference_mode():
                    detections = self.model.predict(
                        frame_rgb,
                        threshold=(
                            self.config
                            .confidence_threshold
                        ),
                    )

                vehicle_detections = (
                    filter_vehicle_classes(
                        detections,
                        self.config,
                    )
                )

                # ============================================
                # V. MOTION VALIDATION
                # ============================================

                if (
                    warmup
                    and
                    self.config
                    .suppress_detections_during_warmup
                ):
                    motion_validated = (
                        empty_detections(
                            vehicle_detections
                        )
                    )

                else:
                    motion_validated = (
                        filter_motion_validated_vehicles(
                            detections=(
                                vehicle_detections
                            ),
                            frame_difference_mask=(
                                frame_difference_mask
                            ),
                            background_mask=(
                                background_mask
                            ),
                            combined_motion_mask=(
                                combined_motion_mask
                            ),
                            config=(
                                self.config
                            ),
                        )
                    )

                # ============================================
                # VI. BYTETRACK
                # ============================================

                tracked_detections = (
                    self.update_tracker(
                        tracker,
                        motion_validated,
                    )
                )

                # ============================================
                # VII. ORIGIN-LOCKED COUNT
                # ============================================

                (
                    tracked_observations,
                    new_events,
                ) = self.evaluate_tracks_and_count(
                    tracked_detections=(
                        tracked_detections
                    ),
                    frame_number=(
                        frame_number
                    ),
                    source_fps=(
                        source_fps
                    ),
                    downtown_inner_zone=(
                        downtown_inner_zone
                    ),
                    approach_zones=(
                        approach_zones
                    ),
                    pixel_scale=(
                        pixel_scale
                    ),
                )

                # ============================================
                # VIII. ANNOTATION
                # ============================================

                annotated_frame = (
                    self.annotate_frame(
                        frame_bgr=(
                            frame_bgr
                        ),
                        tracks=(
                            tracked_observations
                        ),
                        downtown_zone=(
                            downtown_zone
                        ),
                        downtown_inner_zone=(
                            downtown_inner_zone
                        ),
                        approach_zones=(
                            approach_zones
                        ),
                    )
                )

                if (
                    save_output_video
                    and
                    video_writer is not None
                ):
                    video_writer.write(
                        annotated_frame
                    )

                # ============================================
                # IX. EVIDENCE
                # ============================================

                if (
                    save_evidence
                    and
                    frame_number
                    in evidence_frames
                ):
                    evidence_path = (
                        output_directory
                        / (
                            "evidence_frame_"
                            f"{frame_number}.jpg"
                        )
                    )

                    cv2.imwrite(
                        str(
                            evidence_path
                        ),
                        annotated_frame,
                    )

                    self.evidence_paths.append(
                        evidence_path
                    )

                    diagnostic = create_diagnostic_view(
                        frame_difference_mask,
                        background_mask,
                        combined_motion_mask,
                    )

                    diagnostic_path = (
                        output_directory
                        / (
                            "diagnostic_frame_"
                            f"{frame_number}.jpg"
                        )
                    )

                    cv2.imwrite(
                        str(
                            diagnostic_path
                        ),
                        diagnostic,
                    )

                    self.diagnostic_paths.append(
                        diagnostic_path
                    )

                # ============================================
                # X. TELEMETRY
                # ============================================

                frame_elapsed = max(
                    1e-9,
                    time.perf_counter()
                    - frame_started,
                )

                processing_fps = (
                    1.0
                    / frame_elapsed
                )

                active_vehicle_count = sum(
                    1
                    for track in tracked_observations
                    if track.is_moving
                )

                total_count = len(
                    self.events
                )

                elapsed_video_minutes = (
                    (
                        frame_number
                        / max(
                            source_fps,
                            1e-9,
                        )
                    )
                    / 60.0
                )

                if (
                    elapsed_video_minutes
                    > 0
                ):
                    vehicles_per_minute = (
                        total_count
                        / elapsed_video_minutes
                    )

                else:
                    vehicles_per_minute = 0.0

                approach_counts = {
                    "left": int(
                        self.approach_counts.get(
                            "left",
                            0,
                        )
                    ),
                    "bottom": int(
                        self.approach_counts.get(
                            "bottom",
                            0,
                        )
                    ),
                    "right": int(
                        self.approach_counts.get(
                            "right",
                            0,
                        )
                    ),
                }

                self.history.append(
                    {
                        "frame":
                            frame_number,

                        "time_seconds":
                            frame_number
                            / source_fps,

                        "active_vehicles":
                            active_vehicle_count,

                        "unique_tracks":
                            len(
                                self.unique_track_ids
                            ),

                        "total_count":
                            total_count,

                        "left_count":
                            approach_counts[
                                "left"
                            ],

                        "bottom_count":
                            approach_counts[
                                "bottom"
                            ],

                        "right_count":
                            approach_counts[
                                "right"
                            ],

                        "vehicles_per_minute":
                            vehicles_per_minute,

                        "processing_fps":
                            processing_fps,
                    }
                )

                yield CountingFrameResult(
                    frame_number=(
                        frame_number
                    ),
                    total_frames=(
                        total_frames
                    ),
                    annotated_frame=(
                        annotated_frame
                    ),
                    frame_difference_mask=(
                        frame_difference_mask
                    ),
                    background_mask=(
                        background_mask
                    ),
                    combined_motion_mask=(
                        combined_motion_mask
                    ),
                    confirmed_tracks=(
                        tracked_observations
                    ),
                    new_events=(
                        new_events
                    ),
                    active_vehicle_count=(
                        active_vehicle_count
                    ),
                    unique_track_count=(
                        len(
                            self.unique_track_ids
                        )
                    ),
                    total_count=(
                        total_count
                    ),
                    approach_counts=(
                        approach_counts
                    ),
                    class_counts=(
                        dict(
                            self.class_counts
                        )
                    ),
                    vehicles_per_minute=(
                        vehicles_per_minute
                    ),
                    processing_fps=(
                        processing_fps
                    ),
                    warmup=(
                        warmup
                    ),
                    progress=min(
                        1.0,
                        frame_number
                        / max(
                            total_frames,
                            1,
                        ),
                    ),
                    downtown_zone=(
                        downtown_zone
                    ),
                    downtown_inner_zone=(
                        downtown_inner_zone
                    ),
                )

                previous_gray = (
                    current_gray
                )

        finally:
            capture.release()

            if video_writer is not None:
                video_writer.release()

        # ====================================================
        # FINALISE
        # ====================================================

        processing_seconds = max(
            1e-9,
            time.perf_counter()
            - processing_started,
        )

        average_processing_fps = (
            frames_processed
            / processing_seconds
        )

        if (
            save_output_video
            and
            raw_output_path.is_file()
        ):
            self.output_video_path = (
                transcode_h264(
                    raw_output_path
                )
            )

        self.event_csv_path = (
            self.save_events_csv(
                output_directory
            )
        )

        total_count = len(
            self.events
        )

        if duration_minutes > 0:
            final_vehicles_per_minute = (
                total_count
                / duration_minutes
            )

        else:
            final_vehicles_per_minute = 0.0

        confidences = [
            event.confidence
            for event in self.events
        ]

        if confidences:
            average_confidence = float(
                np.mean(
                    confidences
                )
            )

        else:
            average_confidence = 0.0

        maximum_active = max(
            (
                row[
                    "active_vehicles"
                ]
                for row in self.history
            ),
            default=0,
        )

        final_approach_counts = {
            "left": int(
                self.approach_counts.get(
                    "left",
                    0,
                )
            ),
            "bottom": int(
                self.approach_counts.get(
                    "bottom",
                    0,
                )
            ),
            "right": int(
                self.approach_counts.get(
                    "right",
                    0,
                )
            ),
        }

        self.summary = {
            "source_fps":
                source_fps,

            "source_frames":
                total_frames,

            "duration_seconds":
                duration_seconds,

            "duration_minutes":
                duration_minutes,

            "frames_processed":
                frames_processed,

            "processing_seconds":
                processing_seconds,

            "average_processing_fps":
                average_processing_fps,

            "unique_confirmed_tracks":
                len(
                    self.unique_track_ids
                ),

            "maximum_active_vehicles":
                maximum_active,

            "total_count":
                total_count,

            "vehicles_per_minute":
                final_vehicles_per_minute,

            "average_count_confidence":
                average_confidence,

            "approach_counts":
                final_approach_counts,

            "class_counts":
                dict(
                    self.class_counts
                ),

            "counted_tracker_ids":
                sorted(
                    self.counted_tracker_ids
                ),

            "direction_labelling_method": (
                "Scene-calibrated origin locking. "
                "Left-side/Main-Street sources are labelled "
                "City centre / left. Right-edge sources above "
                "the lower corner are labelled Right-side. "
                "Vehicles originating in the bottom-right "
                "corner are explicitly labelled Bottom origin. "
                "Early trajectory back-projection is used only "
                "when the initial source position is ambiguous."
            ),

            "counting_rule": (
                "Count confirmed moving vehicles entering "
                "Downtown from City-centre/left, Bottom-origin "
                "or Right-side approaches. Upper/Downtown-origin "
                "traffic is rejected."
            ),

            "downtown_zone_source":
                list(
                    self.config
                    .downtown_zone_source
                ),

            "output_video": (
                str(
                    self.output_video_path
                )
                if (
                    self.output_video_path
                    is not None
                )
                else None
            ),

            "raw_output_video": (
                str(
                    self.raw_output_video_path
                )
                if (
                    self.raw_output_video_path
                    is not None
                )
                else None
            ),

            "event_csv": (
                str(
                    self.event_csv_path
                )
                if (
                    self.event_csv_path
                    is not None
                )
                else None
            ),

            "evidence_paths": [
                str(path)
                for path in self.evidence_paths
            ],

            "diagnostic_paths": [
                str(path)
                for path in self.diagnostic_paths
            ],
        }

        with open(
            output_directory
            / "run_summary.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.summary,
                file,
                indent=2,
            )
