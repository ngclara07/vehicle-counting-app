# Launch:
# python -m streamlit run app.py

from __future__ import annotations

import os
import tempfile

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

from rfdetr import RFDETRNano

from vehicle_counting_pipeline import (
    PipelineConfig,
    VehicleCountingPipeline,
    get_video_metadata,
)


# ============================================================
# PATHS
# ============================================================


BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parent
)

MEDIA_DIR = (
    BASE_DIR
    / "media"
)

OUTPUTS_DIR = (
    BASE_DIR
    / "outputs"
)

ASSETS_DIR = (
    BASE_DIR
    / "assets"
)

MEDIA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PAGE
# ============================================================


st.set_page_config(
    page_title=(
        "Traffic Vision Lab | "
        "Exercise 1.2"
    ),
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================


css_path = (
    ASSETS_DIR
    / "styles.css"
)

if css_path.is_file():

    st.markdown(
        (
            "<style>"
            +
            css_path.read_text(
                encoding="utf-8"
            )
            +
            "</style>"
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL
# ============================================================


@st.cache_resource(
    show_spinner=False,
)
def load_detection_model():

    cpu_count = (
        os.cpu_count()
        or 1
    )

    torch.set_num_threads(
        min(
            2,
            max(
                1,
                cpu_count - 1,
            ),
        )
    )

    return RFDETRNano()


# ============================================================
# FILE HELPERS
# ============================================================


def save_uploaded_video(
    uploaded_file,
):

    suffix = (
        Path(
            uploaded_file.name
        ).suffix
        or ".mp4"
    )

    temporary_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )
    )

    temporary_file.write(
        uploaded_file
        .getbuffer()
    )

    temporary_file.close()

    return Path(
        temporary_file.name
    )


def discover_media_videos():

    supported = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }

    return sorted(
        [
            path

            for path
            in MEDIA_DIR.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in supported
            )
        ]
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================


def show_video_metadata(
    metadata,
):

    (
        c1,
        c2,
        c3,
        c4,
    ) = st.columns(4)

    c1.metric(
        "Resolution",
        (
            f"{metadata['width']} × "
            f"{metadata['height']}"
        ),
    )

    c2.metric(
        "Source FPS",
        (
            f"{metadata['fps']:.2f}"
        ),
    )

    c3.metric(
        "Frames",
        (
            f"{metadata['total_frames']:,}"
        ),
    )

    duration = (
        metadata[
            "duration_seconds"
        ]
    )

    minutes = int(
        duration // 60
    )

    seconds = int(
        duration % 60
    )

    c4.metric(
        "Duration",
        (
            f"{minutes:02d}:"
            f"{seconds:02d}"
        ),
    )


def events_to_dataframe(
    events,
):

    rows = []

    for event in events:

        rows.append(
            {
                "Event":
                    event.event_number,

                "Time":
                    (
                        f"{event.time_seconds:.2f}s"
                    ),

                "Track ID":
                    (
                        f"#{event.tracker_id:03d}"
                    ),

                "Class":
                    event
                    .class_name
                    .title(),

                "Confidence":
                    round(
                        event.confidence,
                        3,
                    ),

                "Origin / Approach":
                    event
                    .approach
                    .replace(
                        "->",
                        "→",
                    ),

                "Label source":
                    event.label_source,

                "Origin point":
                    (
                        f"("
                        f"{event.origin_x:.1f}, "
                        f"{event.origin_y:.1f}"
                        f")"
                    ),

                "Reason":
                    event.count_reason,

                "Running Total":
                    event.total_count,
            }
        )

    return pd.DataFrame(
        rows
    )


def show_final_analytics(
    summary,
    history,
    events,
):

    st.markdown(
        "### Task 2 Counting Summary"
    )

    counts = (
        summary.get(
            "approach_counts",
            {},
        )
    )

    (
        c1,
        c2,
        c3,
        c4,
        c5,
    ) = st.columns(5)

    c1.metric(
        "Total → Downtown",
        summary.get(
            "total_count",
            0,
        ),
    )

    c2.metric(
        "City Centre / Left",
        counts.get(
            "left",
            0,
        ),
    )

    c3.metric(
        "Bottom Origin",
        counts.get(
            "bottom",
            0,
        ),
    )

    c4.metric(
        "Right Origin",
        counts.get(
            "right",
            0,
        ),
    )

    c5.metric(
        "Cars / Minute",
        (
            f"{summary.get('vehicles_per_minute', 0):.2f}"
        ),
    )

    (
        p1,
        p2,
        p3,
    ) = st.columns(3)

    p1.metric(
        "Average Processing FPS",
        (
            f"{summary.get('average_processing_fps', 0):.2f}"
        ),
    )

    p2.metric(
        "Maximum Moving Vehicles",
        summary.get(
            "maximum_active_vehicles",
            0,
        ),
    )

    p3.metric(
        "Unique Track IDs",
        summary.get(
            "unique_confirmed_tracks",
            0,
        ),
    )

    if history:

        history_df = (
            pd.DataFrame(
                history
            )
        )

        if not (
            history_df.empty
        ):

            st.markdown(
                (
                    "#### Cumulative vehicles "
                    "travelling towards Downtown"
                )
            )

            st.line_chart(
                history_df
                .set_index(
                    "time_seconds"
                )[
                    [
                        "total_count"
                    ]
                ],
                use_container_width=True,
            )

            st.markdown(
                "#### Origin-locked approach counts"
            )

            approach_df = (
                history_df
                .set_index(
                    "time_seconds"
                )[
                    [
                        "left_count",
                        "bottom_count",
                        "right_count",
                    ]
                ]
                .rename(
                    columns={
                        "left_count":
                            (
                                "City centre / left"
                            ),

                        "bottom_count":
                            (
                                "Bottom origin"
                            ),

                        "right_count":
                            (
                                "Right origin"
                            ),
                    }
                )
            )

            st.line_chart(
                approach_df,
                use_container_width=True,
            )

            st.markdown(
                "#### Active moving vehicles"
            )

            st.line_chart(
                history_df
                .set_index(
                    "time_seconds"
                )[
                    [
                        "active_vehicles"
                    ]
                ],
                use_container_width=True,
            )

            st.markdown(
                "#### Vehicles per minute"
            )

            st.line_chart(
                history_df
                .set_index(
                    "time_seconds"
                )[
                    [
                        "vehicles_per_minute"
                    ]
                ],
                use_container_width=True,
            )

    class_counts = (
        summary.get(
            "class_counts",
            {},
        )
    )

    if class_counts:

        class_df = (
            pd.DataFrame(
                {
                    "Vehicle class": [
                        name.title()

                        for name
                        in class_counts
                    ],

                    "Count":
                        list(
                            class_counts
                            .values()
                        ),
                }
            )
        )

        st.markdown(
            "#### Counted vehicle classes"
        )

        st.bar_chart(
            class_df
            .set_index(
                "Vehicle class"
            ),
            use_container_width=True,
        )

    if events:

        st.markdown(
            (
                "#### Complete origin-locked "
                "Downtown event log"
            )
        )

        st.dataframe(
            events_to_dataframe(
                events
            ),
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# HEADER
# ============================================================


header_left, header_right = (
    st.columns(
        [
            5,
            1.2,
        ]
    )
)

with header_left:

    st.markdown(
        """
        <div class="eyebrow">
            CM3065 · INTELLIGENT SIGNAL PROCESSING
        </div>

        <div class="main-title">
            Traffic Vision Lab
        </div>

        <div class="main-subtitle">
            Exercise 1.2 · Vehicle Counting
            Towards Downtown From All Directions
        </div>
        """,
        unsafe_allow_html=True,
    )


with header_right:

    st.markdown(
        """
        <div class="system-status">
            <span class="status-dot"></span>
            SYSTEM READY
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PIPELINE
# ============================================================


st.markdown(
    """
    <div class="pipeline-strip">

        <span>FRAME Δ</span>
        <b>＋</b>

        <span>MOG2</span>
        <b>→</b>

        <span>MOTION FUSION</span>
        <b>→</b>

        <span>RF-DETR NANO</span>
        <b>→</b>

        <span>BYTETRACK</span>
        <b>→</b>

        <span>ORIGIN LOCK</span>
        <b>→</b>

        <span>TRAJECTORY</span>
        <b>→</b>

        <span>DOWNTOWN</span>
        <b>→</b>

        <span>COUNT</span>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.markdown(
        "## Counting Control"
    )

    st.caption(
        (
            "Exercise 1.2 · "
            "Vehicle Counting"
        )
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    st.markdown(
        "### Video Source"
    )

    media_videos = (
        discover_media_videos()
    )

    source_options = [
        "Upload video"
    ]

    source_lookup = {}

    for path in (
        media_videos
    ):

        source_options.insert(
            len(
                source_options
            )
            - 1,
            path.name,
        )

        source_lookup[
            path.name
        ] = path

    source_choice = (
        st.radio(
            "Source",
            source_options,
            label_visibility=(
                "collapsed"
            ),
        )
    )

    uploaded_video = None

    if (
        source_choice
        == "Upload video"
    ):

        uploaded_video = (
            st.file_uploader(
                "Upload traffic recording",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                ],
            )
        )

    st.divider()

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    with st.expander(
        "RF-DETR Nano",
        expanded=True,
    ):

        confidence_threshold = (
            st.slider(
                "Detection confidence",
                0.10,
                0.95,
                0.45,
                0.05,
            )
        )

        vehicle_classes = (
            st.multiselect(
                "Vehicle classes",
                [
                    "car",
                    "truck",
                    "bus",
                ],
                default=[
                    "car",
                    "truck",
                    "bus",
                ],
            )
        )

    # --------------------------------------------------------
    # FRAME DIFFERENCE
    # --------------------------------------------------------

    with st.expander(
        "Frame Differencing"
    ):

        frame_difference_threshold = (
            st.slider(
                "Difference threshold",
                1,
                100,
                16,
            )
        )

        frame_difference_dilation = (
            st.slider(
                "Dilation iterations",
                0,
                5,
                2,
            )
        )

    # --------------------------------------------------------
    # MOG2
    # --------------------------------------------------------

    with st.expander(
        "MOG2 Background Subtraction"
    ):

        background_history = (
            st.slider(
                "Background history",
                50,
                1000,
                500,
                50,
            )
        )

        background_variance = (
            st.slider(
                "Variance threshold",
                5,
                80,
                28,
            )
        )

        warmup_frames = (
            st.slider(
                "Warm-up frames",
                0,
                150,
                40,
            )
        )

        suppress_warmup = (
            st.checkbox(
                (
                    "Suppress detections "
                    "during warm-up"
                ),
                value=True,
            )
        )

    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    with st.expander(
        "Motion Validation"
    ):

        minimum_box_area = (
            st.slider(
                "Minimum box area",
                50,
                3000,
                220,
                10,
            )
        )

        min_diff_occupancy = (
            st.slider(
                (
                    "Frame-difference "
                    "occupancy"
                ),
                0.001,
                0.100,
                0.008,
                0.001,
                format="%.3f",
            )
        )

        min_bg_occupancy = (
            st.slider(
                "MOG2 occupancy",
                0.001,
                0.100,
                0.012,
                0.001,
                format="%.3f",
            )
        )

        min_fused_occupancy = (
            st.slider(
                "Fused occupancy",
                0.001,
                0.100,
                0.004,
                0.001,
                format="%.3f",
            )
        )

        minimum_component_area = (
            st.slider(
                (
                    "Minimum connected "
                    "component pixels"
                ),
                1,
                300,
                8,
            )
        )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    with st.expander(
        "ByteTrack / Trajectory"
    ):

        track_history_length = (
            st.slider(
                "Track history length",
                4,
                30,
                14,
            )
        )

        minimum_observations = (
            st.slider(
                "Minimum observations",
                2,
                10,
                3,
            )
        )

        tracker_lost_buffer = (
            st.slider(
                "Lost-track buffer",
                5,
                100,
                20,
            )
        )

    # --------------------------------------------------------
    # DIRECTIONAL LABEL FIX
    # --------------------------------------------------------

    with st.expander(
        "Directional Origin Labelling",
        expanded=True,
    ):

        st.success(
            (
                "Origin-locked labelling is enabled. "
                "A City-centre vehicle remains labelled "
                "City Centre / Left even after it turns "
                "upward near Downtown."
            )
        )

        st.caption(
            (
                "The approach is determined from the "
                "vehicle's earliest stable trajectory "
                "and is then permanently locked for "
                "that ByteTrack ID."
            )
        )

        approach_lock_updates = (
            st.slider(
                (
                    "Updates before "
                    "origin classification"
                ),
                2,
                8,
                3,
            )
        )

        approach_lock_displacement = (
            st.slider(
                (
                    "Minimum displacement "
                    "for origin lock"
                ),
                2.0,
                20.0,
                6.0,
                1.0,
            )
        )

        axis_dominance = (
            st.slider(
                (
                    "Horizontal / vertical "
                    "origin separation"
                ),
                0.50,
                1.20,
                0.80,
                0.05,
            )
        )

        show_approach_zones = (
            st.checkbox(
                (
                    "Show origin-region "
                    "debug overlays"
                ),
                value=False,
            )
        )

        st.caption(
            (
                "Enable the debug overlays temporarily "
                "to verify where LEFT, BOTTOM and RIGHT "
                "origins are being acquired."
            )
        )

    # --------------------------------------------------------
    # DOWNTOWN COUNT
    # --------------------------------------------------------

    with st.expander(
        "Downtown Destination"
    ):

        minimum_zone_frames = (
            st.slider(
                (
                    "Frames inside Downtown "
                    "before count"
                ),
                1,
                8,
                2,
            )
        )

        minimum_track_updates = (
            st.slider(
                (
                    "Minimum track updates "
                    "before count"
                ),
                2,
                10,
                3,
            )
        )

        minimum_direction_pixels = (
            st.slider(
                "Minimum inbound displacement",
                1.0,
                20.0,
                5.0,
                1.0,
            )
        )

        duplicate_distance = (
            st.slider(
                (
                    "ID-switch duplicate "
                    "distance"
                ),
                20.0,
                120.0,
                65.0,
                5.0,
            )
        )

    # --------------------------------------------------------
    # APPLICATION
    # --------------------------------------------------------

    with st.expander(
        "Application"
    ):

        processing_width = (
            st.select_slider(
                "Processing width",
                [
                    576,
                    640,
                    768,
                    854,
                    960,
                    1040,
                ],
                value=576,
            )
        )

        display_stride = (
            st.slider(
                (
                    "UI refresh every "
                    "N frames"
                ),
                1,
                15,
                3,
            )
        )

        save_output_video = (
            st.checkbox(
                "Save counting video",
                True,
            )
        )

        save_evidence = (
            st.checkbox(
                "Save evidence frames",
                True,
            )
        )

    st.divider()

    run_button = (
        st.button(
            "▶ RUN VEHICLE COUNTING",
            type="primary",
            use_container_width=True,
        )
    )

    if st.button(
        "Clear previous results",
        use_container_width=True,
    ):

        st.session_state.pop(
            "exercise_1_2_last_run",
            None,
        )


# ============================================================
# CONFIG
# ============================================================


config = PipelineConfig(

    processing_width=(
        processing_width
    ),

    confidence_threshold=(
        confidence_threshold
    ),

    allowed_vehicle_classes=tuple(
        vehicle_classes
    ),

    frame_difference_threshold=(
        frame_difference_threshold
    ),

    frame_difference_dilation_iterations=(
        frame_difference_dilation
    ),

    background_history=(
        background_history
    ),

    background_variance_threshold=(
        background_variance
    ),

    background_warmup_frames=(
        warmup_frames
    ),

    suppress_detections_during_warmup=(
        suppress_warmup
    ),

    minimum_box_area=(
        minimum_box_area
    ),

    minimum_frame_diff_occupancy=(
        min_diff_occupancy
    ),

    minimum_background_occupancy=(
        min_bg_occupancy
    ),

    minimum_combined_occupancy=(
        min_fused_occupancy
    ),

    minimum_largest_component_area=(
        minimum_component_area
    ),

    track_history_length=(
        track_history_length
    ),

    minimum_track_observations=(
        minimum_observations
    ),

    tracker_lost_buffer=(
        tracker_lost_buffer
    ),

    approach_lock_minimum_updates=(
        approach_lock_updates
    ),

    approach_lock_minimum_displacement_pixels=(
        approach_lock_displacement
    ),

    approach_axis_dominance_ratio=(
        axis_dominance
    ),

    minimum_frames_in_downtown_zone=(
        minimum_zone_frames
    ),

    minimum_track_updates_before_count=(
        minimum_track_updates
    ),

    minimum_valid_approach_displacement_pixels=(
        minimum_direction_pixels
    ),

    duplicate_count_distance_pixels=(
        duplicate_distance
    ),

    show_approach_zones=(
        show_approach_zones
    ),
)


# ============================================================
# TABS
# ============================================================


(
    counting_tab,
    diagnostics_tab,
    analytics_tab,
    evidence_tab,
    configuration_tab,
) = st.tabs(
    [
        "🚘 Live Counting",
        "🔬 Motion Diagnostics",
        "📊 Counting Analytics",
        "🖼 Evidence",
        "⚙ Configuration",
    ]
)


# ============================================================
# LIVE
# ============================================================


with counting_tab:

    st.markdown(
        "### Intelligent Vehicle Counter"
    )

    metrics = (
        st.columns(6)
    )

    total_metric = (
        metrics[0].empty()
    )

    left_metric = (
        metrics[1].empty()
    )

    bottom_metric = (
        metrics[2].empty()
    )

    right_metric = (
        metrics[3].empty()
    )

    active_metric = (
        metrics[4].empty()
    )

    fps_metric = (
        metrics[5].empty()
    )

    video_placeholder = (
        st.empty()
    )

    status_placeholder = (
        st.empty()
    )

    progress_placeholder = (
        st.empty()
    )

    st.markdown(
        "#### Latest Downtown count events"
    )

    event_table_placeholder = (
        st.empty()
    )

    final_output_container = (
        st.container()
    )


# ============================================================
# DIAGNOSTICS
# ============================================================


with diagnostics_tab:

    st.markdown(
        "### Classical Motion Analysis"
    )

    st.caption(
        (
            "Frame differencing and MOG2 "
            "must both support the semantic "
            "RF-DETR vehicle detection."
        )
    )

    (
        d1,
        d2,
        d3,
    ) = st.columns(3)

    with d1:

        st.markdown(
            "**Frame Difference**"
        )

        diff_placeholder = (
            st.empty()
        )

    with d2:

        st.markdown(
            "**MOG2 Foreground**"
        )

        bg_placeholder = (
            st.empty()
        )

    with d3:

        st.markdown(
            "**Fused Motion**"
        )

        fused_placeholder = (
            st.empty()
        )


with analytics_tab:

    analytics_container = (
        st.container()
    )


with evidence_tab:

    evidence_container = (
        st.container()
    )


# ============================================================
# CONFIGURATION
# ============================================================


with configuration_tab:

    st.markdown(
        "### Experimental Configuration"
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Parameter":
                        key,

                    "Value":
                        str(
                            value
                        ),
                }

                for (
                    key,
                    value,
                )
                in asdict(
                    config
                ).items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.success(
        (
            "Directional labelling method: "
            "ORIGIN LOCKED. A track's approach "
            "cannot change after the first stable "
            "origin classification."
        )
    )


# ============================================================
# RUN
# ============================================================


if run_button:

    source_path = None

    temporary_upload = None

    if (
        source_choice
        != "Upload video"
    ):

        source_path = (
            source_lookup[
                source_choice
            ]
        )

    elif (
        uploaded_video
        is None
    ):

        st.error(
            (
                "Please upload a video "
                "before starting."
            )
        )

        st.stop()

    else:

        temporary_upload = (
            save_uploaded_video(
                uploaded_video
            )
        )

        source_path = (
            temporary_upload
        )

    try:

        metadata = (
            get_video_metadata(
                source_path
            )
        )

        with counting_tab:

            show_video_metadata(
                metadata
            )

        status_placeholder.info(
            "Loading RF-DETR Nano..."
        )

        with st.spinner(
            "Initialising RF-DETR Nano..."
        ):

            model = (
                load_detection_model()
            )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        run_directory = (
            OUTPUTS_DIR
            /
            (
                f"run_{timestamp}"
            )
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        pipeline = (
            VehicleCountingPipeline(
                model=model,
                config=config,
            )
        )

        progress_bar = (
            progress_placeholder
            .progress(
                0.0
            )
        )

        event_history = []

        for result in (
            pipeline.process_video(

                video_path=(
                    source_path
                ),

                output_directory=(
                    run_directory
                ),

                save_output_video=(
                    save_output_video
                ),

                save_evidence=(
                    save_evidence
                ),
            )
        ):

            if result.new_events:

                event_history.extend(
                    result.new_events
                )

            refresh = (

                result.frame_number
                % display_stride
                == 0

                or

                result.progress
                >= 0.999
            )

            if not refresh:
                continue

            frame_rgb = (
                cv2.cvtColor(
                    result.annotated_frame,
                    cv2.COLOR_BGR2RGB,
                )
            )

            video_placeholder.image(
                frame_rgb,
                use_container_width=True,
            )

            total_metric.metric(
                "Total → Downtown",
                result.total_count,
            )

            left_metric.metric(
                "City Centre / Left",
                result
                .approach_counts
                .get(
                    "left",
                    0,
                ),
            )

            bottom_metric.metric(
                "Bottom Origin",
                result
                .approach_counts
                .get(
                    "bottom",
                    0,
                ),
            )

            right_metric.metric(
                "Right Origin",
                result
                .approach_counts
                .get(
                    "right",
                    0,
                ),
            )

            active_metric.metric(
                "Moving Now",
                result
                .active_vehicle_count,
            )

            fps_metric.metric(
                "Processing FPS",
                (
                    f"{result.processing_fps:.2f}"
                ),
            )

            if result.warmup:

                status_placeholder.warning(
                    (
                        "MOG2 background model "
                        "warming up..."
                    )
                )

            else:

                status_placeholder.success(
                    (
                        "● ORIGIN-LOCKED COUNTING ACTIVE · "
                        f"Frame {result.frame_number:,}"
                        f" / "
                        f"{result.total_frames:,}"
                    )
                )

            progress_bar.progress(
                result.progress
            )

            if event_history:

                event_table_placeholder.dataframe(
                    events_to_dataframe(
                        event_history[
                            -10:
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            else:

                event_table_placeholder.info(
                    (
                        "No confirmed inbound "
                        "vehicle has entered "
                        "Downtown yet."
                    )
                )

            diff_placeholder.image(
                result
                .frame_difference_mask,
                clamp=True,
                use_container_width=True,
            )

            bg_placeholder.image(
                result
                .background_mask,
                clamp=True,
                use_container_width=True,
            )

            fused_placeholder.image(
                result
                .combined_motion_mask,
                clamp=True,
                use_container_width=True,
            )

        progress_bar.progress(
            1.0
        )

        status_placeholder.success(
            (
                "✓ Origin-locked Downtown "
                "vehicle counting complete"
            )
        )

        summary = (
            pipeline.summary
        )

        st.session_state[
            "exercise_1_2_last_run"
        ] = {
            "summary":
                summary,

            "history":
                pipeline.history,

            "events":
                pipeline.events,

            "run_directory":
                str(
                    run_directory
                ),
        }

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        with final_output_container:

            st.markdown(
                "### Task 2 Result"
            )

            result_columns = (
                st.columns(2)
            )

            result_columns[
                0
            ].metric(
                (
                    "Total vehicles "
                    "towards Downtown"
                ),
                summary.get(
                    "total_count",
                    0,
                ),
            )

            result_columns[
                1
            ].metric(
                "Vehicles / minute",
                (
                    f"{summary.get('vehicles_per_minute', 0):.2f}"
                ),
            )

            output_text = (
                summary.get(
                    "output_video"
                )
            )

            if output_text:

                output_path = Path(
                    output_text
                )

                if (
                    output_path
                    .is_file()
                ):

                    st.markdown(
                        "### Final Counting Video"
                    )

                    st.video(
                        str(
                            output_path
                        )
                    )

                    with open(
                        output_path,
                        "rb",
                    ) as file:

                        st.download_button(
                            (
                                "⬇ Download "
                                "counting video"
                            ),
                            data=file,
                            file_name=(
                                "exercise_1_2_"
                                "vehicle_counting.mp4"
                            ),
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True,
                        )

            csv_text = (
                summary.get(
                    "event_csv"
                )
            )

            if csv_text:

                csv_path = Path(
                    csv_text
                )

                if csv_path.is_file():

                    with open(
                        csv_path,
                        "rb",
                    ) as file:

                        st.download_button(
                            (
                                "⬇ Download "
                                "origin-labelled events CSV"
                            ),
                            data=file,
                            file_name=(
                                "vehicle_counting_events.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True,
                        )

        with analytics_container:

            show_final_analytics(
                summary,
                pipeline.history,
                pipeline.events,
            )

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        with evidence_container:

            st.markdown(
                "### Counting Evidence"
            )

            evidence_paths = [
                Path(
                    path
                )

                for path
                in summary.get(
                    "evidence_paths",
                    [],
                )
            ]

            if evidence_paths:

                columns = (
                    st.columns(
                        min(
                            3,
                            len(
                                evidence_paths
                            ),
                        )
                    )
                )

                for (
                    index,
                    path,
                ) in enumerate(
                    evidence_paths
                ):

                    if path.is_file():

                        columns[
                            index
                            % len(columns)
                        ].image(
                            str(
                                path
                            ),
                            caption=(
                                path.stem
                            ),
                            use_container_width=True,
                        )

            diagnostic_paths = [
                Path(
                    path
                )

                for path
                in summary.get(
                    "diagnostic_paths",
                    [],
                )
            ]

            if diagnostic_paths:

                st.markdown(
                    "### Motion Diagnostic Evidence"
                )

                for path in (
                    diagnostic_paths
                ):

                    if path.is_file():

                        st.image(
                            str(
                                path
                            ),
                            caption=(
                                path.stem
                            ),
                            use_container_width=True,
                        )

        st.toast(
            (
                "Exercise 1.2 completed."
            ),
            icon="✅",
        )

    except Exception as error:

        status_placeholder.error(
            "Vehicle counting failed."
        )

        st.exception(
            error
        )

    finally:

        if (
            temporary_upload
            is not None
            and
            temporary_upload.exists()
        ):

            try:

                temporary_upload.unlink()

            except OSError:

                pass


# ============================================================
# IDLE
# ============================================================


else:

    total_metric.metric(
        "Total → Downtown",
        "—",
    )

    left_metric.metric(
        "City Centre / Left",
        "—",
    )

    bottom_metric.metric(
        "Bottom Origin",
        "—",
    )

    right_metric.metric(
        "Right Origin",
        "—",
    )

    active_metric.metric(
        "Moving Now",
        "—",
    )

    fps_metric.metric(
        "Processing FPS",
        "—",
    )

    video_placeholder.markdown(
        """
        <div class="empty-monitor">

            <div class="monitor-icon">
                ◎
            </div>

            <div class="monitor-title">
                Origin-locked vehicle counter ready
            </div>

            <div class="monitor-text">
                Direction labels represent the vehicle's
                <strong>origin</strong>, not the direction
                of its final turn near Downtown.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    status_placeholder.info(
        (
            "Waiting for a Downtown "
            "vehicle-counting run."
        )
    )

    previous = (
        st.session_state.get(
            "exercise_1_2_last_run"
        )
    )

    if previous:

        with analytics_container:

            show_final_analytics(
                previous[
                    "summary"
                ],
                previous[
                    "history"
                ],
                previous[
                    "events"
                ],
            )
