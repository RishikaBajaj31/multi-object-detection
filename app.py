from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "Ultralytics"))

from main import run_pipeline


st.set_page_config(page_title="Multi-Object Tracking", layout="wide")

if "result_bundle" not in st.session_state:
    st.session_state.result_bundle = None

st.title("Multi-Object Detection and Persistent ID Tracking")
st.caption("DeepSORT-based public marathon tracking demo with downloadable outputs.")

DEFAULT_SOURCE_URL = "https://samplelib.com/sample-mp4.html"
DEFAULT_DOWNLOAD_URL = "https://samplelib.com/mp4/sample-30s.mp4"
DEFAULT_FILENAME = "default_marathon.mp4"
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DEFAULT_CANDIDATES = [
    PROJECT_ROOT / "sample_videos" / DEFAULT_FILENAME,
    PROJECT_ROOT / DEFAULT_FILENAME,
    Path.home() / "Downloads" / "marathon-15741.mp4",
]


def resolve_default_video(target_path: Path) -> Path:
    """
    Resolve a hidden default public marathon sample for the app.

    Resolution order:
    1. bundled local sample in the repo
    2. downloaded local sample in the user's Downloads folder
    3. backend download from the public source URL

    The public URL is intentionally kept out of the UI, but the summary JSON still records
    the original source used for the assignment.
    """
    for candidate in LOCAL_DEFAULT_CANDIDATES:
        if candidate.exists():
            destination = target_path / DEFAULT_FILENAME
            shutil.copy2(candidate, destination)
            return destination

    return download_default_video(target_path)


def download_default_video(target_path: Path) -> Path:
    downloaded_path = target_path / DEFAULT_FILENAME
    request = urllib.request.Request(
        DEFAULT_DOWNLOAD_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, downloaded_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)

    if not downloaded_path.exists() or downloaded_path.stat().st_size == 0:
        raise FileNotFoundError("Default sample video download did not produce the expected file.")
    return downloaded_path


st.markdown(
    """
    <style>
    .hero-card {
        padding: 1.25rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(12,32,57,0.95), rgba(19,76,89,0.88));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .hero-text {
        color: rgba(255,255,255,0.82);
        font-size: 0.98rem;
    }
    </style>
    <div class="hero-card">
        <div class="hero-title">Default Demo Source</div>
        <div class="hero-text">
            The app runs a built-in public sample by default and processes it with YOLOv8 + DeepSORT.
            Results include annotated video, tracking CSV, summary JSON, preview image, and optional heatmap.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1.3, 1.2, 1.0])
with col1:
    model_name = st.text_input("Model", value="yolov8n.pt", disabled=True)
with col2:
    image_size = st.number_input("Image Size", min_value=320, max_value=1920, value=640, step=32)
with col3:
    save_heatmap = st.checkbox("Save Heatmap", value=True)

frame_skip = 2

if st.button("Run Tracking", type="primary", use_container_width=True):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            with st.spinner("Preparing the default public sports sample..."):
                input_path = resolve_default_video(tmp_path)
            source_url = DEFAULT_SOURCE_URL

            output_dir = tmp_path / "outputs"
            args = SimpleNamespace(
                input=str(input_path),
                output_dir=str(output_dir),
                source_url=source_url,
                model=model_name.strip() or "yolov8n.pt",
                conf=0.35,
                iou=0.45,
                imgsz=int(image_size),
                classes="0",
                max_frames=0,
                log_every=25,
                trajectory_length=30,
                save_heatmap=save_heatmap,
                frame_skip=frame_skip,
            )

            with st.spinner("Processing video. This can take a few minutes on CPU."):
                summary = run_pipeline(args)

            output_video = Path(summary["output_video"])
            tracking_log = Path(summary["tracking_log"])
            preview_frame = Path(summary["preview_frame"])
            heatmap_path = Path(summary["heatmap_path"]) if summary.get("heatmap_path") else None
            summary_path = output_dir / f"{input_path.stem}_summary.json"
            tracking_df = pd.read_csv(tracking_log) if tracking_log.exists() else pd.DataFrame()
            st.session_state.result_bundle = {
                "summary": summary,
                "output_video_name": output_video.name if output_video.exists() else None,
                "output_video_bytes": output_video.read_bytes() if output_video.exists() else None,
                "tracking_log_name": tracking_log.name if tracking_log.exists() else None,
                "tracking_log_bytes": tracking_log.read_bytes() if tracking_log.exists() else None,
                "preview_frame_name": preview_frame.name if preview_frame.exists() else None,
                "preview_frame_bytes": preview_frame.read_bytes() if preview_frame.exists() else None,
                "summary_json_name": summary_path.name if summary_path.exists() else f"{input_path.stem}_summary.json",
                "summary_json_bytes": summary_path.read_bytes()
                if summary_path.exists()
                else json.dumps(summary, indent=2).encode("utf-8"),
                "heatmap_name": heatmap_path.name if heatmap_path is not None and heatmap_path.exists() else None,
                "heatmap_bytes": heatmap_path.read_bytes()
                if heatmap_path is not None and heatmap_path.exists()
                else None,
                "tracking_df": tracking_df,
            }
        except Exception as exc:
            st.error(
                "Processing failed while preparing the default sample or running the pipeline. "
                f"Details: {exc}"
            )

result_bundle = st.session_state.result_bundle
if result_bundle is not None:
    summary = result_bundle["summary"]
    st.success("Processing complete.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Frames Processed", f'{summary["frames_processed"]}')
    metric_cols[1].metric("Unique IDs", f'{summary["unique_tracked_subjects"]}')
    metric_cols[2].metric("Video FPS", f'{summary["video_fps"]}')
    metric_cols[3].metric("Processing FPS", f'{summary["average_processing_fps"]}')

    insight_cols = st.columns([1.5, 1.2])
    with insight_cols[0]:
        st.subheader("Tracked Output")
        if result_bundle["output_video_bytes"] is not None:
            st.video(result_bundle["output_video_bytes"])
        if result_bundle["preview_frame_bytes"] is not None:
            st.image(
                result_bundle["preview_frame_bytes"],
                caption="Annotated preview frame",
                use_container_width=True,
            )

    with insight_cols[1]:
        st.subheader("Tracking Summary")
        summary_table = pd.DataFrame(
            [
                {"Metric": "Detector", "Value": summary["detector"]},
                {"Metric": "Tracker", "Value": summary["tracking_algorithm"]},
                {"Metric": "Frames Processed", "Value": summary["frames_processed"]},
                {"Metric": "Frame Skip", "Value": summary.get("frame_skip", 1)},
                {"Metric": "Unique Tracked Subjects", "Value": summary["unique_tracked_subjects"]},
                {"Metric": "Processing Time (s)", "Value": summary["processing_seconds"]},
                {"Metric": "Average Processing FPS", "Value": summary["average_processing_fps"]},
            ]
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        tracking_df = result_bundle["tracking_df"]
        if tracking_df is not None and not tracking_df.empty:
            per_frame_counts = (
                tracking_df.groupby("frame_index")["track_id"]
                .nunique()
                .reset_index(name="active_tracks")
            )
            st.line_chart(per_frame_counts.set_index("frame_index"))

    if result_bundle["heatmap_bytes"] is not None:
        st.subheader("Movement Heatmap")
        st.image(result_bundle["heatmap_bytes"], caption="Movement heatmap", use_container_width=True)

    st.subheader("Downloads")
    download_cols = st.columns(4)
    if result_bundle["output_video_bytes"] is not None:
        download_cols[0].download_button(
            "Annotated Video",
            data=result_bundle["output_video_bytes"],
            file_name=result_bundle["output_video_name"],
            mime="video/mp4",
            use_container_width=True,
        )
    if result_bundle["tracking_log_bytes"] is not None:
        download_cols[1].download_button(
            "Tracking CSV",
            data=result_bundle["tracking_log_bytes"],
            file_name=result_bundle["tracking_log_name"],
            mime="text/csv",
            use_container_width=True,
        )
    download_cols[2].download_button(
        "Summary JSON",
        data=result_bundle["summary_json_bytes"],
        file_name=result_bundle["summary_json_name"],
        mime="application/json",
        use_container_width=True,
    )
    if result_bundle["preview_frame_bytes"] is not None:
        download_cols[3].download_button(
            "Preview Image",
            data=result_bundle["preview_frame_bytes"],
            file_name=result_bundle["preview_frame_name"],
            mime="image/jpeg",
            use_container_width=True,
        )

    if result_bundle["heatmap_bytes"] is not None:
        st.download_button(
            "Download Heatmap",
            data=result_bundle["heatmap_bytes"],
            file_name=result_bundle["heatmap_name"],
            mime="image/jpeg",
            use_container_width=True,
        )
