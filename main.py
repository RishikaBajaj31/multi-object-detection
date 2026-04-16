from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg

from detection import DetectionConfig, YOLODetector
from tracking import DeepSORTTracker, TrackerConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end YOLOv8 + DeepSORT pipeline for multi-object detection and persistent tracking."
    )
    parser.add_argument("--input", required=True, help="Path to the input video.")
    parser.add_argument("--output-dir", default="outputs", help="Directory used for outputs.")
    parser.add_argument("--source-url", required=True, help="Original public source URL for the video.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--classes", default="0", help="Comma-separated class IDs to detect.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional quick test limit.")
    parser.add_argument("--log-every", type=int, default=25, help="Print progress every N frames.")
    parser.add_argument("--trajectory-length", type=int, default=30, help="Trajectory trail history.")
    parser.add_argument("--save-heatmap", action="store_true", help="Save final movement heatmap image.")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame to reduce runtime.")
    return parser.parse_args()


def create_csv_writer(path: Path):
    csv_file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame_index",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "speed_px_per_frame",
        ],
    )
    writer.writeheader()
    return csv_file, writer


def create_video_writer(output_path: Path, fps: float, width: int, height: int):
    """
    Prefer a browser-friendly MP4 codec for Streamlit playback.

    `avc1`/H.264 is much more likely to play inline in browsers than `mp4v`.
    If the local OpenCV build cannot create an H.264 writer, we fall back to `mp4v`
    so file generation still succeeds.
    """
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Could not create video writer for output: {output_path}")


def convert_video_for_web(video_path: Path):
    """
    Re-encode the generated video to H.264/yuv420p so browser players can render it.
    This keeps the rest of the pipeline unchanged and only normalizes the final output.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    web_ready_path = video_path.with_name(f"{video_path.stem}_web.mp4")

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(web_ready_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    web_ready_path.replace(video_path)


def run_pipeline(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [int(value.strip()) for value in args.classes.split(",") if value.strip()]
    detector = YOLODetector(
        DetectionConfig(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            image_size=args.imgsz,
            classes=classes,
        )
    )
    tracker = DeepSORTTracker(
        TrackerConfig(
            trajectory_length=args.trajectory_length,
        )
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = output_dir / f"{input_path.stem}_tracked.mp4"
    preview_frame_path = output_dir / f"{input_path.stem}_preview.jpg"
    tracking_log_path = output_dir / f"{input_path.stem}_tracking_log.csv"
    summary_path = output_dir / f"{input_path.stem}_summary.json"
    heatmap_path = output_dir / f"{input_path.stem}_heatmap.jpg"

    writer = create_video_writer(output_video_path, fps, width, height)
    csv_file, csv_writer = create_csv_writer(tracking_log_path)

    start_time = time.perf_counter()
    processed_frames = 0
    preview_saved = False
    last_annotated_frame = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            processed_frames += 1
            if args.max_frames and processed_frames > args.max_frames:
                break

            # Frame skipping is a practical CPU optimization for the live app/demo.
            # We still preserve the same pipeline, but only run detector + tracker on
            # every Nth frame to reduce total processing time.
            if args.frame_skip > 1 and ((processed_frames - 1) % args.frame_skip != 0):
                continue

            detections = detector.detect(frame)

            elapsed = time.perf_counter() - start_time
            processing_fps = processed_frames / elapsed if elapsed > 0 else 0.0
            records = tracker.update(frame, detections, frame_index=processed_frames - 1)
            annotated = tracker.draw_tracks(frame, records, fps_text=f"Processing FPS: {processing_fps:.2f}")

            for record in records:
                csv_writer.writerow(record.__dict__)

            writer.write(annotated)
            last_annotated_frame = annotated

            if not preview_saved:
                cv2.imwrite(str(preview_frame_path), annotated)
                preview_saved = True

            if processed_frames % args.log_every == 0:
                print(
                    f"Processed {processed_frames}/{total_frames if total_frames else '?'} frames | "
                    f"Detections: {len(detections)} | Unique IDs: {len(tracker.unique_ids)} | "
                    f"FPS: {processing_fps:.2f}",
                    flush=True,
                )
    finally:
        capture.release()
        writer.release()
        csv_file.close()

    convert_video_for_web(output_video_path)

    if args.save_heatmap and last_annotated_frame is not None:
        heatmap = tracker.build_heatmap_overlay(last_annotated_frame)
        cv2.imwrite(str(heatmap_path), heatmap)

    total_time = time.perf_counter() - start_time
    summary = {
        "title": "Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage",
        "detector": args.model,
        "tracking_algorithm": "DeepSORT",
        "input_video": str(input_path.resolve()),
        "source_url": args.source_url,
        "frames_processed": processed_frames,
        "source_frame_count": total_frames,
        "video_fps": fps,
        "processing_seconds": round(total_time, 2),
        "average_processing_fps": round(processed_frames / total_time, 2) if total_time else 0.0,
        "frame_skip": args.frame_skip,
        "unique_tracked_subjects": len(tracker.unique_ids),
        "output_video": str(output_video_path.resolve()),
        "tracking_log": str(tracking_log_path.resolve()),
        "preview_frame": str(preview_frame_path.resolve()),
        "tracked_classes": classes,
        "enhancements_enabled": [
            "trajectory_visualization",
            "unique_object_counting",
            "movement_tracking",
            "simple_speed_estimation",
            "optional_heatmap",
        ],
        "heatmap_path": str(heatmap_path.resolve()) if args.save_heatmap else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
