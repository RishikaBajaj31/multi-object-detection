# Multi-Object Detection and Persistent ID Tracking

This project upgrades a basic YOLO-based detection system into a complete **YOLOv8 + DeepSORT** pipeline for public sports or event footage. It detects people in each frame, assigns stable IDs over time, visualizes trajectories, estimates simple movement speed, and saves annotated outputs for submission.

## Project Overview

The pipeline is split into three clean modules:

- `detection.py`: runs YOLOv8 object detection
- `tracking.py`: runs DeepSORT tracking and ID management
- `main.py`: reads the video, processes each frame, draws annotations, and saves outputs

The system is designed for assignment-style use on local machines or Google Colab, with a focus on correctness, readability, and practical tracking behavior.

## Why YOLOv8 + DeepSORT

- **YOLOv8** provides strong real-time object detection with simple Python integration.
- **DeepSORT** improves ID consistency by combining:
  - motion prediction using a Kalman filter
  - data association across frames
  - appearance embeddings, which help separate similar-looking objects

This makes DeepSORT a strong choice when subjects overlap, become partially occluded, or move quickly.

## Features Implemented

- YOLOv8 detection per frame
- DeepSORT persistent multi-object tracking
- unique and stable ID labels such as `ID:3`
- trajectory trails
- unique object counting
- movement history storage
- simple speed estimation in pixels per frame
- FPS display
- optional movement heatmap export
- CSV tracking log
- JSON summary report
- annotated MP4 output

## File Structure

- `detection.py`
- `tracking.py`
- `main.py`
- `requirements.txt`
- `README.md`
- `technical_report.md`
- `outputs/`

## Installation

### Local machine

```bash
pip install -r requirements.txt
```

### Google Colab

```bash
!pip install -r requirements.txt
```

If `yolov8n.pt` is not present, Ultralytics downloads it automatically on first run.

## How To Run

### Windows PowerShell

```powershell
python main.py --input "C:\Users\rishi\Downloads\marathon-15741.mp4" --source-url "https://pixabay.com/videos/marathon-marathon-runners-running-15741/" --output-dir outputs --model yolov8n.pt --save-heatmap
```

### Linux / macOS / Colab

```bash
python main.py \
  --input "/content/marathon-15741.mp4" \
  --source-url "https://pixabay.com/videos/marathon-marathon-runners-running-15741/" \
  --output-dir outputs \
  --model yolov8n.pt \
  --save-heatmap
```

### Streamlit UI

```bash
streamlit run app.py
```

The Streamlit interface lets a user:

- upload a video optionally
- click `Run Tracking`
- preview and download generated outputs

If no video is uploaded, the app looks for a bundled local marathon sample instead of trying to scrape a remote video site at runtime.
For the most reliable local or deployed behavior, place the default sample video at `sample_videos/default_marathon.mp4`. The UI will use that hidden local sample first and otherwise prompt the user to upload a video manually.

### Quick test on fewer frames

```bash
python main.py --input "path/to/video.mp4" --source-url "PUBLIC_VIDEO_URL" --output-dir outputs --max-frames 100
```

## Input Format

- any public sports or public-event video
- recommended footage contains multiple moving people or players
- default configuration tracks class `0` which corresponds to `person`

## Output Format

The script saves:

- `outputs/<video_name>_tracked.mp4`
- `outputs/<video_name>_tracking_log.csv`
- `outputs/<video_name>_preview.jpg`
- `outputs/<video_name>_summary.json`
- `outputs/<video_name>_heatmap.jpg` if `--save-heatmap` is enabled

## Assumptions

- the input video is publicly accessible and the source URL is known
- people are the primary class of interest
- speed is estimated in **pixels per frame**, not real-world units
- camera calibration is not available

## Handling Real-World Challenges

- **Occlusion**: DeepSORT keeps tracks alive for short periods using track history and motion prediction
- **Motion blur**: YOLOv8 remains reasonably robust, and DeepSORT can reconnect tracks when detections recover
- **Camera motion / zoom**: motion prediction and temporal association reduce sudden ID loss
- **Similar objects**: appearance embeddings help DeepSORT distinguish nearby people better than pure box matching

## Limitations

- speed estimation is approximate because there is no camera calibration
- full occlusion for long durations can still cause new IDs
- very dense crowds may still produce ID switches
- performance on CPU can be slower for high-resolution videos

## Dependencies

- Python 3.10+
- OpenCV
- PyTorch
- Ultralytics
- deep-sort-realtime
- NumPy

## Public Video Used In This Project

- recommended local filename: `C:\Users\rishi\Downloads\marathon-15741.mp4`
- public source link: `https://pixabay.com/videos/marathon-marathon-runners-running-15741/`

This is a publicly accessible **marathon / race event** video, which matches the assignment's preferred categories more closely than a generic public event clip.

## Model / Tracker Choice Summary

- detector: `YOLOv8n`
- tracker: `DeepSORT`

This combination was chosen because it is simple to run, easy to explain, and strong enough for practical multi-object tracking assignments without over-engineering the solution.
