# Technical Report

## 1. Overview

This project implements a multi-object detection and persistent ID tracking pipeline for public sports or event footage. The final system uses **YOLOv8** for object detection and **DeepSORT** for multi-object tracking. The main objective is to detect all relevant people in a video and assign each one a unique and persistent ID across frames, even when practical challenges such as occlusion, overlap, motion blur, and camera movement are present. For the assignment-aligned example, the project uses a public **marathon / race event** video from Pixabay: `https://pixabay.com/videos/marathon-marathon-runners-running-15741/`.

## 2. Model Used

The detector used in this project is **YOLOv8**, specifically the lightweight `yolov8n.pt` checkpoint by default. YOLOv8 was selected because it offers a strong balance between speed and detection quality, is easy to deploy, and integrates cleanly into a Python/OpenCV pipeline. It is also well suited for frame-by-frame video inference.

The tracking algorithm used is **DeepSORT**. DeepSORT extends SORT by adding an appearance embedding model on top of the motion-based tracking pipeline. This makes it better suited for keeping identities stable in crowded scenes or when objects move close to one another.

## 3. Why YOLOv8 + DeepSORT Was Chosen

This combination was selected for three main reasons:

1. **Practicality**: both models are widely used, well documented, and straightforward to integrate in a compact project.
2. **Accuracy vs. simplicity**: YOLOv8 provides reliable detections, while DeepSORT provides stronger identity consistency than naive IOU-only or centroid-based tracking.
3. **Better ID stability**: DeepSORT combines motion and appearance information, which helps reduce identity switching when subjects overlap or look similar.

## 4. How ID Consistency Is Maintained

ID consistency is maintained by separating the pipeline into detection and tracking stages.

1. YOLOv8 detects objects in each frame and outputs bounding boxes with confidence scores.
2. These detections are passed to DeepSORT.
3. DeepSORT predicts where existing tracks should appear in the next frame using motion history.
4. It then compares current detections to existing tracks using both:
   - motion similarity
   - appearance similarity
5. If a detection matches an existing track, the same ID is retained.
6. If a detection does not match any existing track strongly enough, a new track ID is created.

This design reduces ID switching compared to simple matching strategies. The tracker also keeps tracks alive for a short period during missed detections, which helps when brief occlusion happens.

## 5. Real-World Challenges

### Occlusion

When a person is partially hidden behind another person or object, DeepSORT can often preserve the same ID using track history and motion prediction. This helps avoid immediate ID reassignment during short occlusions.

### Motion Blur

Blur can reduce detector confidence temporarily. When detections recover, DeepSORT may reconnect the track if the motion and appearance cues remain consistent.

### Camera Motion

In event footage, the camera may pan or zoom. DeepSORT's temporal association helps preserve identity better than frame-independent detection alone, although very aggressive camera motion may still degrade performance.

### Similar-Looking Subjects

This is where DeepSORT is especially helpful. Appearance embeddings provide additional information beyond box overlap, making it more robust when nearby subjects wear similar clothes or move close together.

## 6. Enhancements Implemented

The upgraded system includes several optional enhancements that strengthen the submission:

- **Trajectory visualization**: path lines show how each tracked subject moves over time
- **Unique object counting**: total number of unique tracked IDs is displayed
- **Movement tracking**: center positions are stored for every track
- **Simple speed estimation**: speed is estimated in pixels per frame from recent movement history
- **Optional heatmap generation**: movement intensity can be saved as a heatmap image

These additions make the output easier to interpret and demonstrate that the tracker is maintaining temporal consistency.

## 7. Failure Cases Observed

The pipeline is practical, but not perfect. Some failure cases still exist:

- long or complete occlusions can cause a new ID to be assigned
- dense crowds can increase identity switching
- strong blur or low lighting can reduce detector quality
- speed estimation is only approximate because the project does not use camera calibration or world coordinates

## 8. Possible Improvements

Several improvements could further strengthen the project:

- use a larger YOLOv8 model such as `yolov8s` or `yolov8m`
- tune DeepSORT hyperparameters for the chosen video domain
- add team clustering or jersey color grouping for sports footage
- use homography or camera calibration for real-world speed estimation
- add quantitative metrics such as MOTA, IDF1, or track fragmentation statistics

## 9. Conclusion

The final upgraded project satisfies the assignment requirements by providing:

- modular code structure
- YOLOv8-based detection
- DeepSORT-based persistent ID tracking
- annotated output video
- tracking log and summary outputs
- README and technical documentation

The selected public source is a marathon video with multiple moving runners, which matches the assignment's preferred sports/race categories and is suitable for demonstrating persistent multi-object tracking.

The resulting system is clear, practical, and suitable for both local execution and Google Colab.
