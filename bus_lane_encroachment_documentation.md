# Bus Lane Encroachment Detection and Classification — Technical Documentation

## 1. Overview

This project builds a system that detects **unauthorized vehicles (encroachment)** — cars, motorcycles, and trucks — driving inside a dedicated **bus lane**, using CCTV footage. It combines three stages:

1. **Video ingestion & frame extraction** — pull frames out of raw CCTV video.
2. **Bus lane isolation** — mask/crop each frame down to only the bus lane region using a manually drawn boundary line.
3. **Auto-labeling with YOLOv8** — run a pretrained object detector on the isolated region to decide, per frame, whether an "encroaching" vehicle class is present.
4. **Lightweight classifier training** — fine-tune a MobileNetV2 image classifier on the isolated crops using YOLO's outputs as training labels, so that at inference time a much smaller/faster model can flag encroachment without running full YOLO detection each time.

The end goal is a deployable, low-latency binary classifier (`Sterile` vs `Encroachment`) that can run on each video frame in near real time.

---

## 2. Pipeline Architecture

```
Raw CCTV video (.mp4/.avi)
        │
        ▼
[Step 1] Frame extraction (1 fps, capped at 2000 frames/video)
        │
        ▼
[Step 2] Manual bus-lane boundary calibration (single straight line per sample frame)
        │
        ▼
[Step 3] ROI isolation (two variants: full-res for YOLO, 224×224 padded for MobileNet)
        │
        ▼
[Step 4] YOLOv8n detection on ROI → auto-label each frame (0 = Sterile, 1 = Encroachment)
        │
        ▼
[Step 5] Class balancing (cap Sterile at 5× Encroachment count) → balanced CSV
        │
        ▼
[Step 6] Train/validation split (80/20, stratified by label) → isolate + save crops to disk
        │
        ▼
[Step 7] MobileNetV2 transfer learning (2-stage: frozen backbone → fine-tune last 30 layers)
        │
        ▼
[Step 8] Evaluation (confusion matrix, ROC/AUC, misclassified examples)
        │
        ▼
[Step 9] Inference — single-frame prediction + annotated video generation
```

---

## 3. Step-by-Step Breakdown

### Step 1 — Video Download
- Uses `gdown` to pull a folder of CCTV footage from Google Drive into `/content/Footage CCTV/JPO AMARIS`.

### Step 2 — Frame Extraction (`FRAMES_DIR`)
- Iterates over every `.mp4`/`.MP4`/`.avi` file found.
- Extracts **1 frame per second** (`FRAME_INTERVAL_SEC = 1`), capped at **2000 frames per video**.
- Frames are read **sequentially** (`cap.read()`, no seeking) and written as JPEGs at quality 90.
- Output: a flat folder of frames named `{video_name}_frame_{frame_number:06d}.jpg`.

### Step 3 — Manual Lane Boundary Calibration
- A single sample frame (`frame_files[0]`) is loaded.
- Two points are hand-picked and hardcoded:
  - `point_top = (400, 0)` — where the lane boundary line meets the top edge of the frame.
  - `point_bottom = (220, h)` — where it meets the bottom edge.
- A straight line is drawn between these two points and visually inspected; the user adjusts the coordinates until the line "hugs" the physical bus lane barrier in the footage.
- **This calibration is done once, on one frame, and reused for every frame in every video for the rest of the pipeline.**

### Step 4 — ROI Isolation
Two isolation functions are defined, both built around the same polygon logic:

```python
roi_corners = [[0, 0], point_top, point_bottom, [0, h]]
mask = fillPoly(roi_corners) → binary mask
masked = bitwise_and(image, mask)
cropped = masked[:, 0:crop_w]   # crop_w = max(point_top.x, point_bottom.x)
```

- **`isolate_bus_lane()`** — used for the MobileNet training set. Additionally pad-resizes the cropped region into a fixed `224×224` canvas (letterboxed, aspect-ratio preserved, remaining space filled with black).
- **`isolate_bus_lane_fullres()`** — used only for YOLO auto-labeling. Keeps native resolution (no resize) so small/distant vehicles aren't lost to downscaling before detection.

Everything to the right of the line (outside the bus lane) is blacked out in both cases.

### Step 5 — YOLO Auto-Labeling
- Loads a pretrained **YOLOv8n** (COCO weights, `yolov8n.pt`), run on GPU if available.
- Defines `encroachment_classes = [2, 3, 7]` → COCO IDs for **car (2), motorcycle (3), truck (7)**. Bus (5) is deliberately excluded, since buses are allowed in the lane. Bicycles (1) and pedestrians (0) are also excluded.
- Processes original frames in batches of 16:
  1. Read frame from disk.
  2. Apply `isolate_bus_lane_fullres()`.
  3. Run YOLO inference on the batch (`imgsz=640`, `conf=0.20`, `half=True` on GPU).
  4. Label = `1` (Encroachment) if **any** detected box in the masked region belongs to `encroachment_classes`, else `0` (Sterile).
- Results saved to `yolo_annotations.csv` (`frame_path`, `label`).
- **Class balancing:** Sterile frames are randomly subsampled down to `5 × len(encroachment frames)` (or fewer if not enough Sterile frames exist), shuffled, and saved to `yolo_annotations_balanced.csv`.

### Step 6 — Dataset Preparation
- Reads `yolo_annotations_balanced.csv`.
- Splits into train (80%) / validation (20%) using `train_test_split(..., stratify=label, random_state=42)` — **splitting is done per-frame, at random, not per-video or per-time-block.**
- For every row, re-reads the *original* frame from disk, applies `isolate_bus_lane()` (224×224 padded crop), and writes it into:
  ```
  mobilenet_data/
    train/
      Sterile/
      Encroachment/
    validation/
      Sterile/
      Encroachment/
  ```
- A sanity-check image is displayed to confirm the saved files are isolated crops, not full frames.

### Step 7 — MobileNetV2 Training
- **Input size:** 224×224×3 (`TARGET_SIZE`).
- **Augmentation (train only):** rotation ±15°, width/height shift ±15%, zoom ±15%, brightness [0.6, 1.3], no horizontal flip (sensible — flipping would reverse the lane's real-world left/right orientation).
- **Architecture:** `MobileNetV2(weights='imagenet', include_top=False)` → `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(1, activation='sigmoid')`.
- **Two-stage transfer learning:**
  - *Stage 1:* backbone fully frozen, train only the classification head. Adam, lr = 1e-4, up to 40 epochs with early stopping (`patience=6` on `val_loss`).
  - *Stage 2 (fine-tuning):* unfreeze only the **last 30 layers** of the backbone, recompile with a much lower learning rate (1e-5), train up to 20 more epochs with its own early stopping/checkpointing.
- **Class imbalance handling:** in addition to the 5:1 subsampling done earlier, `class_weight` is computed from the actual train split and passed to `model.fit()` — i.e., imbalance is corrected **twice** (once by subsampling, once by reweighting).
- **Callbacks:** `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` (best weights by `val_loss`) for both stages.
- **Metrics tracked:** accuracy, recall, precision (F1 computed manually after `model.evaluate()`).
- Final weights saved to `mobilenetv2_bus_lane.weights.h5` and full model to `mobilenetv2_bus_lane_classifier.h5`.

### Step 8 — Evaluation
- Combined training curves (loss/accuracy/precision/recall) across both stages, with a vertical marker at the fine-tuning boundary.
- Confusion matrix and `classification_report` on the validation set (class-name aware via `val_gen.class_indices`).
- ROC curve + AUC.
- Histogram of predicted probabilities, split by true class, to visualize decision-boundary separation.
- Grid of misclassified validation examples for manual inspection.

### Step 9 — Inference
- `predict_frame()`: isolates the ROI from a raw frame → `preprocess_input` → MobileNetV2 forward pass → returns `(label, probability, roi)`.
- Batch visual check on 8 sample frames (original + line overlay on top row, isolated crop + prediction on bottom row).
- `generate_annotated_video()`: runs inference over a chosen time window of a source video, draws the lane boundary line and a color-coded status banner (red = Encroachment, green = Sterile) onto each frame, and writes an annotated `.mp4`. Supports `process_every_nth_frame` to skip frames for speed (reusing the last prediction in between).

---

## 4. Key Design Decisions

| Decision | Rationale |
|---|---|
| Mask + crop to bus lane before classification | Reduces background clutter/distraction so the classifier only "sees" the relevant region |
| Two-resolution ROI (full-res for YOLO, 224×224 for MobileNet) | Avoids losing small/distant vehicles to downscaling during auto-labeling, while keeping MobileNet's expected input size |
| YOLOv8n for auto-labeling | Avoids manual annotation of thousands of frames; leverages an existing pretrained detector |
| MobileNetV2 for deployment | Much smaller/faster than running full YOLO detection every frame; suited for edge/real-time deployment |
| 5:1 Sterile:Encroachment cap + class weights | Encroachment events are naturally rare; both techniques combined to prevent the classifier from trivially predicting "Sterile" always |
| Two-stage transfer learning (freeze → fine-tune last 30 layers) | Standard practice to avoid destroying pretrained ImageNet features early, then adapt higher-level features to the domain |
| No horizontal flip augmentation | Preserves the real-world left/right meaning of "inside vs outside the lane" |

---

## 5. Known Limitations / Flaws

### 5.1 Label circularity (distillation, not ground truth)
The training labels are **YOLOv8n's own predictions** on the masked ROI — not human-verified ground truth. MobileNetV2 is therefore trained to imitate YOLO's behavior on these crops, including any of YOLO's systematic errors (missed small/distant vehicles, false positives from shadows/reflections at the low `conf=0.20` threshold). Reported accuracy/F1 measures **agreement with YOLO**, not real-world correctness. This is a valid *distillation* strategy for building a faster model, but it should not be presented as an independently validated detector without spot-checking a sample of labels against human review.

### 5.2 Single static calibration line reused globally
`point_top`/`point_bottom` is hand-tuned on **one frame from one video** and then applied to every frame of every video. Any difference in camera angle, zoom, position, or a curved (non-linear) real-world lane boundary will misalign the mask for other footage — silently corrupting both labels and training crops for those videos.

### 5.3 Train/validation leakage
Frames are extracted every second from continuous video and then split **randomly per-frame** (`train_test_split`) rather than by video or contiguous time block. Consecutive frames of the same vehicle/event are highly correlated, so near-duplicate frames can end up on both sides of the split, inflating validation metrics.

### 5.4 Domain shift from aggressive masking
Blacking out everything outside the lane and letterbox-padding to 224×224 creates image statistics unlike the natural photos MobileNetV2 was pretrained on. The visible triangular region also sits in a roughly consistent position/shape across frames, creating a risk that the model partially keys off mask geometry/position rather than the vehicle itself.

### 5.5 No temporal reasoning
Classification is done independently per frame. A vehicle briefly crossing the boundary (e.g., turning, avoiding an obstacle) is flagged identically to one that stops or drives at length in the lane. There is no object tracking, dwell-time threshold, or event-level aggregation — features generally needed for real enforcement/evidence use cases.

### 5.6 Balanced evaluation set doesn't reflect real-world prevalence
Validation metrics are computed on the artificially balanced (5:1) subset, not the true real-world class distribution (encroachments are likely far rarer). Real-world precision will probably be lower than reported.

### 5.7 Class definition assumptions
`encroachment_classes = [2, 3, 7]` (car, motorcycle, truck) excludes bicycles (class 1) and pedestrians (class 0). Confirm this matches the intended legal/policy definition of "encroachment" for the specific bus lane being monitored.

---

## 6. Suggested Improvements

1. **Human-verify a sample of YOLO auto-labels** (e.g., a few hundred frames) before trusting them as training ground truth; report agreement rate between YOLO and human review.
2. **Per-video/per-camera line calibration**, or better, an automated or semi-automated lane-boundary detection step (e.g., lane markings, homography from camera calibration) instead of one hardcoded straight line.
3. **Split train/val by video or by time block**, not by random frame, to prevent leakage.
4. **Add object tracking** (e.g., ByteTrack/DeepSORT on top of YOLO) to convert frame-level flags into vehicle-level encroachment *events*, with a minimum-duration threshold before flagging a violation.
5. **Evaluate on a held-out, naturally-distributed (unbalanced) test set** in addition to the balanced training/validation sets, to get a realistic precision/recall estimate.
6. **Sanity-check what the classifier is attending to** (e.g., Grad-CAM) to confirm it's responding to vehicles and not mask/crop geometry.
7. **Raise or tune the YOLO confidence threshold** used for auto-labeling, or use a larger/more accurate YOLO variant (e.g., `yolov8s`/`yolov8m`) for label generation specifically, even if a smaller model is used for final deployment.

---

## 7. File/Artifact Reference

| File | Produced by | Purpose |
|---|---|---|
| `video_frames/` | Step 2 | Raw extracted frames (1 fps) |
| `isolated_bus_lane_frames/` | Step 3 (demo cell) | Example 224×224 isolated crops |
| `yolo_annotations.csv` | Step 5 | Full YOLO-derived labels (all frames) |
| `yolo_annotations_balanced.csv` | Step 5 | 5:1 Sterile:Encroachment balanced label set |
| `mobilenet_data/train/`, `mobilenet_data/validation/` | Step 6 | Isolated crops organized by class for `ImageDataGenerator` |
| `best_stage1.weights.h5` | Step 7 (Stage 1) | Best frozen-backbone checkpoint |
| `best_finetuned.weights.h5` | Step 7 (Stage 2) | Best fine-tuned checkpoint |
| `mobilenetv2_bus_lane.weights.h5` | Step 7 | Final model weights |
| `mobilenetv2_bus_lane_classifier.h5` | Step 7 | Final full saved model |
| `annotated_output.mp4` | Step 9 | Example annotated output video with live predictions overlaid |

---

## 8. Summary

The notebook implements a **two-model pipeline**: a pretrained YOLOv8n detector generates weak/auto labels on a manually-masked bus-lane region, and a MobileNetV2 classifier is then trained (via two-stage transfer learning) to reproduce those labels in a lighter, faster form suitable for near-real-time deployment. The overall structure is sound as an engineering approach to building a fast classifier from an existing detector, but its real-world reliability is bounded by (a) the quality of YOLO's auto-labels, which are unverified, and (b) the accuracy of a single, manually-calibrated straight-line lane boundary applied uniformly across all footage. Neither issue is a bug in the code — both are design choices worth revisiting before treating this as a production-ready enforcement tool.
