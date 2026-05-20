# 🔍 Interactive Real-Time Object Detection Suite

An interactive, real-time object detection suite powered by **YOLOv8** and **OpenCV**. This system allows users to dynamically target and detect objects across static images, video files, and live webcam feeds using custom query-based filtering.

---

## 🎯 Key Features

This project utilizes the state-of-the-art YOLOv8 model trained on the COCO dataset to identify up to **80 distinct object categories** (including vehicles, pedestrians, electronics, and household items).

* **Multi-Source Support:** Process static images, pre-recorded video files, or a live webcam stream.
* **Dynamic Target Filtering:** Filter detections in real-time by entering target labels (e.g., `car person`). The system ignores all other classes, optimizing visual focus and performance.
* **Interactive Querying:** Change target classes mid-stream in webcam and video modes by pressing a hotkey.
* **Hardware Optimized:** Automatic GPU/CUDA acceleration detection with FP16 half-precision execution for boosted framerates.

| Module | Purpose | Output |
| :--- | :--- | :--- |
| `picture.py` | Inference on static images | Saves annotated image (`detected_<filename>.jpg`) |
| `video.py` | Processing video files frame-by-frame | Saves optimized MP4 video (`output_detected.mp4`) |
| `webcam.py` | Live webcam feed inference | Interactive real-time window display |

---

## 🛠️ Tech Stack & Dependencies

* **YOLOv8** (Ultralytics) — Deep learning model architecture for object detection.
* **OpenCV** — Real-time computer vision processing, video I/O, and image annotation.
* **PyTorch** — Deep learning framework support (with CUDA GPU acceleration).

---

## 📦 Setup & Installation

### 1. Install Dependencies
Ensure you have **Python 3.8+** installed, then install the required libraries:

```bash
pip install ultralytics opencv-python torch
```

### 2. Model Weights
On the initial run, the application will **automatically download** the required YOLOv8 weights (e.g., `yolov8x.pt` ~137 MB) from Ultralytics. Alternatively, you can place your own `.pt` model files directly in the root directory.

---

## 🚀 How to Use

### Dynamic Target Input
When running any of the scripts, the terminal will prompt you to enter target objects:
* Type space-separated object names (e.g., `car person handbag`).
* Press **Enter** to start.
* To detect all 80 COCO classes, simply press **Enter** without typing any keywords.

---

### 1. Image Inference
Place your image in the project root directory (or update the `IMAGE_PATH` variable in `picture.py`) and run:
```bash
python picture.py
```
* **Output:** The annotated image is saved in the root directory.

---

### 2. Video File Processing
Place your video in the project root directory (or update the `VIDEO_PATH` variable in `video.py`) and run:
```bash
python video.py
```
* **Interactive Keys:**
  * Press **`t`** at any point to update the target query live from the terminal.
  * Press **`q`** to safely terminate processing early.
* **Output:** Saves the processed video as `output_detected.mp4`.

---

### 3. Live Webcam Detection
Initiate a real-time detection session using your primary system camera:
```bash
python webcam.py
```
* **Interactive Keys:**
  * Press **`t`** to change the target objects list dynamically.
  * Press **`q`** to exit the camera feed.

---

## ⚡ Performance Optimization

Benchmark results on a CUDA-enabled GPU:

| Mode | Performance | Notes |
| :--- | :--- | :--- |
| **Static Image** | ~1.0s / image | Full-resolution forward pass |
| **Video Processing** | ~15-20 FPS | Batched frame inference |
| **Webcam Stream** | ~15-20 FPS | Live feed with FP16 half-precision enabled |

> [!TIP]
> The scripts automatically leverage CUDA if a compatible NVIDIA GPU is detected, applying **FP16 half-precision** and input scaling (416px) to maximize real-time throughput.

---

## 📂 Project Directory Structure

```
CV Terminal/
├── picture.py          # Static image detection module
├── video.py            # Video file processing module
├── webcam.py           # Live webcam stream detection module
├── .gitignore          # Git exclusion rules (ignores model weights and caches)
├── image1.jpeg         # Sample test image
├── image2.jpeg         # Sample test image
├── video1.mp4          # Sample test video
├── video2.mp4          # Sample test video
├── video3.mp4          # Sample test video
└── README.md           # Project Documentation
```

---

## 🎨 Supported COCO Object Classes

The underlying model supports **80 object categories**, classified into:

* **People:** `person`, `backpack`, `handbag`, `tie`, `suitcase`
* **Vehicles:** `car`, `bus`, `truck`, `motorcycle`, `bicycle`, `airplane`, `boat`, `train`
* **Animals:** `cat`, `dog`, `bird`, `horse`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
* **Food Items:** `banana`, `apple`, `pizza`, `cake`, `sandwich`, `hot dog`, `donut`
* **Electronics:** `cell phone`, `keyboard`, `mouse`, `remote`, `microwave`, `oven`
* **Indoor/Home:** `chair`, `couch`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `bottle`, `cup`

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
