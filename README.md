# 🔍 Real-Time Object Detection System

A simple **object detection** tool built with **YOLOv8** that can find objects in **pictures**, **videos**, and your **live webcam feed**. You just type in what you want to find (like "car" or "person") and it highlights those objects for you.

> Built for the **Computer Vision Lab Final** — BSAI 5th Semester, COMSATS University Islamabad.

---

## 🎯 What Does This Do?

This project can detect **80 different types of objects** (people, cars, animals, furniture, etc.) using the YOLOv8 AI model. It works in three ways:

| Script | What It Does |
|--------|-------------|
| `picture.py` | Detects objects in a **single image** and saves the result |
| `video.py` | Detects objects in a **video file** frame-by-frame and saves the output |
| `webcam.py` | Detects objects **live** from your webcam in real time |

**Key feature:** You can tell it *exactly what* to look for. For example, type `car person` and it will only highlight cars and people — ignoring everything else.

---

## 🛠️ Tech Stack

- **YOLOv8x** (Extra-Large) — The AI model that does the detection ([Ultralytics](https://github.com/ultralytics/ultralytics))
- **OpenCV** — Reads images/video, draws boxes, shows the output window
- **PyTorch** — Runs the AI model (uses your GPU if you have one, otherwise CPU)
- **COCO Dataset** — The model is pre-trained on 80 common object categories

---

## 📦 Setup

### 1. Install Python

Make sure you have **Python 3.8+** installed.

### 2. Install the required libraries

```bash
pip install ultralytics opencv-python torch
```

### 3. Download the model weights

The first time you run any script, it will **automatically download** the `yolov8x.pt` model file (~137 MB). Or you can place it manually in the project folder.

---

## 🚀 How to Use

### Detect objects in an image

1. Put your image in the project folder (or edit the `IMAGE_PATH` variable in `picture.py`)
2. Run:
   ```bash
   python picture.py
   ```
3. Type in the objects you want to find (e.g., `bench laptop bottle`)
4. It shows the result and saves it as `detected_<filename>.jpg`

### Detect objects in a video

1. Put your video file in the project folder (or edit the `VIDEO_PATH` variable in `video.py`)
2. Run:
   ```bash
   python video.py
   ```
3. Type in what to look for
4. Press **`t`** anytime to change what you're searching for
5. Press **`q`** to quit
6. The output is saved as `output_detected.mp4`

### Detect objects from your webcam

1. Run:
   ```bash
   python webcam.py
   ```
2. Type in what to look for
3. Press **`t`** to change targets, **`q`** to quit

---

## ⚡ Performance

Tested on a system with a CUDA-compatible GPU:

| Mode | FPS | Notes |
|------|-----|-------|
| Picture | N/A | ~1 second per image |
| Video | ~15 FPS | Processes every frame |
| Webcam | ~15 FPS | Real-time live feed |

> FPS is boosted using **FP16 half-precision** and a smaller input size (416px) — so it runs fast without losing much accuracy.

---

## 📂 Project Files

```
CV Terminal/
├── picture.py          # Image detection script
├── video.py            # Video detection script
├── webcam.py           # Webcam detection script
├── yolov8x.pt          # YOLOv8 Extra-Large model weights
├── yolov8m.pt          # YOLOv8 Medium model weights
├── yolov8n.pt          # YOLOv8 Nano model weights
├── image1.jpeg         # Sample test image
├── image2.jpeg         # Sample test image
├── video1.mp4          # Sample test video
├── video2.mp4          # Sample test video
├── video3.mp4          # Sample test video
└── README.md           # This file
```

---

## 🧠 How It Works (Simple Explanation)

1. **Load the model** — YOLOv8x is loaded onto your GPU (or CPU if no GPU)
2. **You type what to find** — e.g., `car person dog`
3. **It matches your words** to the 80 object categories it knows
4. **It scans the image/frame** — The AI looks at the entire image in one pass and finds all matching objects
5. **It draws boxes** — Green rectangles are drawn around detected objects with the label and confidence score
6. **It shows a summary** — At the end, you get a count of everything it found

---

## 🎨 What Objects Can It Detect?

It can detect **80 categories** from the COCO dataset, including:

| Category | Examples |
|----------|----------|
| People & Accessories | person, backpack, handbag, tie, suitcase |
| Vehicles | car, bus, truck, motorcycle, bicycle, airplane, boat, train |
| Animals | cat, dog, bird, horse, cow, elephant, bear, zebra, giraffe |
| Food | banana, apple, pizza, cake, sandwich, hot dog, donut |
| Household | chair, couch, bed, dining table, toilet, tv, laptop |
| Kitchen | bottle, cup, fork, knife, spoon, bowl, wine glass |
| Electronics | cell phone, keyboard, mouse, remote, microwave, oven |
| Outdoor | traffic light, fire hydrant, stop sign, bench, parking meter |
| Sports | sports ball, baseball bat, tennis racket, skateboard, surfboard, skis |

---

## ⚠️ Things to Know

- **GPU recommended** — It works on CPU too, but it'll be much slower
- **Image/video paths are hardcoded** — You'll need to edit the file path variables at the top of `picture.py` and `video.py` if your files have different names
- The confidence threshold is set to **0.25** — objects detected with less than 25% confidence are ignored

---

## 👥 Course Info

| | |
|---|---|
| **University** | COMSATS University Islamabad |
| **Department** | Computer Science |
| **Course** | Computer Vision |
| **Instructor** | Maheen Gul |
| **Semester** | 5th (Spring 2026) |

---

## 📄 License

This is an academic project built for learning purposes.
