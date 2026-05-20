import cv2
import torch
import time
from ultralytics import YOLO

VIDEO_PATH = "video2.mp4"

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def get_class_ids_from_text(text):
    words = text.replace(",", " ").split()
    class_ids = []
    for word in words:
        w = word.lower().strip()
        for i, c in enumerate(COCO_NAMES):
            if w == c or w in c or c in w:
                class_ids.append(i)
                break
    seen = set()
    unique_ids = []
    for cid in class_ids:
        if cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)
    return unique_ids

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video file " + VIDEO_PATH)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading YOLO on " + device + "...")
    model = YOLO("yolov8x.pt")
    model.to(device)

    target_input = input("What objects to find? (e.g., car person bottle): ")
    class_ids = get_class_ids_from_text(target_input)
    if not class_ids:
        print("No valid objects. Detecting all.")
        class_ids = None
    else:
        print("Searching for: " + str([COCO_NAMES[cid] for cid in class_ids]))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('output_detected.mp4', fourcc,
                                cap.get(cv2.CAP_PROP_FPS),
                                (int(cap.get(3)), int(cap.get(4))))

    print("Press 'q' to quit, 't' to change objects.")

    prev_time = time.time()
    frame_count = 0
    total_detections = 0
    detection_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        if class_ids is None:
            results = model(frame, conf=0.25, verbose=False, half=True, imgsz=416)
        else:
            results = model(frame, classes=class_ids, conf=0.25, verbose=False, half=True, imgsz=416)

        boxes = results[0].boxes
        det_count = 0
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = COCO_NAMES[cls] + ": " + str(round(conf, 2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                det_count += 1
                total_detections += 1
                cls_name = COCO_NAMES[cls]
                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1

        # FPS calculation
        frame_count += 1
        now = time.time()
        if frame_count % 10 == 0:
            fps = 10 / (now - prev_time)
            prev_time = now
        else:
            elapsed = now - prev_time
            fps = (frame_count % 10) / elapsed if elapsed > 0 else 0
        cv2.putText(frame, "FPS: " + str(round(fps, 1)), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        cv2.putText(frame, "Looking for: " + target_input + " | Found: " + str(det_count), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        out_video.write(frame)
        cv2.imshow("Video Object Search", frame)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        elif key == ord('t'):
            new_target = input("New objects: ")
            target_input = new_target
            class_ids = get_class_ids_from_text(target_input)
            if not class_ids:
                print("No valid objects. Detecting all.")
                class_ids = None
            else:
                print("Now searching for: " + str([COCO_NAMES[cid] for cid in class_ids]))

    out_video.release()
    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - (prev_time - (frame_count % 10) / max(fps, 1)) if frame_count > 0 else 0
    print("\n--- Session Summary ---")
    print("Total frames processed: " + str(frame_count))
    if frame_count > 0:
        print("Average FPS: " + str(round(fps, 1)))
    print("Total detections: " + str(total_detections))
    if detection_counts:
        print("Detections by class:")
        for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
            print("  " + cls_name + ": " + str(count))
    print("Output saved to: output_detected.mp4")


main()