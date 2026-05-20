import cv2
import torch
import time
from ultralytics import YOLO

IMAGE_PATH = "image2.jpeg"  

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
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: cannot load image " + IMAGE_PATH)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading YOLO on " + device)
    model = YOLO("yolov8x.pt")
    model.to(device)

    target_input = input("What objects to find? (e.g., car person bottle): ")
    class_ids = get_class_ids_from_text(target_input)

    start_time = time.time()
    if not class_ids:
        print("No valid objects. Detecting all objects.")
        results = model(img, conf=0.25, verbose=False)
    else:
        print("Searching for: " + str([COCO_NAMES[cid] for cid in class_ids]))
        results = model(img, classes=class_ids, conf=0.25, verbose=False)
    inference_time = time.time() - start_time

    boxes = results[0].boxes
    detections = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = COCO_NAMES[cls] + ": " + str(round(conf, 2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            detections.append((COCO_NAMES[cls], conf))

    print("\n--- Detection Summary ---")
    print("Inference time: " + str(round(inference_time, 3)) + "s")
    print("Found " + str(len(detections)) + " object(s):")
    detection_counts = {}
    for obj, conf in detections:
        print("  - " + obj + " (" + str(round(conf, 2)) + ")")
        detection_counts[obj] = detection_counts.get(obj, 0) + 1
    if detection_counts:
        print("Detections by class:")
        for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
            print("  " + cls_name + ": " + str(count))

    out_path = "detected_" + IMAGE_PATH.split("/")[-1].split("\\")[-1]
    cv2.imwrite(out_path, img)
    print("Saved result to " + out_path)

    cv2.imshow("Image Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()