from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")
model.to(device)

def detect_persons(frame):
    results = model(frame, conf=0.4, classes=[0], device=device)
    boxes = []

    for r in results:
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

    return boxes
