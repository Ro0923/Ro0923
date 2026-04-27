import cv2
import torch
from ultralytics import YOLO
import os

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/_2RYnSFPD_U_0.avi"
YOLO_WEIGHTS = "yolov8n.pt"
CONF_THRESH = 0.4
SAVE_DIR = "detected_frames"
MAX_SAVE = 2
# ---------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

yolo = YOLO(YOLO_WEIGHTS).to(device)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Video not opening"

frame_id = 0
saved_count = 0
best_conf = 0.0
best_frame = None
first_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = yolo(frame, conf=CONF_THRESH, verbose=False)[0]

    detected = False

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls.item())
            if cls != 0:
                continue

            detected = True
            conf = box.conf.item()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Save FIRST detected frame
            if first_frame is None:
                first_frame = frame.copy()
                cv2.imwrite(os.path.join(SAVE_DIR, "first_detected.jpg"), first_frame)
                saved_count += 1

            # Track BEST confidence frame
            if conf > best_conf:
                best_conf = conf
                best_frame = frame.copy()

    cv2.imshow("YOLO STEP-1 (Press Q to stop)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# -------- SAVE BEST FRAME --------
if best_frame is not None and saved_count < MAX_SAVE:
    cv2.imwrite(os.path.join(SAVE_DIR, "best_detected.jpg"), best_frame)

# -------- SHOW FINAL FRAMES --------
if first_frame is not None:
    cv2.imshow("First Detected Frame", first_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if best_frame is not None:
    cv2.imshow("Best Detected Frame", best_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
