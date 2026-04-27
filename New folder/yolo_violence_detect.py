import cv2
import torch
import numpy as np
from ultralytics import YOLO
from model import ViolenceNet
import os
from collections import deque

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/0xoSMgad_0.avi"
YOLO_WEIGHTS = "yolov8n.pt"
MODEL_PATH = "best_model.pt"

IMG_SIZE = 224
CLIP_LEN = 16
CONF_THRESH = 0.45   # violence confidence
SAVE_DIR = "final_detected"
# ---------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load models
yolo = YOLO(YOLO_WEIGHTS).to(device)

model = ViolenceNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Video not opening"

# Buffers
frame_buffer = deque(maxlen=CLIP_LEN)
gray_buffer = deque(maxlen=CLIP_LEN)
flow_buffer = deque(maxlen=CLIP_LEN - 1)

violence_detected = False
best_conf = 0.0
best_frame = None
frame_id = 0

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    display = frame.copy()

    # YOLO person detection
    results = yolo(frame, conf=0.4, verbose=False)[0]

    if results.boxes is None:
        cv2.imshow("Violence Detection", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for box in results.boxes:
        cls = int(box.cls.item())
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            continue

        person_resized = preprocess_frame(person)
        gray = cv2.cvtColor(
            cv2.resize(person, (IMG_SIZE, IMG_SIZE)),
            cv2.COLOR_BGR2GRAY
        )

        frame_buffer.append(person_resized)
        gray_buffer.append(gray)

        if len(gray_buffer) >= 2:
            flow = cv2.calcOpticalFlowFarneback(
                gray_buffer[-2], gray_buffer[-1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_buffer.append(flow)

        # When we have enough frames → predict
        if len(frame_buffer) == CLIP_LEN and len(flow_buffer) == CLIP_LEN - 1:
            frames = np.array(frame_buffer)
            flows = np.array(flow_buffer)

            flows = flows / (np.max(np.abs(flows)) + 1e-6)

            frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            flows = torch.tensor(flows, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(frames, flows)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, dim=1)

            conf = conf.item()
            pred = pred.item()

            label = "Fight" if pred == 1 else "NonFight"
            color = (0, 0, 255) if pred == 1 else (0, 255, 0)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Save ONLY strongest violence frame
            if pred == 1 and conf > CONF_THRESH and conf > best_conf:
                best_conf = conf
                best_frame = display.copy()
                cv2.imwrite(os.path.join(SAVE_DIR, "violence_detected.jpg"), best_frame)
                violence_detected = True

    cv2.imshow("Violence Detection", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- SHOW FINAL FRAME ----------
if violence_detected:
    print(f"🔥 Violence detected with confidence {best_conf:.2f}")
    cv2.imshow("FINAL VIOLENCE FRAME", best_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("✅ No violence detected in video")
