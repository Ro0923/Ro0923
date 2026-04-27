import cv2
import torch
import numpy as np
import uuid
from torchvision import transforms
from model import ViolenceNet   # your model file

# -------------------------------
# CONFIG
# -------------------------------
CLIP_LEN = 16
VIOLENCE_THRESHOLD = 0.7
MIN_CONSECUTIVE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# TRANSFORMS
# -------------------------------
rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

flow_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# OPTICAL FLOW
# -------------------------------
def compute_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    flow_x = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    flow_y = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

    flow_img = np.stack([flow_x, flow_y], axis=2).astype(np.uint8)
    return flow_img

# -------------------------------
# CONFIDENCE ANALYSIS
# -------------------------------
def analyze_confidence(conf_list):
    if len(conf_list) == 0:
        return 0.0, False

    final_conf = float(np.mean(conf_list))

    consecutive = 0
    violence = False

    for p in conf_list:
        if p >= VIOLENCE_THRESHOLD:
            consecutive += 1
            if consecutive >= MIN_CONSECUTIVE:
                violence = True
        else:
            consecutive = 0

    return final_conf, violence

# -------------------------------
# VIDEO PREDICTION
# -------------------------------
def predict_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    clip_confidences = []
    detected_frame = None
    violence_counter = 0

    prev_frame = None

    while cap.isOpened():
        rgb_frames = []
        flow_frames = []
        raw_frames = []

        for _ in range(CLIP_LEN):
            ret, frame = cap.read()
            if not ret:
                break

            raw_frames.append(frame)

            rgb_frames.append(rgb_transform(frame))

            if prev_frame is None:
                flow_img = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.uint8)
            else:
                flow_img = compute_optical_flow(prev_frame, frame)

            flow_frames.append(flow_transform(flow_img))
            prev_frame = frame

        if len(rgb_frames) < CLIP_LEN:
            break

        rgb_tensor = torch.stack(rgb_frames).unsqueeze(0).to(DEVICE)
        flow_tensor = torch.stack(flow_frames).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(rgb_tensor, flow_tensor)
            prob = torch.softmax(outputs, dim=1)[0][1].item()

        clip_confidences.append(prob)

        # Consecutive logic
        if prob >= VIOLENCE_THRESHOLD:
            violence_counter += 1
            if violence_counter >= MIN_CONSECUTIVE and detected_frame is None:
                detected_frame = raw_frames[-1].copy()
        else:
            violence_counter = 0

    cap.release()

    final_confidence, violence_detected = analyze_confidence(
        clip_confidences
    )

    output_frame_path = None
    if detected_frame is not None:
        output_frame_path = f"detected_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(output_frame_path, detected_frame)

    return {
        "violence": violence_detected,
        "confidence": round(final_confidence, 4),
        "detected_frame": output_frame_path,
        "clip_confidences": clip_confidences
    }

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    model = ViolenceNet().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()

    video_path = "test_video.mp4"

    result = predict_video(video_path, model)

    print("Violence Detected :", result["violence"])
    print("Final Confidence :", result["confidence"])
    print("Detected Frame   :", result["detected_frame"])
