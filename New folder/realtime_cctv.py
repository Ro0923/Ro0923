import cv2
import torch
import numpy as np
from model import ViolenceNet
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViolenceNet().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)  # webcam / CCTV stream
prev_frame = None
frames, flows = [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(transform(frame))

    if prev_frame is None:
        flow = np.zeros((frame.shape[0], frame.shape[1], 2))
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

    flows.append(flow)
    prev_frame = frame

    if len(frames) == 16:
        frames_t = torch.stack(frames).unsqueeze(0).to(device)
        flows_t = torch.tensor(flows).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

        with torch.no_grad():
            prob = torch.softmax(model(frames_t, flows_t), dim=1)[0][1].item()

        if prob > 0.75:
            cv2.putText(frame, "VIOLENCE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frames, flows = [], []

    cv2.imshow("CCTV Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
