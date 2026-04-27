import torch
import numpy as np
from model import ViolenceNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ViolenceNet().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

def predict(frames_path, flow_path):
    frames = np.load(frames_path) / 255.0
    flow = np.load(flow_path)

    flow = flow / (np.max(np.abs(flow)) + 1e-6)

    frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
    flow = torch.tensor(flow, dtype=torch.float32).permute(0, 3, 1, 2)

    frames = frames.unsqueeze(0).to(device)
    flow = flow.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frames, flow)
        pred = torch.argmax(output, dim=1).item()

    return "Fight" if pred == 1 else "NonFight"


# 🔹 Example test
import os
import random

if __name__ == "__main__":
    folder = "Processed_Clean/val/Fight"
    files = [f for f in os.listdir(folder) if f.endswith("_frames.npy")]
    sample = random.choice(files)

    frames_path = os.path.join(folder, sample)
    flow_path = frames_path.replace("_frames.npy", "_optflow.npy")

    print("Testing file:", sample)
    label = predict(frames_path, flow_path)
    print("Prediction:", label)

