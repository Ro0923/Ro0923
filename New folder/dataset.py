import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RWF2000Dataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.samples = []
        self.labels = {"NonFight": 0, "Fight": 1}

        split_dir = os.path.join(root_dir, split)

        for label in ["Fight", "NonFight"]:
            label_dir = os.path.join(split_dir, label)
            for file in os.listdir(label_dir):
                if file.endswith("_frames.npy"):
                    frame_path = os.path.join(label_dir, file)
                    flow_path = frame_path.replace("_frames.npy", "_optflow.npy")
                    self.samples.append((frame_path, flow_path, self.labels[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, flow_path, label = self.samples[idx]

        frames = np.load(frame_path) / 255.0
        flow = np.load(flow_path)

        frames = torch.tensor(frames, dtype=torch.float32)
        flow = torch.tensor(flow, dtype=torch.float32)

        # [T, H, W, C] -> [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2)
        flow = flow.permute(0, 3, 1, 2)

        return frames, flow, label
