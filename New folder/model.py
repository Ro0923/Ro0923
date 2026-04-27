import torch
import torch.nn as nn

class ViolenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # RGB stream (3 channels)
        self.rgb_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Optical flow stream (2 channels)
        self.flow_cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Fusion layer (64 from RGB + 64 from Flow = 128)
        self.fc = nn.Linear(128, 2)

    def forward(self, frames, flows):
        """
        frames: [Batch, Time_RGB, Channels, Height, Width]
        flows:  [Batch, Time_Flow, Channels, Height, Width]
        """
        B, T_rgb, C_rgb, H, W = frames.shape
        _, T_flow, C_flow, _, _ = flows.shape

        # 1. Sync the time dimensions (Fixes your IndexError)
        # If flows has 15 and frames has 16, we take the first 15 of both.
        T = min(T_rgb, T_flow)
        frames = frames[:, :T]
        flows = flows[:, :T]

        # 2. Parallel Processing (Faster than 'for' loops)
        # We collapse Batch and Time into one dimension: [B*T, C, H, W]
        frames_flat = frames.reshape(B * T, C_rgb, H, W)
        flows_flat = flows.reshape(B * T, C_flow, H, W)

        # 3. Extract features through CNNs
        rgb_feats = self.rgb_cnn(frames_flat)   # [B*T, 64, 1, 1]
        flow_feats = self.flow_cnn(flows_flat) # [B*T, 64, 1, 1]

        # 4. Reshape back to separate Batch and Time
        rgb_feats = rgb_feats.view(B, T, 64)
        flow_feats = flow_feats.view(B, T, 64)

        # 5. Temporal Pooling (Mean across the time dimension)
        rgb_feat_mean = torch.mean(rgb_feats, dim=1)   # [B, 64]
        flow_feat_mean = torch.mean(flow_feats, dim=1) # [B, 64]

        # 6. Fusion and Classification
        fused = torch.cat((rgb_feat_mean, flow_feat_mean), dim=1) # [B, 128]
        return self.fc(fused)