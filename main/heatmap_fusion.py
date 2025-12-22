import torch
import torch.nn as nn

class HeatmapFusion(nn.Module):
    def __init__(self, num_keypoints, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

    def forward(self, x, heatmap):
        """
        x: feature map (B, C, H, W)
        heatmap: predicted heatmap (B, K, H, W)
        """
        fused = x + heatmap  # Simple additive fusion, could be concat + conv also
        out = self.conv1(fused)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out
