"""
UltraMNet PyTorch model definition (45x60 input, ~27K params).

This is the PyTorch equivalent of the MLX UltraMNet model.
Architecture matches exactly: Conv1→Conv2→Conv3→Conv4→GAP→FC1→FC2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UltraMNetPyTorch(nn.Module):
    """
    Ultra Micro CNN for hand detection.
    Input: 45x60x2 (depth + amplitude), channel-first
    Output: 2-class logits (no-hand / hand)
    """

    def __init__(self, num_channels=2, num_classes=2, dropout_rate=0.3):
        super().__init__()

        # Conv blocks
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Classifier
        self.fc1 = nn.Linear(64, 32)
        self.fc_bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → B×8×22×30
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → B×16×11×15
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # → B×32×5×7
        x = self.dropout(self.pool(F.relu(self.bn4(self.conv4(x)))))  # → B×64×2×3

        # Global Average Pooling → B×64
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_ultramnet_pt(model_path: str, device="cpu") -> UltraMNetPyTorch:
    """Load the converted PyTorch UltraMNet model."""
    model = UltraMNetPyTorch()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    # Test
    model = UltraMNetPyTorch()
    x = torch.randn(2, 2, 45, 60)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
