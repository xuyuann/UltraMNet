"""
NanoNet PyTorch model definition (45x60 input, ~7K params).

Uses Depthwise Separable Convolutions (DSConv) for Conv3 and Conv4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv2d(nn.Module):
    """Depthwise Separable Conv2d: depthwise (groups=C_in) + pointwise (1x1).
    No activation between dw and pw — matches MLX training model."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                           padding=padding, groups=in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class NanoNetPyTorch(nn.Module):
    """
    NanoNet: ~7K parameters.
    Conv1+2: standard convs. Conv3+4: depthwise separable.
    """

    def __init__(self, num_channels=2, num_classes=2, dropout_rate=0.3):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3_dw = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.conv3_pw = nn.Conv2d(16, 32, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4_dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.conv4_pw = nn.Conv2d(32, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(64, 32)
        self.fc_bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # MLX DSConv: pw(dw(x)) — no activation between dw and pw
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3_pw(self.conv3_dw(x)))))
        x = self.dropout(self.pool(F.relu(self.bn4(self.conv4_pw(self.conv4_dw(x))))))

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_nano_net_pt(model_path: str, device="cpu") -> NanoNetPyTorch:
    """Load the converted PyTorch NanoNet model."""
    model = NanoNetPyTorch()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    model = NanoNetPyTorch()
    x = torch.randn(2, 2, 45, 60)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
