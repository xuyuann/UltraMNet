"""
NanoNet - Depthwise Separable CNN for hand detection (45x60 input).
Parameters: ~7K (74% reduction from UltraMNet's 27K).

Architecture is identical to UltraMNet except Conv3 and Conv4 are replaced
with Depthwise Separable Convolutions (DSConv), which split a standard k×k
conv into a depthwise pass (per-channel spatial filtering) followed by a
pointwise 1×1 conv (cross-channel mixing). Feature map shapes and channel
widths are unchanged, so the rest of the pipeline (GAP, FC) is unmodified.

Parameter breakdown:
  Conv1  (2→8,  k=3):  152      (standard, first layer has only 2 channels)
  BN1:                   16
  Conv2  (8→16, k=3):  1168     (standard, still small)
  BN2:                   32
  DSConv3(16→32, k=3):  704     (DW=160 + PW=544, was 4640 standard)
  BN3:                   64
  DSConv4(32→64, k=3): 2432     (DW=320 + PW=2112, was 18496 standard)
  BN4:                  128
  FC1    (64→32):      2080
  FC_BN1:                64
  FC2    (32→2):         66
  ─────────────────────────
  TOTAL:               6906  (~7K, vs 26906 in UltraMNet)
"""

import mlx.core as mx
import mlx.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from UltraMNet.config import NUM_CHANNELS, NUM_CLASSES, DROPOUT_RATE


class DSConv2d(nn.Module):
    """Depthwise Separable Conv2d: depthwise (groups=C_in) + pointwise (1×1)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding, groups=in_channels
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x):
        return self.pw(self.dw(x))


class HandDetectorNanoNet(nn.Module):
    """
    NanoNet: ~7K parameters.

    Conv1+2 remain standard (their parameter counts are small).
    Conv3+4 use DSConv (depthwise+pointwise), cutting those two layers from
    23,136 params down to 3,136 — the primary source of savings.
    """

    def __init__(self, num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES,
                 dropout_rate=DROPOUT_RATE):
        super().__init__()

        # Standard convolutions (small, kept as-is)
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm(16)

        # Depthwise Separable convolutions (large, replaced to save params)
        self.conv3 = DSConv2d(16, 32)
        self.bn3   = nn.BatchNorm(32)

        self.conv4 = DSConv2d(32, 64)
        self.bn4   = nn.BatchNorm(64)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Classifier (unchanged from UltraMNet)
        self.fc1    = nn.Linear(64, 32)
        self.fc_bn1 = nn.BatchNorm(32)
        self.fc2    = nn.Linear(32, num_classes)

    def __call__(self, x):
        # x: (B, H, W, C)  — MLX channel-last format
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))   # → 22×30×8
        x = self.pool(nn.relu(self.bn2(self.conv2(x))))   # → 11×15×16
        x = self.pool(nn.relu(self.bn3(self.conv3(x))))   # → 5×7×32
        x = self.dropout(self.pool(nn.relu(self.bn4(self.conv4(x)))))  # → 2×3×64

        x = mx.mean(x, axis=(1, 2))          # Global Average Pooling → (B, 64)

        x = nn.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model_nano():
    """Create and return the NanoNet model."""
    return HandDetectorNanoNet()


def count_parameters():
    """Print detailed parameter count and return total."""
    params = {
        'Conv1  (2→8,  k=3)':   3*3*2*8   + 8,
        'BN1':                   8*2,
        'Conv2  (8→16, k=3)':   3*3*8*16  + 16,
        'BN2':                   16*2,
        'DW3   (16→16, k=3)':   3*3*16    + 16,
        'PW3   (16→32, k=1)':   1*1*16*32 + 32,
        'BN3':                   32*2,
        'DW4   (32→32, k=3)':   3*3*32    + 32,
        'PW4   (32→64, k=1)':   1*1*32*64 + 64,
        'BN4':                   64*2,
        'FC1   (64→32)':         64*32     + 32,
        'FC_BN1':                32*2,
        'FC2   (32→2)':          32*2      + 2,
    }
    ultra_total = 26_906
    total = sum(params.values())

    print('=' * 60)
    print('NANONET PARAMETER COUNT')
    print('=' * 60)
    for name, count in params.items():
        print(f'{name:<25} {count:>8,}  ({count/total*100:.1f}%)')
    print('-' * 60)
    print(f'{"TOTAL":<25} {total:>8,}')
    print(f'vs UltraMNet:              {ultra_total:>8,}  (-{(1-total/ultra_total)*100:.1f}%)')
    print(f'Model size:                {total*4/1024:>7.1f} KB  (float32)')
    print(f'Input:                     45×60×2')
    print('=' * 60)
    return total


if __name__ == '__main__':
    model = get_model_nano()

    # Test forward pass
    x = mx.random.normal((2, 45, 60, NUM_CHANNELS))
    out = model(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {out.shape}')
    print()
    count_parameters()
