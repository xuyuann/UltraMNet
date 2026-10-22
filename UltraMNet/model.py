"""
Ultra Micro CNN model for hand detection (45x60 input).
Minimal parameters: ~27K.
"""

import mlx.core as mx
import mlx.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from UltraMNet.config import NUM_CHANNELS, NUM_CLASSES, DROPOUT_RATE


class HandDetectorCNNUltraMicro(nn.Module):
    """
    Ultra Micro CNN - minimal parameters for 45x60 input.
    
    Architecture:
    Conv1: 2 -> 8   (k=3)
    Conv2: 8 -> 16  (k=3)
    Conv3: 16 -> 32 (k=3)
    Conv4: 32 -> 64 (k=3)
    FC1: 64 -> 32
    FC2: 32 -> 2
    
    Total parameters: ~27K
    """
    
    def __init__(self, num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()
        
        # Convolutional layers (reduced channels for 45x60)
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(32)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(64)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier (reduced)
        self.fc1 = nn.Linear(64, 32)
        self.fc_bn1 = nn.BatchNorm(32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def __call__(self, x):
        # x shape: (B, H, W, C) for MLX
        # Conv block 1: 45x60 -> 45x60 -> 22x30
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2: 22x30 -> 22x30 -> 11x15
        x = self.pool(nn.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3: 11x15 -> 11x15 -> 5x7
        x = self.pool(nn.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4: 5x7 -> 5x7 -> 2x3 (with dropout)
        x = self.dropout(self.pool(nn.relu(self.bn4(self.conv4(x)))))
        
        # Global average pooling: 2x3 -> 1x1
        x = mx.mean(x, axis=(1, 2))  # (B, 2, 3, 64) -> (B, 64)
        
        # Classifier
        x = nn.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model_ultra_micro():
    """Create and return the ultra micro model."""
    return HandDetectorCNNUltraMicro()


def count_parameters():
    """Count model parameters."""
    params = {
        'Conv1 (2->8)': 3*3*2*8 + 8,
        'BN1': 8*2,
        'Conv2 (8->16)': 3*3*8*16 + 16,
        'BN2': 16*2,
        'Conv3 (16->32)': 3*3*16*32 + 32,
        'BN3': 32*2,
        'Conv4 (32->64)': 3*3*32*64 + 64,
        'BN4': 64*2,
        'FC1 (64->32)': 64*32 + 32,
        'FC_BN1': 32*2,
        'FC2 (32->2)': 32*2 + 2,
    }
    
    total = sum(params.values())
    
    print('='*60)
    print('ULTRA MICRO MODEL PARAMETER COUNT')
    print('='*60)
    for name, count in params.items():
        print(f'{name:30s} {count:>10,} ({count/total*100:.1f}%)')
    print('-'*60)
    print(f'{"TOTAL":30s} {total:>10,}')
    print(f'Reduction: {(1 - total/422498)*100:.1f}% from original (422K)')
    print(f'Model size: {total * 4 / 1024:.1f} KB (float32)')
    print(f'Input: 45x60x2')
    print(f'FLOPs: ~3.5M (vs 40M original, -91%)')
    print('='*60)
    
    return total


if __name__ == "__main__":
    # Test model
    model = get_model_ultra_micro()
    print(model)
    
    # Test forward pass
    x = mx.random.normal((2, 45, 60, NUM_CHANNELS))
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    print()
    count_parameters()
