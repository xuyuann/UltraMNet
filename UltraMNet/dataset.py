"""
Dataset class for ultra micro CNN with 45x60 input resolution.
Downsamples 90x120 images using simple pixel skipping.
"""

import os
import numpy as np
import mlx.core as mx
from pathlib import Path
import scipy.io
import sys
sys.path.append(str(Path(__file__).parent.parent))

from UltraMNet.config import (
    POSITIVE_DIR, NEGATIVE_DIR, IMG_SIZE, NUM_CHANNELS,
    BATCH_SIZE, TRAIN_SPLIT, USE_AUGMENTATION,
    H_FLIP_PROB, DEPTH_NOISE_STD,
    TEST_POSITIVE_DIR, TEST_NEGATIVE_DIR
)


def downsample_pixel_skip(image, factor=2):
    """Simple pixel skipping downsampling (very low compute cost)."""
    return image[::factor, ::factor].copy()


class HandDepthDatasetUltraMicro:
    """Dataset for ultra micro CNN with 45x60 input."""
    
    def __init__(self, file_paths, labels, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load data (90x120)
        data = self._load_file(self.file_paths[idx])
        
        # Extract channels
        depth = data['depth_mm'].astype(np.float32)
        amp = data['amp'].astype(np.float32)
        
        # Downsample to 45x60 using pixel skipping (very cheap)
        depth_ds = downsample_pixel_skip(depth, factor=2)
        amp_ds = downsample_pixel_skip(amp, factor=2)
        
        # Check for NaN in raw data
        if np.isnan(depth_ds).any():
            depth_ds = np.nan_to_num(depth_ds, nan=0.0)
        if np.isnan(amp_ds).any():
            amp_ds = np.nan_to_num(amp_ds, nan=0.0)
        
        # Normalize depth to 0-1 range
        depth_min = depth_ds.min()
        depth_max = depth_ds.max()
        depth_range = depth_max - depth_min
        if depth_range > 1e-6:
            depth_norm = (depth_ds - depth_min) / depth_range
        else:
            depth_norm = np.zeros_like(depth_ds)
        
        # Normalize amplitude to 0-1 range
        amp_max = amp_ds.max()
        if amp_max > 1e-6:
            amp_norm = amp_ds / amp_max
        else:
            amp_norm = np.zeros_like(amp_ds)
        
        # Stack channels (H, W, C) for MLX
        image = np.stack([depth_norm, amp_norm], axis=-1)
        
        # Data augmentation
        if self.is_train and USE_AUGMENTATION:
            image = self._augment(image)
        
        # Convert to MLX array (H, W, C)
        image_mx = mx.array(image)
        label_mx = mx.array(self.labels[idx])
        
        return image_mx, label_mx
    
    def _augment(self, image):
        """Manual data augmentation."""
        # Random horizontal flip
        if np.random.random() < H_FLIP_PROB:
            image = image[:, ::-1, :].copy()
        
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.randn(*image.shape).astype(np.float32) * (DEPTH_NOISE_STD / 1000.0)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _load_file(self, filepath):
        """Load .npz or .mat file."""
        filepath = Path(filepath)
        if filepath.suffix == '.npz':
            return np.load(filepath, allow_pickle=True)
        elif filepath.suffix == '.mat':
            return scipy.io.loadmat(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


class BatchLoader:
    """Simple batch loader for MLX."""
    
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.n_batches = (len(dataset) + batch_size - 1) // batch_size
        self.reset()
    
    def __len__(self):
        return self.n_batches
    
    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
    
    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.current_idx:end_idx]
        
        images = []
        labels = []
        for idx in batch_indices:
            img, lbl = self.dataset[idx]
            images.append(np.array(img))
            labels.append(np.array(lbl))
        
        # Stack to (B, H, W, C) for MLX
        images = np.stack(images)
        labels = np.stack(labels)
        
        self.current_idx = end_idx
        
        return mx.array(images), mx.array(labels)


def get_dataloaders():
    """Create train/val batch loaders."""
    # Collect positive samples
    pos_files = sorted(list(POSITIVE_DIR.glob("*.npz")))
    pos_labels = [1] * len(pos_files)

    # Collect negative samples
    neg_files = sorted(list(NEGATIVE_DIR.glob("*.npz")))
    neg_labels = [0] * len(neg_files)

    # Combine
    all_files = pos_files + neg_files
    all_labels = pos_labels + neg_labels

    print(f"Positive samples: {len(pos_files)}")
    print(f"Negative samples: {len(neg_files)}")
    print(f"Total samples: {len(all_files)}")

    # Split into train/val
    num_total = len(all_files)
    num_train = int(num_total * TRAIN_SPLIT)

    # Shuffle indices (with fixed seed for reproducibility)
    np.random.seed(42)
    indices = np.random.permutation(num_total)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_files = [all_files[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]

    val_files = [all_files[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    # Create datasets
    train_dataset = HandDepthDatasetUltraMicro(train_files, train_labels, is_train=True)
    val_dataset = HandDepthDatasetUltraMicro(val_files, val_labels, is_train=False)

    print(f"Training samples: {len(train_dataset)} (45x60)")
    print(f"Validation samples: {len(val_dataset)} (45x60)")

    # Create batch loaders
    train_loader = BatchLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = BatchLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def get_test_dataloaders():
    """Create test batch loader from datasets2."""
    # Collect positive samples
    pos_files = sorted(list(TEST_POSITIVE_DIR.glob("*.npz")))
    pos_labels = [1] * len(pos_files)

    # Collect negative samples
    neg_files = sorted(list(TEST_NEGATIVE_DIR.glob("*.npz")))
    neg_labels = [0] * len(neg_files)

    # Combine
    all_files = pos_files + neg_files
    all_labels = pos_labels + neg_labels

    print(f"Test positive samples: {len(pos_files)}")
    print(f"Test negative samples: {len(neg_files)}")
    print(f"Test total samples: {len(all_files)}")

    # Create dataset
    test_dataset = HandDepthDatasetUltraMicro(all_files, all_labels, is_train=False)

    print(f"Test samples: {len(test_dataset)} (45x60)")

    # Create batch loader
    test_loader = BatchLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader


if __name__ == "__main__":
    # Test dataset loading
    train_loader, val_loader = get_dataloaders()
    
    # Check a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
