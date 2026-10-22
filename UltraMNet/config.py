"""
Ultra Micro CNN Configuration - 45x60 input resolution, ~27K parameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "datasets"
DATASET1_DIR = DATASET_DIR / "datasets1"  # Training dataset
DATASET2_DIR = DATASET_DIR / "datasets2"  # Testing dataset
POSITIVE_DIR = DATASET1_DIR / "positive"
NEGATIVE_DIR = DATASET1_DIR / "negative"
TEST_POSITIVE_DIR = DATASET2_DIR / "positive"
TEST_NEGATIVE_DIR = DATASET2_DIR / "negative"
MODEL_DIR = PROJECT_DIR / "UltraMNet" / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Input resolution (downsampled from 90x120)
IMG_SIZE = (45, 60)  # Height, Width
NUM_CHANNELS = 2  # depth + amp
NUM_CLASSES = 2  # hand (positive) vs no-hand (negative)

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

# Data augmentation
USE_AUGMENTATION = True
H_FLIP_PROB = 0.5
DEPTH_NOISE_STD = 5.0  # mm

# Model parameters
DROPOUT_RATE = 0.3

# Inference parameters
CONFIDENCE_THRESHOLD = 0.5
ULTRA_BOX_SIZE = 8  # 8x8 bounding box for ultra micro model

# Checkpoint — UltraMNet (27K)
BEST_MODEL_PATH = MODEL_DIR / "best_ultra_micro_model.safetensors"
LAST_MODEL_PATH = MODEL_DIR / "last_ultra_micro_model.safetensors"

# Checkpoint — NanoNet (7K)
NANO_BEST_MODEL_PATH = MODEL_DIR / "best_nano_model.safetensors"
NANO_LAST_MODEL_PATH = MODEL_DIR / "last_nano_model.safetensors"
