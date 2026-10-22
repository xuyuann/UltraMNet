"""
Test script for NanoNet hand detection CNN on datasets2.
"""

import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from UltraMNet.config import (
    NANO_BEST_MODEL_PATH, BATCH_SIZE
)
from UltraMNet.dataset import get_test_dataloaders
from UltraMNet.model_nano import HandDetectorNanoNet, count_parameters


def test_model(model, test_loader):
    """Test the model on the test set."""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    
    all_predictions = []
    all_labels = []

    for images, labels in test_loader:
        logits = model(images)
        predicted = mx.argmax(logits, axis=1)
        
        batch_size = images.shape[0]
        total_correct += (predicted == labels).sum().item()
        total_samples += batch_size
        
        for i in range(batch_size):
            label = labels[i].item()
            pred = predicted[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
            
            all_predictions.append(pred)
            all_labels.append(label)

    overall_acc = 100.0 * total_correct / max(total_samples, 1)
    pos_acc = 100.0 * class_correct[1] / max(class_total[1], 1)
    neg_acc = 100.0 * class_correct[0] / max(class_total[0], 1)
    
    return overall_acc, pos_acc, neg_acc, all_predictions, all_labels


def main():
    print("=" * 80)
    print("Testing NanoNet on datasets2")
    print("=" * 80)
    
    # Load test data
    print("\nLoading test data from datasets2...")
    test_loader = get_test_dataloaders()
    
    # Create model
    print("\nCreating NanoNet model...")
    model = HandDetectorNanoNet()
    
    # Count parameters
    total_params = count_parameters()
    print(f"Model parameters: {total_params:,}")
    
    # Load best model weights
    print(f"\nLoading best model from: {NANO_BEST_MODEL_PATH}")
    try:
        weights = mx.load(str(NANO_BEST_MODEL_PATH))
        model.load_weights(list(weights.items()))
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Using randomly initialized model (not trained)")
        return
    
    # Test the model
    print("\n" + "=" * 80)
    print("Running tests...")
    print("=" * 80)
    
    overall_acc, pos_acc, neg_acc, predictions, labels = test_model(model, test_loader)
    
    # Report results
    print("\n" + "=" * 80)
    print("TEST RESULTS - NanoNet")
    print("=" * 80)
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Hand (Positive) Accuracy: {pos_acc:.2f}%")
    print(f"No-Hand (Negative) Accuracy: {neg_acc:.2f}%")
    print(f"Total samples tested: {len(labels)}")
    print(f"Model parameters: {total_params:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
