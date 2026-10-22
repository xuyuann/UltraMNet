"""
Training script for NanoNet hand detection CNN (~7K parameters).
Same pipeline as train.py — only the model and checkpoint paths differ.
Input: 45x60 (downsampled from 90x120 via pixel skipping).
"""

import sys
import time
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from UltraMNet.config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    NANO_BEST_MODEL_PATH, NANO_LAST_MODEL_PATH,
)
from UltraMNet.dataset import get_dataloaders, get_test_dataloaders
from UltraMNet.model_nano import HandDetectorNanoNet, count_parameters


def loss_fn(model, images, labels):
    logits = model(images)
    return mx.mean(nn.losses.cross_entropy(logits, labels))


def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = total_correct = total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, images, labels)

        if not mx.isnan(loss).item():
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters())

            batch_size = images.shape[0]
            total_loss    += loss.item() * batch_size
            total_samples += batch_size

            logits    = model(images)
            predicted = mx.argmax(logits, axis=1)
            total_correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 20 == 0 and total_samples > 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {total_loss/total_samples:.4f} '
                  f'Acc: {100.*total_correct/total_samples:.2f}%')

    return (total_loss / max(total_samples, 1),
            100. * total_correct / max(total_samples, 1))


def validate(model, val_loader):
    model.eval()
    total_loss = total_correct = total_samples = 0
    class_correct = {0: 0, 1: 0}
    class_total   = {0: 0, 1: 0}

    for images, labels in val_loader:
        logits = model(images)
        loss   = mx.mean(nn.losses.cross_entropy(logits, labels))

        batch_size     = images.shape[0]
        total_loss    += loss.item() * batch_size
        total_samples += batch_size

        predicted     = mx.argmax(logits, axis=1)
        total_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            lbl = labels[i].item()
            class_total[lbl]   += 1
            if predicted[i].item() == lbl:
                class_correct[lbl] += 1

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_acc  = 100. * total_correct / max(total_samples, 1)
    pos_acc    = 100. * class_correct[1] / max(class_total[1], 1)
    neg_acc    = 100. * class_correct[0] / max(class_total[0], 1)
    return epoch_loss, epoch_acc, pos_acc, neg_acc


def main():
    print("NanoNet training — ~7K params, 45×60 input (Apple Silicon MLX)")

    NANO_BEST_MODEL_PATH.parent.mkdir(exist_ok=True)

    print("\nLoading training data from datasets1...")
    train_loader, val_loader = get_dataloaders()
    
    print("\nLoading test data from datasets2...")
    test_loader = get_test_dataloaders()

    print("\nCreating NanoNet model...")
    model = HandDetectorNanoNet()
    count_parameters()

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    history = {k: [] for k in
               ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                'val_pos_acc', 'val_neg_acc',
                'test_loss', 'test_acc', 'test_pos_acc', 'test_neg_acc']}

    best_val_acc      = 0.0
    patience_counter  = 0
    early_stop        = 15

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Training data: datasets1 (split into train/val)")
    print(f"Test data: datasets2 (separate test set)")
    print(f"Best model → {NANO_BEST_MODEL_PATH}")
    print('-' * 80)

    start = time.time()

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_pos_acc, val_neg_acc = validate(model, val_loader)
        test_loss, test_acc, test_pos_acc, test_neg_acc = validate(model, test_loader)

        for k, v in zip(history, [train_loss, train_acc, val_loss,
                                   val_acc, val_pos_acc, val_neg_acc,
                                   test_loss, test_acc, test_pos_acc, test_neg_acc]):
            history[k].append(v)

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  Val Hand Acc: {val_pos_acc:.2f}% | Val No-Hand Acc: {val_neg_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"  Test Hand Acc: {test_pos_acc:.2f}% | Test No-Hand Acc: {test_neg_acc:.2f}%")
        print(f"  Time: {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            weights = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(str(NANO_BEST_MODEL_PATH), weights, metadata={
                'epoch':       str(epoch + 1),
                'val_acc':     str(val_acc),
                'val_pos_acc': str(val_pos_acc),
                'val_neg_acc': str(val_neg_acc),
                'test_acc': str(test_acc),
                'test_pos_acc': str(test_pos_acc),
                'test_neg_acc': str(test_neg_acc),
            })
            print(f"  >>> Saved best model (Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(NANO_LAST_MODEL_PATH), weights,
                        metadata={'epoch': str(epoch + 1), 'val_acc': str(val_acc)})

    history_path = NANO_BEST_MODEL_PATH.parent / "training_history_nano.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history → {history_path}")

    total_time = time.time() - start
    print(f"\n{'='*80}")
    print(f"Training complete!  Total time: {total_time/60:.1f} min")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {NANO_BEST_MODEL_PATH}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
