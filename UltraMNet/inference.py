"""
Inference script for Ultra Micro hand detection CNN (45x60 input).
Loads 90x120 data, downsamples using pixel skipping, and runs inference.
"""

import os
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
from scipy.ndimage import label as ndimage_label

# Setup import path correctly regardless of how script is invoked
_current_dir = Path(__file__).parent.resolve()
_project_root = _current_dir.parent

# Try module import first (when run as: python -m UltraMNet.inference)
try:
    from UltraMNet.config import (
        NUM_CHANNELS, CONFIDENCE_THRESHOLD,
        BEST_MODEL_PATH, ULTRA_BOX_SIZE
    )
    from UltraMNet.model import HandDetectorCNNUltraMicro
    from UltraMNet.dataset import downsample_pixel_skip
except ImportError:
    # Fallback: direct import (when run as: python inference.py)
    sys.path.insert(0, str(_current_dir))
    from config import (
        NUM_CHANNELS, CONFIDENCE_THRESHOLD,
        BEST_MODEL_PATH, ULTRA_BOX_SIZE
    )
    from model import HandDetectorCNNUltraMicro
    from dataset import downsample_pixel_skip


def load_model(model_path=None):
    """Load trained MLX model."""
    if model_path is None:
        model_path = Path(BEST_MODEL_PATH)
    
    if model_path.suffix == '.pth':
        model_path = model_path.with_suffix('.pth.safetensors')
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first.")
    
    # Load safetensors
    weights = mx.load(str(model_path))
    
    model = HandDetectorCNNUltraMicro()
    
    # Convert weights to proper format
    weight_dict = {k: v for k, v in weights.items() if v.ndim > 0}
    model.load_weights(list(weight_dict.items()))
    mx.eval(model.parameters())
    model.eval()  # CRITICAL: Set to eval mode for correct BatchNorm behavior

    print(f"Loaded model from: {model_path}")

    return model


def load_depth_data(filepath):
    """Load depth and amplitude data from file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
    elif filepath.suffix == '.mat':
        data = scipy.io.loadmat(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    depth = data['depth_mm'].astype(np.float32)
    amp = data['amp'].astype(np.float32)
    
    return depth, amp, data


def preprocess_and_downsample(depth, amp):
    """Downsample to 45x60 and preprocess for model input."""
    # Downsample using pixel skipping (very cheap)
    depth_ds = downsample_pixel_skip(depth, factor=2)
    amp_ds = downsample_pixel_skip(amp, factor=2)
    
    # Normalize depth
    depth_min = depth_ds.min()
    depth_max = depth_ds.max()
    if depth_max - depth_min > 0:
        depth_norm = (depth_ds - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = depth_ds - depth_min
    
    # Normalize amplitude
    amp_max = amp_ds.max()
    if amp_max > 0:
        amp_norm = amp_ds / amp_max
    else:
        amp_norm = amp_ds
    
    # Stack and convert to MLX array (1, H, W, C)
    image = np.stack([depth_norm, amp_norm], axis=-1)  # (H, W, C)
    image = mx.array(image)[None]  # (1, H, W, C)
    
    return image, (depth_min, depth_max), depth_ds


def detect_hand(model, depth, amp, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Detect hand in downsampled depth image.
    Returns results in original 90x120 coordinates.
    """
    # Preprocess and downsample
    image_tensor, norm_params, depth_ds = preprocess_and_downsample(depth, amp)
    
    # Predict
    logits = model(image_tensor)
    probabilities = nn.softmax(logits, axis=1)
    
    # Extract results
    hand_confidence = probabilities[0, 1].item()
    no_hand_confidence = probabilities[0, 0].item()
    is_hand = hand_confidence >= confidence_threshold
    
    result = {
        'is_hand': is_hand,
        'hand_confidence': hand_confidence,
        'no_hand_confidence': no_hand_confidence,
        'closest_depth': None,
        'hand_region': None,
        'hand_bbox': None
    }
    
    # If hand detected, find hand region and depth
    if is_hand:
        hand_region_ds, bbox_ds, closest_depth_ds = find_hand_region_and_depth(depth_ds)
        
        # Convert back to original coordinates (multiply by 2)
        if bbox_ds is not None:
            x_ds, y_ds, w_ds, h_ds = bbox_ds
            bbox = (x_ds * 2, y_ds * 2, w_ds * 2, h_ds * 2)
        else:
            bbox = None
        
        # Upsample mask
        hand_region = np.kron(hand_region_ds, np.ones((2, 2), dtype=np.uint8))
        
        # Find closest depth in original resolution
        closest_depth = find_closest_depth_in_region(depth, hand_region)
        
        result['hand_region'] = hand_region
        result['hand_bbox'] = bbox
        result['closest_depth'] = closest_depth
    
    return result


def find_hand_region_and_depth(depth_ds, box_size=4):
    """
    Find hand region in 45x60 depth image.
    Returns small bounding box around closest point.
    """
    h, w = depth_ds.shape
    
    # Find the closest 2x2 block
    min_depth = np.inf
    best_y, best_x = 0, 0
    
    for y in range(h - 1):
        for x in range(w - 1):
            block = depth_ds[y:y+2, x:x+2]
            block_mean = block.mean()
            if block_mean < min_depth:
                min_depth = block_mean
                best_y, best_x = y, x
    
    closest_depth = float(depth_ds[best_y:best_y+2, best_x:best_x+2].mean())
    
    # Create small bounding box around the closest point
    center_y = best_y + 1
    center_x = best_x + 1
    
    # Calculate box size (small, max 8x8 for 45x60)
    bx = max(0, center_x - box_size)
    by = max(0, center_y - box_size)
    bw = min(2 * box_size, w - bx)
    bh = min(2 * box_size, h - by)
    
    bbox = (bx, by, bw, bh)
    
    # Create a small mask
    hand_mask = np.zeros_like(depth_ds, dtype=np.uint8)
    hand_mask[by:by+bh, bx:bx+bw] = 1
    
    return hand_mask, bbox, closest_depth


def find_closest_depth_in_region(depth_orig, hand_region):
    """Find closest 2x2 depth in original resolution."""
    h, w = depth_orig.shape
    masked_depth = np.where(hand_region, depth_orig, np.inf)
    
    min_depth = np.inf
    best_y, best_x = 0, 0
    
    for y in range(h - 1):
        for x in range(w - 1):
            block = masked_depth[y:y+2, x:x+2]
            if np.any(block != np.inf):
                block_mean = np.mean(block[block != np.inf])
                if block_mean < min_depth:
                    min_depth = block_mean
                    best_y, best_x = y, x
    
    return float(depth_orig[best_y:best_y+2, best_x:best_x+2].mean())


def visualize_result(depth_orig, amp_orig, result, save_path=None):
    """Visualize detection result."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Original depth image
    im0 = axes[0].imshow(depth_orig, cmap='viridis')
    axes[0].set_title('Original Depth Image (90x120)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Height (pixels)')
    plt.colorbar(im0, ax=axes[0], label='Depth (mm)')
    
    # Hand region with bounding box
    if result['hand_region'] is not None and result['hand_bbox'] is not None:
        masked_depth = np.where(result['hand_region'], depth_orig, np.nan)
        im1 = axes[1].imshow(masked_depth, cmap='hot')
        axes[1].set_title(f'HAND DETECTED (Ultra Micro 45x60)\nConfidence: {result["hand_confidence"]:.2%}', 
                         fontsize=12, fontweight='bold', color='green')
        
        # Draw bounding box
        x, y, w, h = result['hand_bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='lime', linewidth=3, alpha=0.9)
        axes[1].add_patch(rect)
        
        # Add depth annotation
        if result['closest_depth'] is not None:
            axes[1].text(x + w//2, y - 8, 
                       f'Closest: {result["closest_depth"]:.1f} mm',
                       color='lime', fontsize=10, fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    else:
        axes[1].imshow(depth_orig, cmap='gray')
        axes[1].set_title(f'NO HAND DETECTED (Ultra Micro 45x60)\nConfidence: {result["no_hand_confidence"]:.2%}', 
                         fontsize=12, fontweight='bold', color='red')
        im1 = axes[1].images[0]
    
    plt.colorbar(im1, ax=axes[1], label='Depth (mm)')
    
    # Depth histogram
    axes[2].hist(depth_orig.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    if result['closest_depth'] is not None:
        axes[2].axvline(result['closest_depth'], color='red', linewidth=3, 
                       label=f'Closest 2x2: {result["closest_depth"]:.1f} mm')
        axes[2].axvspan(result['closest_depth'] - 10, result['closest_depth'] + 10, 
                       alpha=0.2, color='red')
    axes[2].set_title('Depth Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Depth (mm)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_yscale('log')
    if result['closest_depth'] is not None:
        axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def process_single_file(model, filepath, save_visualization=True):
    """Process a single depth image file."""
    filepath = Path(filepath)
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name}")
    print(f"{'='*60}")
    
    # Load data (90x120)
    depth_orig, amp_orig, raw_data = load_depth_data(filepath)
    
    # Detect hand
    result = detect_hand(model, depth_orig, amp_orig)
    
    # Print results
    print(f"Hand Detected: {result['is_hand']}")
    print(f"  Hand Confidence: {result['hand_confidence']:.2%}")
    print(f"  No-Hand Confidence: {result['no_hand_confidence']:.2%}")
    
    if result['closest_depth'] is not None:
        print(f"\nHand Region:")
        print(f"  Bounding Box: {result['hand_bbox']}")
        print(f"  Closest 2x2 Depth: {result['closest_depth']:.2f} mm")
    else:
        print("\nNo hand region detected.")
    
    # Visualize
    if save_visualization:
        output_path = filepath.parent / (filepath.stem + '_ultra_detection.png')
        visualize_result(depth_orig, amp_orig, result, save_path=output_path)
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra Micro Hand Detection (45x60 input, 27K params)')
    parser.add_argument('input', type=str, nargs='?', 
                       help='Input file or directory')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization saving')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Process input
    if args.input is None:
        # Demo with example files
        print("\nNo input specified. Running demo on sample files...")
        
        pos_file = Path("datasets/positive/sample_positive_001000.npz")
        if pos_file.exists():
            process_single_file(model, pos_file, save_visualization=not args.no_vis)
        
        neg_file = Path("datasets/negative/sample_negative_000001.npz")
        if neg_file.exists():
            process_single_file(model, neg_file, save_visualization=not args.no_vis)
    
    else:
        input_path = Path(args.input)
        
        if input_path.is_file():
            process_single_file(model, input_path, save_visualization=not args.no_vis)
        elif input_path.is_dir():
            # Batch processing
            files = sorted(list(input_path.glob("*.npz")))
            print(f"\nProcessing {len(files)} files from: {input_path}")
            hand_count = 0
            
            for filepath in files[:10]:  # Limit to 10 for demo
                result = process_single_file(model, filepath, save_visualization=not args.no_vis)
                if result['is_hand']:
                    hand_count += 1
            
            print(f"\nProcessed {min(10, len(files))} files, {hand_count} hands detected")
        else:
            print(f"Error: {args.input} not found")
            sys.exit(1)


if __name__ == "__main__":
    main()
