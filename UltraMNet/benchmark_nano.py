"""
Benchmark script for NanoNet inference on Mac M4.
Measures FPS (frames per second) for different inference modes:
1. Single image inference (CPU)
2. Batched inference (CPU)
3. MLX GPU acceleration (if available)
"""

import sys
import time
import numpy as np
import mlx.core as mx
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from UltraMNet.model_nano import HandDetectorNanoNet, get_model_nano
from UltraMNet.config import NUM_CHANNELS


def create_test_data(batch_size=1, height=45, width=60, channels=NUM_CHANNELS):
    """Create random test data simulating real input."""
    return mx.random.normal((batch_size, height, width, channels))


def benchmark_single(model, num_iterations=1000, warmup=100):
    """Benchmark single image inference (batch_size=1)."""
    # Warmup
    for _ in range(warmup):
        x = create_test_data(batch_size=1)
        _ = model(x)
        mx.eval(model.parameters())
    
    # Benchmark - measure actual inference with mx.eval inside loop
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        x = create_test_data(batch_size=1)
        logits = model(x)
        mx.eval(logits)
    
    end = time.perf_counter()
    
    elapsed = end - start
    fps = num_iterations / elapsed
    
    print(f"\n{'='*70}")
    print(f"SINGLE IMAGE INFERENCE (batch_size=1) - With eval sync")
    print(f"{'='*70}")
    print(f"  Iterations:        {num_iterations:,}")
    print(f"  Total time:        {elapsed:.3f} seconds")
    print(f"  Avg per image:     {elapsed/num_iterations*1000:.3f} ms")
    print(f"  FPS:               {fps:,.1f}")
    print(f"{'='*70}")
    
    # Also benchmark without mx.eval to see theoretical max
    start2 = time.perf_counter()
    for _ in range(num_iterations):
        x = create_test_data(batch_size=1)
        logits = model(x)
    
    end2 = time.perf_counter()
    elapsed2 = end2 - start2
    fps2 = num_iterations / elapsed2
    
    print(f"\n{'='*70}")
    print(f"SINGLE IMAGE INFERENCE (batch_size=1) - Async (theoretical max)")
    print(f"{'='*70}")
    print(f"  Iterations:        {num_iterations:,}")
    print(f"  Total time:        {elapsed2:.3f} seconds")
    print(f"  Avg per image:     {elapsed2/num_iterations*1000:.3f} ms")
    print(f"  FPS:               {fps2:,.1f}")
    print(f"{'='*70}")
    
    return fps


def benchmark_batch(model, batch_sizes=[1, 4, 8, 16, 32], num_iterations=100, warmup=10):
    """Benchmark batched inference with different batch sizes."""
    print(f"\n{'='*70}")
    print(f"BATCHED INFERENCE BENCHMARK")
    print(f"{'='*70}")
    print(f"{'Batch Size':<15} {'FPS':<15} {'ms/image':<15} {'Throughput (img/s)':<20}")
    print(f"{'-'*70}")
    
    results = {}
    
    for batch_size in batch_sizes:
        # Warmup
        for _ in range(warmup):
            x = create_test_data(batch_size=batch_size)
            _ = model(x)
            mx.eval(model.parameters())
        
        # Benchmark
        start = time.perf_counter()
        
        for _ in range(num_iterations):
            x = create_test_data(batch_size=batch_size)
            logits = model(x)
            mx.eval(logits)
        
        end = time.perf_counter()
        
        elapsed = end - start
        total_images = batch_size * num_iterations
        fps_batch = total_images / elapsed
        ms_per_image = elapsed / total_images * 1000
        
        results[batch_size] = {
            'fps': fps_batch,
            'ms_per_image': ms_per_image,
            'elapsed': elapsed
        }
        
        print(f"{batch_size:<15} {fps_batch:<15,.1f} {ms_per_image:<15,.2f} {fps_batch:<20,.1f}")
    
    print(f"{'='*70}")
    return results


def benchmark_with_device_info():
    """Print device information and run benchmarks."""
    print(f"\n{'='*70}")
    print(f"NanoNet Inference Benchmark - Mac M4")
    print(f"{'='*70}")
    
    # Print system info
    print(f"\nSystem Information:")
    print(f"  Device:            Apple M4 (Mac)")
    print(f"  Input resolution:  45×60×{NUM_CHANNELS}")
    print(f"  Model parameters:  ~7K")
    print(f"  Model size:        ~31 KB (float32)")
    
    # Create model
    print(f"\nCreating NanoNet model...")
    model = get_model_nano()
    model.eval()
    
    # Test forward pass
    x = create_test_data(batch_size=1)
    out = model(x)
    print(f"  Model created successfully")
    print(f"  Test input shape:  {x.shape}")
    print(f"  Test output shape: {out.shape}")
    
    # Run benchmarks
    single_fps = benchmark_single(model, num_iterations=1000, warmup=100)
    batch_results = benchmark_batch(model, batch_sizes=[1, 4, 8, 16, 32], num_iterations=200, warmup=20)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Single image FPS:     {single_fps:>10,.1f}")
    print(f"  Batch=32 throughput:  {batch_results[32]['fps']:>10,.1f} img/s")
    print(f"  Latency (single):     {1000/single_fps:>10,.2f} ms")
    print(f"  Latency (batch=32):   {batch_results[32]['ms_per_image']:>10,.2f} ms/image")
    print(f"{'='*70}")
    
    # Comparison with reported values
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH DOCUMENTED PERFORMANCE")
    print(f"{'='*70}")
    print(f"  Documented (Cortex-A53 FP32):  60-90 FPS")
    print(f"  Documented (Cortex-A53 INT8):  120-180 FPS")
    print(f"  Expected (Mac M4 FP32):        200-500+ FPS (estimated)")
    print(f"  Measured (this benchmark):     {single_fps:,.1f} FPS")
    print(f"{'='*70}")
    
    return single_fps, batch_results


if __name__ == '__main__':
    fps, batch_results = benchmark_with_device_info()
