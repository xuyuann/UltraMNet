"""
Convert UltraMNet / NanoNet MLX safetensors to PyTorch .pt files.

Usage:
    python -m UltraMNet.convert_to_pytorch          # Convert both
    python -m UltraMNet.convert_to_pytorch ultra     # UltraMNet only
    python -m UltraMNet.convert_to_pytorch nano      # NanoNet only
"""

import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open

MODEL_DIR = Path(__file__).parent / "models"


def convert_mlx_to_pytorch(mlx_path: Path) -> dict:
    """Load MLX safetensors and convert to PyTorch state_dict."""
    state = {}
    with safe_open(str(mlx_path), framework="numpy") as f:
        for key in f.keys():
            arr = f.get_tensor(key)

            # Remap MLX naming to PyTorch naming
            # MLX: conv3.dw.weight -> PyTorch: conv3_dw.weight
            pytorch_key = key
            if ".dw." in key:
                pytorch_key = key.replace(".dw.", "_dw.")
            elif ".pw." in key:
                pytorch_key = key.replace(".pw.", "_pw.")

            # Transpose conv weights (not BN, not FC)
            if ".weight" in pytorch_key and "bn" not in pytorch_key and "fc" not in pytorch_key:
                arr = np.transpose(arr, (0, 3, 1, 2))

            state[pytorch_key] = torch.from_numpy(arr.copy())

    return state


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    models = []
    if mode in ("ultra", "both"):
        models.append(("best_ultra_micro_model.safetensors", "best_ultra_micro_model.pt"))
    if mode in ("nano", "both"):
        models.append(("best_nano_model.safetensors", "best_nano_model.pt"))

    for mlx_name, pt_name in models:
        mlx_path = MODEL_DIR / mlx_name
        pt_path = MODEL_DIR / pt_name

        if not mlx_path.exists():
            print(f"⚠️ 跳过 {mlx_name}: 文件不存在")
            continue

        print(f"\nLoading MLX model: {mlx_path}")
        state_dict = convert_mlx_to_pytorch(mlx_path)
        print(f"  Converted {len(state_dict)} tensors")

        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, pt_path)

        file_size = pt_path.stat().st_size / 1024
        print(f"  Saved PyTorch model: {pt_path}")
        print(f"  File size: {file_size:.1f} KB")

    print("\nDone!")


if __name__ == "__main__":
    main()
