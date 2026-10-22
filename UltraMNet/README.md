# UltraMNet - Ultra Micro CNN for hand detection at 45x60 resolution (~27K parameters)

## 概述

Ultra Micro 模型使用 45x60 输入分辨率（从 90x120 通过像素跳过降采样），参数量仅 27K，非常适合低功耗嵌入式部署。

## 使用方法

### 训练

```bash
# 从项目根目录运行
python -m UltraMNet.train
```

### 推理

```bash
# 单文件检测
python -m UltraMNet.inference datasets/positive/sample_positive_001000.npz

# 指定模型路径
python -m UltraMNet.inference datasets/positive/sample_positive_001000.npz \
  --model UltraMNet/models/best_ultra_micro_model.safetensors

# 不保存可视化
python -m UltraMNet.inference datasets/positive/sample_positive_001000.npz --no-vis
```

## 关键特性

### 降采样方法

使用简单的像素跳过（pixel skipping）从 90x120 降到 45x60：

```python
depth_45x60 = depth_90x120[::2, ::2]
```

**计算成本**：几乎为零（仅内存复制）

### 模型架构

```
Input: 45 × 60 × 2 (depth + amplitude)
    ↓
Conv2d(2, 8) + BN + ReLU + MaxPool   → 22×30×8
    ↓
Conv2d(8, 16) + BN + ReLU + MaxPool  → 11×15×16
    ↓
Conv2d(16, 32) + BN + ReLU + MaxPool → 5×7×32
    ↓
Conv2d(32, 64) + BN + ReLU + MaxPool + Dropout → 2×3×64
    ↓
Global Average Pooling → 64
    ↓
FC(64, 32) + BN + ReLU + Dropout
    ↓
FC(32, 2) → [hand / no-hand]
```

### 性能对比

| 模型 | 参数量 | FLOPs | 验证精度 | Cortex-A53 估算 |
|------|--------|-------|---------|----------------|
| Original (90×120) | 422K | 40M | 99.78% | 3-5 FPS |
| Small (90×120) | 106K | 40M | 99.35% | 3-5 FPS |
| Micro (45×60) | 106K | 9.4M | 99.68% | 12-20 FPS |
| **Ultra Micro (45×60)** | **27K** | **3.5M** | **99.57%** | **30-50 FPS** |

## 文件结构

```
UltraMNet/
├── config.py          # 配置文件
├── model.py           # Ultra Micro CNN 网络定义
├── dataset.py         # 数据加载（含降采样）
├── train.py           # 训练脚本
├── inference.py       # 推理脚本
├── models/            # 训练好的模型
│   └── best_ultra_micro_model.safetensors
└── README.md
```

## 推理输出

推理结果在原始 90x120 坐标空间中，边界框和深度值都映射回原始分辨率。

```
Hand Detected: True
  Hand Confidence: 99.57%
  Bounding Box: (56, 76, 32, 32)  # 原始坐标
  Closest 2x2 Depth: 326.10 mm
```

## Cortex-A53 部署建议

1. **使用 NCNN 或 TFLite 框架**
2. **INT8 量化**：可进一步提升到 50+ FPS
3. **内存占用**：仅 105KB（float32）或 53KB（float16）
