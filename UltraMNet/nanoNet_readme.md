# NanoNet — Hand Detection CNN (~7K Parameters)

基于 **Depthwise Separable Convolution** 对 UltraMNet 进行进一步压缩的极简手检测网络。  
参数量从 27K 压缩至 **7K（减少 74%）**，模型文件从 109 KB 缩至 **31 KB**，  
验证精度不仅未退化，反而提升至 **99.68%**（UltraMNet 99.25%）。

---

## 相对于 UltraMNet 的改动

UltraMNet 的参数分布极度集中：Conv4 一层（32→64）就占了总参数量的 **68.7%**。

```
UltraMNet 参数分布
Conv1 (2→8,   k=3)   152    0.6%
Conv2 (8→16,  k=3)  1168    4.3%
Conv3 (16→32, k=3)  4640   17.2%   ← 可优化
Conv4 (32→64, k=3) 18496   68.7%   ← 主要瓶颈
FC1   (64→32)        2080    7.7%
FC2   (32→2)           66    0.2%
TOTAL               26906
```

NanoNet 仅做**一处结构改动**：将 Conv3 和 Conv4 替换为 **Depthwise Separable Convolution（DSConv）**，其余（Conv1、Conv2、GAP、FC 头）完全不变。

### 什么是 Depthwise Separable Convolution

标准卷积 `k×k×C_in×C_out` 被拆成两步：

```
标准 Conv(16→32, k=3):   3×3×16×32 + 32 = 4,640 参数

DSConv(16→32, k=3):
  Depthwise  Conv(16→16, k=3, groups=16): 3×3×16 + 16  =   160   ← 每 channel 独立空间滤波
  Pointwise  Conv(16→32, k=1):            1×1×16×32 + 32 = 544   ← 跨 channel 混合信息
  合计:                                                      704   (节省 85%)
```

- **Depthwise** 负责捕捉空间特征，每个输入 channel 独立卷积
- **Pointwise** 负责跨 channel 信息融合，等价于线性变换

两者组合后的**感受野和输出形状**与标准卷积完全相同，因此网络其余部分无需任何修改。

---

## 网络结构

```
Input: 45 × 60 × 2  (depth_norm + amp_norm, 从 90×120 像素跳采样)
│
├─ Conv2d(2, 8, k=3, p=1)  + BN + ReLU + MaxPool(2)   → 22×30×8    [标准卷积]
├─ Conv2d(8, 16, k=3, p=1) + BN + ReLU + MaxPool(2)   → 11×15×16   [标准卷积]
│
├─ DSConv(16→32, k=3)      + BN + ReLU + MaxPool(2)   → 5×7×32     [替换为 DSConv]
│     ├─ DW: Conv2d(16,16, k=3, groups=16)
│     └─ PW: Conv2d(16,32, k=1)
│
├─ DSConv(32→64, k=3)      + BN + ReLU + MaxPool(2)   → 2×3×64     [替换为 DSConv]
│     ├─ DW: Conv2d(32,32, k=3, groups=32)
│     └─ PW: Conv2d(32,64, k=1)
│     └─ Dropout(0.3)
│
├─ Global Average Pooling                              → 64
│
├─ Linear(64, 32) + BN + ReLU + Dropout(0.3)
└─ Linear(32, 2)                                       → [no-hand / hand]
```

### 参数分布

| 层 | 参数量 | 占比 |
|---|---:|---:|
| Conv1 (2→8, k=3) | 152 | 2.2% |
| BN1 | 16 | 0.2% |
| Conv2 (8→16, k=3) | 1,168 | 16.9% |
| BN2 | 32 | 0.5% |
| DW3 (16→16, k=3) | 160 | 2.3% |
| PW3 (16→32, k=1) | 544 | 7.9% |
| BN3 | 64 | 0.9% |
| DW4 (32→32, k=3) | 320 | 4.6% |
| PW4 (32→64, k=1) | 2,112 | 30.6% |
| BN4 | 128 | 1.9% |
| FC1 (64→32) | 2,080 | 30.1% |
| FC_BN1 | 64 | 0.9% |
| FC2 (32→2) | 66 | 1.0% |
| **TOTAL** | **6,906** | **100%** |

**模型大小**: 27 KB（float32）

---

## UltraMNet vs NanoNet 对比

### 结构对比

| 组件 | UltraMNet | NanoNet | 变化 |
|---|---|---|---|
| Conv1 | Conv2d(2→8, k=3) | Conv2d(2→8, k=3) | 不变 |
| Conv2 | Conv2d(8→16, k=3) | Conv2d(8→16, k=3) | 不变 |
| Conv3 | Conv2d(16→32, k=3) | DSConv(16→32, k=3) | **替换** |
| Conv4 | Conv2d(32→64, k=3) | DSConv(32→64, k=3) | **替换** |
| GAP | GlobalAvgPool | GlobalAvgPool | 不变 |
| FC头 | 64→32→2 | 64→32→2 | 不变 |
| 输入尺寸 | 45×60×2 | 45×60×2 | 不变 |
| 输出 | 2-class logits | 2-class logits | 不变 |

### 性能对比（验证集，929 样本）

| 指标 | UltraMNet | NanoNet | Δ |
|---|---:|---:|---:|
| 参数量 | 26,906 | **6,906** | −74.3% |
| 模型文件大小 | 109 KB | **31 KB** | −71.6% |
| **最佳验证准确率** | 99.25% | **99.68%** | **+0.43%** |
| Hand 准确率 | 99.85% | **99.85%** | 0 |
| No-Hand 准确率 | 97.86% | **99.29%** | +1.42% |
| 最佳验证 Loss | 0.0246 | **0.0127** | −48% |
| 首次突破 99% 的 epoch | 11 | **6** | −5 |
| 总训练时长 | ~1 min | ~1.5 min | +0.5 min |

> NanoNet 在 No-Hand 精度上提升 **+1.42%**，这是因为更少的参数减少了过拟合，
> 模型对负样本的判断更加稳健。

### 计算量估算（Cortex-A53 参考）

| 模型 | FLOPs | FP32 FPS | INT8 FPS |
|---|---:|---:|---:|
| UltraMNet | ~3.5M | 20–30 | 40–60 |
| **NanoNet** | **~1M** | **60–90** | **120–180** |

---

## 快速开始

### 训练

```bash
# 从项目根目录运行
python -m UltraMNet.train_nano
```

训练参数（与 UltraMNet 相同，在 `UltraMNet/config.py` 中配置）：

| 参数 | 值 |
|---|---|
| Batch size | 32 |
| Epochs (max) | 50 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Early stopping patience | 15 epochs |
| Train/Val split | 80% / 20% |
| 数据增强 | 水平翻转 (p=0.5) + 深度噪声 (std=5mm, p=0.3) |

训练输出：
```
NanoNet training — ~7K params, 45×60 input (Apple Silicon MLX)
...
Best validation accuracy: 99.68%
Best model saved to: UltraMNet/models/best_nano_model.safetensors
Training history  → UltraMNet/models/training_history_nano.json
Total time: ~1.5 min (Apple Silicon M系列)
```

### 推理

NanoNet 使用 UltraMNet 的 inference 脚本，通过 `--model` 参数指定模型：

```bash
# 单文件检测
python -m UltraMNet.inference datasets/positive/sample_positive_001000.npz \
  --model UltraMNet/models/best_nano_model.safetensors

# 批量检测
python -m UltraMNet.inference datasets/positive/ \
  --model UltraMNet/models/best_nano_model.safetensors

# 不保存可视化图片
python -m UltraMNet.inference datasets/positive/sample_positive_001000.npz \
  --model UltraMNet/models/best_nano_model.safetensors --no-vis
```

推理输出示例：
```
============================================================
Processing: sample_positive_001000.npz
============================================================
Hand Detected: True
  Hand Confidence: 98.12%
  No-Hand Confidence: 1.88%

Hand Region:
  Bounding Box: (28, 38, 16, 16)
  Closest 2x2 Depth: 326.10 mm

Saved visualization to: sample_positive_001000_ultra_detection.png
```

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `model_nano.py` | NanoNet 模型定义，包含 `DSConv2d`、`HandDetectorNanoNet`、`count_parameters()` |
| `train_nano.py` | 训练脚本，逻辑与 `train.py` 完全相同，仅模型和 checkpoint 路径不同 |
| `models/best_nano_model.safetensors` | 最佳权重（31 KB） |
| `models/training_history_nano.json` | 完整训练曲线（loss / acc，按 epoch） |

---

## 嵌入式部署建议

NanoNet 的极小体积使其特别适合资源受限场景：

- **模型大小**: 31 KB（float32），INT8 量化后约 **8 KB**
- **推理峰值内存**: < 512 KB
- **推荐框架**: NCNN 或 TFLite（DSConv 均有原生支持）
- **Cortex-A53 预估**: FP32 60–90 FPS，INT8 120–180 FPS
