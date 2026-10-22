# UltraMNet — 26K 参数的手部检测 CNN（45×60，深度 + 幅值）

面向嵌入式部署的极小手部检测网络。输入是 90×120 的深度图与幅值图，按像素跳过降采样到 45×60，
输出二分类（有手 / 无手）加一个 8×8 的包围框。整个项目只有代码，数据集与训练权重不入库（见文末）。

## 两个模型

| 模型 | 参数量 | 权重体积 | 输入 | 说明 |
|---|---|---|---|---|
| **UltraMNet** | 26,906 | 109 KB（FP32 safetensors） | 45×60×2 | 主体模型，Conv4(32→64) 一层占总参数 68.7% |
| **NanoNet** | 6,906（−74.3%） | 31 KB | 45×60×2 | 把 Conv3/4 换成 Depthwise Separable（23,136 → 3,136 参数），No-Hand 精度 +1.42% |

Apple M4（MLX）上的实测推理（`NanoNet_M4_Benchmark_Report.md`）：

| 指标 | NanoNet | 条件 |
|---|---|---|
| 单帧延迟 | **0.60 ms** | batch=1，含 `mx.eval()` 同步等待 |
| 帧率 | **1,670.7 FPS** | 同上，真实可用值 |
| 异步理论峰值 | 11,803 FPS | 不含同步，反映 MLX 图执行优化上限 |
| batch=32 | 0.03 ms/帧，31,800 img/s | 吞吐优先 |

即 60/120/240 FPS 的实时需求下，单帧只占 3.6% / 7.2% / 14.4% 的时间预算。

## 目录

```
UltraMNet/
├── config.py                  # 全部超参、路径、输入分辨率（唯一配置入口）
├── dataset.py                 # 45×60 数据集与 DataLoader（90×120 → 45×60 像素跳过）
├── model.py                   # UltraMNet（MLX，27K）
├── model_nano.py              # NanoNet（MLX，7K，Depthwise Separable）
├── train.py / train_nano.py   # 训练脚本
├── test.py / test_nano.py     # 测试集评估
├── inference.py               # 单样本推理 + 可视化（两个模型共用，--model 切换）
├── benchmark_nano.py          # M4 上的 FPS / 延迟基准
├── model_pytorch.py           # UltraMNet 的 PyTorch 版（27K）
├── model_nano_pytorch.py      # NanoNet 的 PyTorch 版（7K）
├── convert_to_pytorch.py      # MLX safetensors → PyTorch .pt 权重转换
├── README.md                  # UltraMNet 细节（降采样、参数分布、可视化）
├── nanoNet_readme.md          # NanoNet 设计与压缩过程
└── NanoNet_M4_Benchmark_Report.md  # M4 基准报告（上面那张表的来源）
```

## 环境

训练与推理代码基于 **MLX**，需要在 Apple Silicon（macOS）上跑：

```bash
pip install -r requirements.txt
```

Linux / x86 上 MLX 不可用；那种机器只能用 `model_pytorch.py` / `model_nano_pytorch.py` 的
PyTorch 实现（结构等价，不是训练主线）。

## 用法

```bash
# 训练（在仓库根目录跑，UltraMNet 是可 import 的包）
python -m UltraMNet.train            # UltraMNet, 27K
python -m UltraMNet.train_nano       # NanoNet, 7K

# 评估
python -m UltraMNet.test
python -m UltraMNet.test_nano

# 单样本推理
python -m UltraMNet.inference datasets/datasets2/positive/sample_positive_000001.npz
python -m UltraMNet.inference <sample.npz> --model UltraMNet/models/best_nano_model.safetensors
python -m UltraMNet.inference <sample.npz> --no-vis      # 不出可视化

# M4 基准
python -m UltraMNet.benchmark_nano
```

## 关键超参（`UltraMNet/config.py`）

| 项 | 值 |
|---|---|
| 输入分辨率 | 45×60（由 90×120 做 `img[::2, ::2]` 得到，计算成本≈0） |
| 通道数 | 2（depth + amplitude） |
| 批大小 / 轮数 | 32 / 50 |
| 学习率 | 1e-3 |
| 训练/验证划分 | 80% / 20% |
| Dropout | 0.3 |
| 数据增强 | 水平翻转 p=0.5，深度噪声 σ=5.0 mm |
| 推理置信度阈值 | 0.5 |
| 包围框 | 8×8 |

## 数据与权重：不入库

`.gitignore` 显式排除 `datasets/`、`UltraMNet/models/`、`*.npz`、`*.pt`、`*.safetensors` 等，
仓库里只有源码与文档。要让代码跑起来，需要自行准备：

```
datasets/
├── datasets1/            # 训练集（按 80/20 划分）
│   ├── positive/*.npz
│   └── negative/*.npz
└── datasets2/            # 测试集
    ├── positive/*.npz
    └── negative/*.npz
UltraMNet/models/         # 训练产物：best_*/last_* 的 .safetensors
```

每个 `.npz` 样本是 90×120 的深度图与幅值图（`dataset.py` 里的读取逻辑即为格式定义）。
训练产物会写进 `UltraMNet/models/`（该目录被 git 忽略，`config.py` 会自动创建）。

## 与 PyTorch 的关系

MLX 是训练/推理主线，PyTorch 版只用于对照与权重格式转换（`convert_to_pytorch.py`），
两条实现的结构参数一一对应，可直接比参数量与 FLOPs。
