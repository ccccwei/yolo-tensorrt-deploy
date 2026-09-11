<p align="center">
  <img src="docs/assets/logo.svg" alt="YOLO TensorRT Deploy logo" width="80" height="80">
</p>

<h1 align="center">YOLO TensorRT Deploy</h1>

<p align="center">
  从简洁部署到极致速度
</p>

一个循序渐进的 **YOLO C++ / TensorRT / CUDA 部署与优化系列**。从一份容易读懂、能够跑通的推理代码出发，逐步理解性能瓶颈、实现优化，并用同一测试口径比较每一步的收益。

适合有基础 C++ 知识、希望学习部署与性能优化的学生和开发者，也为专业部署工程师提供可复现的对照实验。“极致速度”是探索目标，不是脱离硬件、模型和精度条件的性能承诺。

## 当前篇章：01 · 简洁部署

当前仓库提供系列第一篇的基础实现：使用 YOLO26n 跑通模型转换、图片推理和结果可视化。核心代码集中在 [src/main.cpp](src/main.cpp)，前处理、推理和后处理分别独立为函数，先看清完整流程。

后续篇章围绕性能测量、CUDA 前处理、内存与异步流水线、CUDA Graph 等主题展开，安排见 [系列目录与路线图](ROADMAP.md#series)。这些优化尚未实现，下面的使用说明均对应当前基础版。

## 当前功能

- **模型转换**：下载官方 PT 权重，导出 ONNX，通过 `trtexec` 构建开启 FP16 的 Engine。
- **手写前处理**：CPU Letterbox 缩放补边、BGR → RGB、归一化、HWC → CHW，不依赖 OpenCV DNN。
- **资源复用**：Engine 只加载一次，循环复用执行上下文、GPU 输入输出缓冲区和 CUDA Stream。
- **文件夹推理**：逐图检测、绘制检测框和类别 ID，保存结果并打印各阶段耗时。

## 环境要求

需要 NVIDIA GPU、兼容的驱动及以下依赖：

| 依赖 | 版本 / 要求 |
| --- | --- |
| C++ 编译器 | 支持 C++17 |
| CMake / Make | CMake 3.24+，使用 Unix Makefiles 构建 |
| CUDA Toolkit | 11.8+ |
| TensorRT | 8.6.x，当前构建不支持 TensorRT 10+ |
| OpenCV | 3.4+，需要 core、imgproc、imgcodecs |
| Python / uv | 仅模型导出需要，使用 Python 3.10 |

已验证配置：Linux、RTX 3060、CUDA 11.8、TensorRT 8.6.1、OpenCV 3.4.20。其他版本组合需要自行验证。

## 快速开始

以下命令均在仓库根目录执行。

### 1. 获取项目

```bash
git clone https://github.com/ccccwei/yolo-tensorrt-deploy.git
cd yolo-tensorrt-deploy
```

### 2. 导出 ONNX

先安装 Python 3.10 和 `uv`，再执行：

```bash
uv venv .venv-export --python 3.10
uv pip install --python .venv-export/bin/python -r tools/export-requirements.txt
.venv-export/bin/python tools/export_onnx.py yolo26n.pt
```

脚本自动下载 Ultralytics 官方权重，在 `weights/` 中生成 `yolo26n.pt` 和 `yolo26n.onnx`。PT、ONNX 和 Engine 文件不随仓库提交；运行 C++ 程序不依赖 Python。

### 3. 构建 TensorRT Engine

```bash
sh tools/build_engine.sh weights/yolo26n.onnx
```

默认输出 `weights/yolo26n.fp16.engine`。脚本使用 `/usr/local/TensorRT/bin/trtexec`，安装位置不同时可指定：

```bash
TRTEXEC=/path/to/TensorRT/bin/trtexec sh tools/build_engine.sh weights/yolo26n.onnx
```

请在目标 GPU 和 TensorRT 环境中构建 Engine，不要直接复用其他电脑生成的文件。

### 4. 编译并运行

```bash
cmake --preset release
cmake --build --preset release
./build/release/src/yolo_deploy weights/yolo26n.fp16.engine images
```

如果 CMake 找不到 TensorRT，配置时指定其安装目录：

```bash
cmake --preset release -DTensorRT_ROOT=/path/to/TensorRT
```

## 输入与结果

运行参数依次为 **Engine 路径**、**图片文件夹**。将命令中的 `images` 替换为自己的图片目录即可。

程序统计文件夹中的 JPG、JPEG、PNG 和 BMP 图片，按路径排序后逐张推理，不递归子目录。带检测框的图片按原文件名保存在当前工作目录的 `results/` 中，重复运行会覆盖同名结果。

每张图片打印检测数量及以下耗时，单位为毫秒：

- `Preprocess`：CPU 缩放、补边、通道转换和归一化。
- `Infer (H2D+TRT+D2H)`：数据传入 GPU、TensorRT 执行、结果传回 CPU，以及 CUDA Stream 同步。
- `Postprocess`：置信度过滤、坐标还原和画框。

计时不包含读图、保存图片和日志输出。程序没有额外预热，首张图片的耗时可能较高；推理计时不是纯 GPU 执行时间。

## 模型支持范围

当前 C++ 示例仅支持静态 Batch 1、640 × 640 的 YOLO26 检测模型，输入输出约定如下：

| 张量 | 名称 | 形状 | 类型 / 内容 |
| --- | --- | --- | --- |
| 输入 | `images` | `[1, 3, 640, 640]` | FP32，RGB，像素归一化至 0–1 |
| 输出 | `output0` | `[1, 300, 6]` | FP32，每行为 `[x1, y1, x2, y2, score, class_id]` |

Engine 构建时开启 FP16，但程序要求输入输出张量仍为 FP32。默认置信度阈值为 `0.25`，可修改 [src/main.cpp](src/main.cpp) 中的 `SCORE_THRESHOLD` 后重新编译。

导出脚本也能导出 YOLO11n，但当前 C++ 后处理不支持其输出格式。动态尺寸、多 Batch、分割和姿态模型暂不支持。

## 主要文件

```text
src/main.cpp              # 前处理、TensorRT 推理、后处理和图片循环
tools/export_onnx.py      # 下载权重并导出 ONNX
tools/build_engine.sh     # 使用 trtexec 构建 Engine
tools/export-requirements.txt
images/                   # 测试图片
weights/                  # 模型与 Engine 存放目录
```

模型权重及 `images/` 中的两张测试图片来自 [Ultralytics](https://github.com/ultralytics/ultralytics)。
