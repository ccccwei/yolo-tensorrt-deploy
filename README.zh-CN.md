<p align="center">
  <img src="docs/assets/logo.svg" alt="YOLO TensorRT Deploy logo" width="80" height="80">
</p>

<h1 align="center">YOLO TensorRT Deploy</h1>

<p align="center">
  从简洁部署到极致速度
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

YOLO26n 的 C++ / TensorRT / CUDA 部署与优化系列。从单文件推理开始，逐步优化前处理、内存传输和执行流程。

[推理流程](docs/architecture.md) · [系列目录](ROADMAP.md#series)

## 版本与性能

| 版本 | 实现 | 延迟 (ms) | 加速比 |
| --- | --- | ---: | ---: |
| [01 基础版](src/main.cpp) | CPU 前后处理 | — | — |

目前只有基础版。`—` 表示尚无完成预热与重复测试的正式基准数据，逐图日志不作为稳态延迟或加速比依据。后续版本及对照数据完成后再加入此表。

已验证环境：RTX 3060、Linux、CUDA 11.8、TensorRT 8.6.1、OpenCV 3.4.20。性能对比需固定硬件、模型、输入尺寸、精度和计时边界。

## 快速开始

依赖：NVIDIA GPU 及兼容驱动、C++17 编译器、CMake 3.24+、Make、CUDA Toolkit 11.8+、TensorRT 8.6.x、OpenCV 3.4+（core、imgproc、imgcodecs）。模型导出另需 Python 3.10 和 `uv`；C++ 推理不依赖 Python。

```bash
git clone https://github.com/ccccwei/yolo-tensorrt-deploy.git
cd yolo-tensorrt-deploy

# 1. 下载权重并导出 ONNX
uv venv .venv-export --python 3.10
uv pip install --python .venv-export/bin/python -r tools/export-requirements.txt
.venv-export/bin/python tools/export_onnx.py yolo26n.pt

# 2. 构建 FP16 Engine
sh tools/build_engine.sh weights/yolo26n.onnx

# 3. 编译
cmake --preset release
cmake --build --preset release

# 4. 推理图片文件夹
./build/release/src/yolo_deploy weights/yolo26n.fp16.engine images
```

PT、ONNX 和 Engine 不随仓库提交；导出脚本自动下载官方权重到 `weights/`。Engine 请在目标 GPU 和 TensorRT 环境中生成。

默认 `trtexec` 路径为 `/usr/local/TensorRT/bin/trtexec`。安装位置不同时，替换对应的构建与配置命令：

```bash
TRTEXEC=/path/to/TensorRT/bin/trtexec sh tools/build_engine.sh weights/yolo26n.onnx
cmake --preset release -DTensorRT_ROOT=/path/to/TensorRT
```

## 使用说明

运行参数依次为 `Engine 路径`、`图片文件夹`。支持 JPG、JPEG、PNG、BMP，按路径排序逐图推理，不递归子目录。检测结果保存在当前工作目录的 `results/`，重复运行会覆盖同名图片。

每张图片打印检测数量及三个阶段的耗时（ms）：

- `Preprocess`：CPU 缩放、补边、通道转换和归一化。
- `Infer (H2D+TRT+D2H)`：数据传入 GPU、TensorRT 执行、结果传回 CPU，以及 CUDA Stream 同步。
- `Postprocess`：置信度过滤、坐标还原和画框。

计时不含读图、保存图片和日志输出。未额外预热，首图可能较慢；`Infer` 不是纯 GPU 执行时间。

## 当前限制

当前 C++ 示例仅支持静态 Batch 1、640 × 640 的 YOLO26 检测模型，输入输出约定如下：

| 张量 | 名称 | 形状 | 类型 / 内容 |
| --- | --- | --- | --- |
| 输入 | `images` | `[1, 3, 640, 640]` | FP32，RGB，像素归一化至 0–1 |
| 输出 | `output0` | `[1, 300, 6]` | FP32，每行为 `[x1, y1, x2, y2, score, class_id]` |

Engine 内部允许 FP16，输入输出仍须为 FP32。默认置信度阈值 `0.25`，在 [src/main.cpp](src/main.cpp) 中修改 `SCORE_THRESHOLD` 后重新编译。

当前构建不支持 TensorRT 10+。导出脚本可导出 YOLO11n，但 C++ 后处理不支持其输出格式；动态尺寸、多 Batch、分割和姿态模型也暂不支持。

模型权重及 `images/` 中的两张测试图片来自 [Ultralytics](https://github.com/ultralytics/ultralytics)。
