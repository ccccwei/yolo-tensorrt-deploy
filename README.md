# yolo-tensorrt-deploy

YOLO26 object detection deployment in C++ with NVIDIA TensorRT and CUDA. A single-file example with ONNX export, manual CPU preprocessing, image-folder inference, and per-stage timing.

简洁的 C++ / TensorRT YOLO26n 图像检测示例。单文件实现前处理、推理和后处理，支持文件夹逐图检测、画框和分阶段计时。

## 环境

CMake 3.24+、C++17 编译器、CUDA Toolkit 11.8+、TensorRT 8.6.x、OpenCV 3.4+（core、imgproc、imgcodecs）。已在 RTX 3060、CUDA 11.8、TensorRT 8.6.1、OpenCV 3.4.20 上验证。

## 准备模型

PT、ONNX 和 Engine 文件不随仓库提交。导出需要 Python 3.10 和 uv；运行 C++ 程序不依赖 Python。

```bash
uv venv .venv-export --python 3.10
uv pip install --python .venv-export/bin/python -r tools/export-requirements.txt
.venv-export/bin/python tools/export_onnx.py yolo26n.pt
sh tools/build_engine.sh weights/yolo26n.onnx
```

导出脚本自动下载 Ultralytics 官方权重。转换脚本默认使用 `/usr/local/TensorRT/bin/trtexec`，可通过 `TRTEXEC` 环境变量指定位置。Engine 应在目标 GPU 和 TensorRT 环境中生成。

## 编译

```bash
cmake --preset release
cmake --build --preset release
```

## 推理

```bash
./build/release/src/yolo_deploy weights/yolo26n.fp16.engine images
```

程序会统计并推理文件夹中的 JPG、JPEG、PNG 和 BMP 图片，带检测框的图片保存到 `results/`。

`images/` 中的两张测试图片来自 Ultralytics 安装包。

每张图片打印预处理、推理和后处理耗时，单位为毫秒。推理耗时包含 H2D、TensorRT 执行、D2H 和同步，后处理包含画框；不计入读图、保存图片和日志输出。未额外预热，首张图片的耗时可能较高。

当前仅支持静态 Batch 1、RGB、640×640 的 YOLO26 Engine。
