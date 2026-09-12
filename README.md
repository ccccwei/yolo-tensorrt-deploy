<p align="center">
  <img src="docs/assets/logo.svg" alt="YOLO TensorRT Deploy logo" width="80" height="80">
</p>

<h1 align="center">YOLO TensorRT Deploy</h1>

<p align="center">
  From simple deployment to peak performance
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

A step-by-step YOLO26n deployment and optimization series using C++, TensorRT, and CUDA. Start with single-file inference, then progressively optimize preprocessing, memory transfers, and execution.

[Inference pipeline (Chinese)](docs/architecture.md) · [Series roadmap (Chinese)](ROADMAP.md#series)

## Versions and performance

| Version | Implementation | Latency (ms) | Speedup |
| --- | --- | ---: | ---: |
| [01 Baseline](src/main.cpp) | CPU preprocessing and postprocessing | — | — |

Only the baseline is available. `—` means a formal benchmark with warmup and repeated measurements is not yet available; per-image logs are not steady-state latency or speedup results. Further versions and comparisons will be added when ready.

Tested on RTX 3060, Linux, CUDA 11.8, TensorRT 8.6.1, and OpenCV 3.4.20. Performance comparisons must use the same hardware, model, input size, precision, and measurement boundaries.

## Quick start

Requirements: an NVIDIA GPU with a compatible driver, a C++17 compiler, CMake 3.24+, Make, CUDA Toolkit 11.8+, TensorRT 8.6.x, and OpenCV 3.4+ (core, imgproc, imgcodecs). Model export also requires Python 3.10 and `uv`; C++ inference does not depend on Python.

```bash
git clone https://github.com/ccccwei/yolo-tensorrt-deploy.git
cd yolo-tensorrt-deploy

# 1. Download weights and export ONNX
uv venv .venv-export --python 3.10
uv pip install --python .venv-export/bin/python -r tools/export-requirements.txt
.venv-export/bin/python tools/export_onnx.py yolo26n.pt

# 2. Build an FP16 engine
sh tools/build_engine.sh weights/yolo26n.onnx

# 3. Build the executable
cmake --preset release
cmake --build --preset release

# 4. Run inference on an image folder
./build/release/src/yolo_deploy weights/yolo26n.fp16.engine images
```

PT, ONNX, and engine files are not included in the repository. The export script downloads official weights into `weights/`. Build the engine on the target GPU with the target TensorRT environment.

The default `trtexec` path is `/usr/local/TensorRT/bin/trtexec`. For a different installation path, replace the corresponding engine-build and CMake configuration commands:

```bash
TRTEXEC=/path/to/TensorRT/bin/trtexec sh tools/build_engine.sh weights/yolo26n.onnx
cmake --preset release -DTensorRT_ROOT=/path/to/TensorRT
```

## Usage

The two arguments are `engine_path` and `image_folder`. JPG, JPEG, PNG, and BMP files are processed in path order, without scanning subdirectories. Annotated images are saved to `results/` under the current working directory. Running again overwrites files with the same names.

Each image prints its detection count and three stage timings in milliseconds:

- `Preprocess`: CPU resizing, padding, channel conversion, and normalization.
- `Infer (H2D+TRT+D2H)`: host-to-device transfer, TensorRT execution, device-to-host transfer, and CUDA stream synchronization.
- `Postprocess`: confidence filtering, coordinate restoration, and drawing boxes.

Timings exclude image loading, saving, and logging. There is no separate warmup, so the first image may be slower. `Infer` is not GPU-only execution time.

## Current limitations

The C++ example supports only static YOLO26 detection models with batch size 1 and a 640 × 640 input:

| Tensor | Name | Shape | Type / contents |
| --- | --- | --- | --- |
| Input | `images` | `[1, 3, 640, 640]` | FP32, RGB, pixel values normalized to 0–1 |
| Output | `output0` | `[1, 300, 6]` | FP32, each row is `[x1, y1, x2, y2, score, class_id]` |

The engine may use FP16 internally, but input and output tensors must remain FP32. The default confidence threshold is `0.25`; change `SCORE_THRESHOLD` in [src/main.cpp](src/main.cpp) and rebuild to adjust it.

The current build does not support TensorRT 10+. The export script can export YOLO11n, but the C++ postprocessor does not support its output format. Dynamic shapes, batch sizes greater than 1, segmentation, and pose models are also unsupported.

Model weights and the two sample images in `images/` come from [Ultralytics](https://github.com/ultralytics/ultralytics).
