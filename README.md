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

A YOLO26m deployment series using C++, TensorRT, and CUDA, from an [annotated baseline](src/01_baseline/main.cpp) to GPU processing and pipeline optimization.

[Inference pipeline (Chinese)](docs/architecture.md) · [Series roadmap (Chinese)](ROADMAP.md#series) · [Optimization notes (Chinese)](docs/notes/README.md)

## Versions and performance

| Version | What it covers | Notes | 26m speedup (P50) | 26n historical speedup (P50) |
| --- | --- | --- | --- | --- |
| [01 Baseline](src/01_baseline/) | CPU pre/postprocessing, GPU inference, optional benchmarking | [01 (中文)](docs/notes/01_baseline.md) | 1.00× | 1.00× |
| [02 CUDA preprocessing/postprocessing](src/02_cuda_prepost/) | Fused preprocessing; optional GPU filtering and coordinate restoration | [02 (中文)](docs/notes/02_cuda_prepost.md) | 1.20× | 1.44× |
| [03 Pinned memory](src/03_pinned_memory/) | Reusable page-locked transfer buffers, with CPU postprocessing | [03 (中文)](docs/notes/03_pinned_memory.md) | 1.18× | 1.38× |
| [04 Async pipeline](src/04_async_pipeline/) | Two independent slots overlap requests; one-slot control | [04 (中文)](docs/notes/04_async_pipeline.md) | 0.72× | 1.06× |
| [05 CUDA Graph](src/05_cuda_graph/) | Capture H2D → preprocessing → TensorRT → D2H; CPU postprocessing | [05 (中文)](docs/notes/05_cuda_graph.md) | 1.25× | 1.57× |
| [06 Full Graph](src/06_full_graph/) | Capture GPU preprocessing, inference, and postprocessing together | [06 (中文)](docs/notes/06_full_graph.md) | 1.25× | 1.61× |
| [07 Graph pipeline](src/07_graph_pipeline/) | Combine full Graph with two slots; slot/Graph controls | [07 (中文)](docs/notes/07_graph_pipeline.md) | 0.70× | 0.93× |
| [08 Preprocessing maps](src/08_preprocess_maps/) | Precompute 1D interpolation maps for each image size; original-kernel control | [08 (中文)](docs/notes/08_preprocess_maps.md) | 1.25× | 1.58× |
| [09 Registered input](src/09_registered_input/) | Register the decoded image directly, removing its per-request CPU staging copy | [09 (中文)](docs/notes/09_registered_input.md) | 1.29× | 1.70× |
| [10 Split streams](src/10_split_streams/) | One TensorRT context, separate H2D / compute / D2H streams; 1–4 slots | [10 (中文)](docs/notes/10_split_streams.md) | 0.64× | 0.83× |
| [11 Grouped TopK](tools/grouped_topk.py) | Rewrite two TopK selections, preserving weights and I/O; reuse 05 or 09 | [11 (中文)](docs/notes/11_grouped_topk.md) | 1.31× | 1.76× |

Speedup = the same model’s 01 P50 / version P50, using default modes; 11 uses 09 + Grouped TopK. YOLO26m is the default model. [26m results](docs/benchmarks/rtx3060-yolo26m-2026-10-15/README.md) · [26n historical results](docs/benchmarks/rtx3060-unified-baseline-2026-09-15/README.md); the two cohorts are reported separately.

## Quick start

Requirements: NVIDIA GPU and driver, C++17, CMake 3.24+, Make, CUDA 11.8+, TensorRT 8.6.x, and OpenCV 3.4+ (core, imgproc, imgcodecs). Export additionally needs Python 3.10 and `uv`.

```bash
git clone https://github.com/ccccwei/yolo-tensorrt-deploy.git
cd yolo-tensorrt-deploy

# 1. Download weights and export ONNX
uv venv .venv-export --python 3.10
uv pip install --python .venv-export/bin/python -r tools/export-requirements.txt
.venv-export/bin/python tools/export_onnx.py yolo26m.pt

# 2. Build an FP16 engine
sh tools/build_engine.sh weights/yolo26m.onnx

# 3. Build for the local GPU
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release

# 4. Run inference on an image folder
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images
```

Models are not bundled. Export downloads weights into `weights/`; build the engine on the target GPU with its TensorRT environment.

For TensorRT installed outside `/usr/local/TensorRT`, use:

```bash
TRTEXEC=/path/to/TensorRT/bin/trtexec sh tools/build_engine.sh weights/yolo26m.onnx
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native -DTensorRT_ROOT=/path/to/TensorRT
```

For another GPU or builds without a visible GPU, replace `native` with the target architecture, e.g. `86` for RTX 3060.

## Usage

All executables take an engine and image folder. JPG/JPEG/PNG/BMP files are processed in path order, without recursion. Outputs go to `results/<version_directory>/` under the working directory; reruns overwrite CSVs and same-named images.

### 01–03 · Infer, measure, and compare

```bash
# 01 defaults to one inference per image, without warmup.
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images
# Benchmark: measured iterations, then warmup iterations (per image).
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images 1000 200
# 02: the final argument selects CPU (0, default) or GPU (1) postprocessing.
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 0
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_pinned_memory weights/yolo26m.fp16.engine images 1000 200
```

01 defaults to 1 measured iteration and 0 warmups; 02/03 default to 200/50 per image. Iterations must be positive; warmup may be zero. For benchmarks, use matching counts and Release builds on the same GPU.

02 separates its outputs into `results/02_cuda_prepost/cpu_post/` and `gpu_post/`; `GPU_POST=0` is the default.

Each image is decoded once, then warmed up and measured repeatedly. GPU resources are reused; 03 also reuses pinned host buffers.

| Stage | 01 Baseline | 02, CPU postprocessing | 03 Pinned memory |
| --- | --- | --- | --- |
| Preprocess | CPU resize, padding, RGB/CHW, normalization | Raw BGR uint8 H2D + fused CUDA kernel + stream sync | CPU copy to pinned staging + H2D + same CUDA kernel + sync |
| Infer | FP32 input H2D + TensorRT + output D2H + stream sync | TensorRT + output D2H + stream sync | TensorRT + D2H directly into pinned output + sync |
| Postprocess | Confidence filtering and coordinate restoration | Same as 01 | Same algorithm, reading pinned output directly |
| Pipeline | Preprocess + Infer + Postprocess | Same measurement boundary as 01 | Same measurement boundary as 01 |

With `GPU_POST=1`, GPU filtering and result D2H count toward **Infer**; **Postprocess** measures CPU unpacking. Compare **Pipeline** totals across modes.

Times are synchronized CPU wall-clock durations, excluding setup, warmup, file I/O, drawing, and logging. Input H2D moves into **Preprocess** in 02/03, so compare whole-pipeline timings.

03 includes the CPU staging copy in every iteration. Compare it with 02's CPU mode to measure the net effect of pinned memory. [Details](docs/notes/03_pinned_memory.md).

01–03 print mean/P50/P95/P99 per stage and image, and save:

- `timings.csv`: per-iteration stage and total times in milliseconds.
- `detections.csv`: each image's final detection results.
- Annotated images, saved after measurement.

### 04–07 · Test each optimization

Defaults: 200 measured iterations and 50 warmups per image. Additional switches follow these counts:

| Executable | Additional arguments |
| --- | --- |
| `yolo_async_pipeline` | `[SLOTS=2]` — 1 or 2 |
| `yolo_cuda_graph` | `[GRAPH=1]` — 0 or 1 |
| `yolo_full_graph` | `[GRAPH=1]` — 0 or 1 |
| `yolo_graph_pipeline` | `[SLOTS=2] [GRAPH=1]` |

```bash
./build/release/src/yolo_async_pipeline weights/yolo26m.fp16.engine images 1000 200 2
./build/release/src/yolo_cuda_graph weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_full_graph weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_graph_pipeline weights/yolo26m.fp16.engine images 1000 200 2 1
```

Compare one/two slots or Graph off/on; 07 supports all four combinations. Save each run's outputs before rerunning.

05/06 process images serially. 04/07 preload images and distribute requests across slots, each with its own context, stream, and buffers. CPU staging is always timed.

04–07 also save `throughput.csv`. Timing fields:

- `staging_ms` / `gpu_ms`: CPU staging / GPU chain, including transfers.
- `postprocess_ms` / `total_ms`: CPU result processing / full request latency.
- `throughput.csv`: completed samples / measured wall time, including pipeline fill/drain; not `1000 / P50`.

Setup, warmup, and file I/O are excluded. GPU event time is not pure model time; request latency excludes waiting for a free slot. Graph setup is reported separately.

### 08–10 · Additional controlled experiments

The same arguments and defaults apply; these switches come last:

| Executable | Additional arguments |
| --- | --- |
| `yolo_preprocess_maps` | `[MAPS=1]` — 0 or 1; Graph always on |
| `yolo_registered_input` | `[REGISTERED=1]` — 0 or 1; Graph always on |
| `yolo_split_streams` | `[SLOTS=2] [SPLIT=1]` — slots 1–4, split 0 or 1; no Graph |

```bash
./build/release/src/yolo_preprocess_maps weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_registered_input weights/yolo26m.fp16.engine images 1000 200 1
# Same slots, different scheduling: preserve both runs' outputs before comparing.
./build/release/src/yolo_split_streams weights/yolo26m.fp16.engine images 1000 200 2 0
./build/release/src/yolo_split_streams weights/yolo26m.fp16.engine images 1000 200 2 1
```

08 precomputes geometry maps during setup. 09 registers the original image for H2D, removing its CPU staging copy; the input allocation must remain valid until GPU work finishes.

10 shares **one** TensorRT context. `SPLIT=1` separates transfer and compute streams; inference remains serialized. Compare off/on at the same slot count.

Outputs match 04–07. 09 adds `setup.csv` for per-image setup/teardown costs, excluded from steady-state timing. See the chapter notes for measurement boundaries.

## Tests

CPU/CUDA preprocessing allows one intensity level (`1/255`) plus floating-point tolerance. Detection outputs may differ slightly; these tests do not measure dataset accuracy.

```bash
ctest --test-dir build/release --output-on-failure
```

Five GPU tests cover preprocessing, pinned memory, postprocessing, maps, and registered input, including boundary cases and cleanup. They require a GPU but no engine. Disable them with `-DBUILD_TESTING=OFF`.

Integration checks require Python, built executables, an engine, and `bus.jpg` / `zidane.jpg`:

```bash
python3 tests/basic_integration.py --bin-dir build/release/src \
  --engine weights/yolo26m.fp16.engine --images images
python3 tests/series_integration.py --bin-dir build/release/src \
  --engine weights/yolo26m.fp16.engine --images images
python3 tests/advanced_integration.py --bin-dir build/release/src \
  --engine weights/yolo26m.fp16.engine --images images
```

Basic covers 01–02; series covers 02–07; advanced covers 08–10 against 03. They check outputs, CSVs, invalid inputs, and scheduling using temporary files. For 11's CPU-only tests, run `.venv-export/bin/python tests/test_grouped_topk.py -v`.

## Current limitations

Supported contract: static YOLO26 detection, batch 1, 640 × 640:

| Tensor | Name | Shape | Type / contents |
| --- | --- | --- | --- |
| Input | `images` | `[1, 3, 640, 640]` | FP32, RGB, pixel values normalized to 0–1 |
| Output | `output0` | `[1, 300, 6]` | FP32, each row is `[x1, y1, x2, y2, score, class_id]` |

Internal FP16 is supported; I/O must stay FP32. Confidence defaults to `0.25`; keep CPU (`main.cpp`) and GPU (`postprocess.cu`) thresholds consistent when changing it.

TensorRT 10+, YOLO11 output, dynamic shapes, larger batches, segmentation, and pose are unsupported.

04/07/10 preload the folder; 07's Graph cache also grows with image count. Graphs require fixed addresses and parameters; 09 additionally requires host registration support. These are bounded-image experiments, not a production serving framework.

Model weights and the two sample images in `images/` come from [Ultralytics](https://github.com/ultralytics/ultralytics).
