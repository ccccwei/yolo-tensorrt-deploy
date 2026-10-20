# RTX 3060 · YOLO26n repeated-image benchmark

Measured on 2026-09-14. [English README](../../../README.md) · [中文 README](../../../README.zh-CN.md)

This report uses the original chapter numbers and timing contracts. Current 01 incorporates old 02, and current 02 combines old 03 and 07. Original data and recorded hashes are unchanged; see the [numbering map](../../chapter-numbering.md) and use the [historical source snapshots](../legacy-sources/pre-merge-2026-09-14/README.md), not the new default single-pass command, to interpret the old results.

## Results

All durations are milliseconds. The headline latency is the nearest-rank P50 of all measured `total_ms` samples pooled across five rounds and two images, **not** an average of per-image or per-round percentiles.

| Version | Samples | Mean | P50 | P95 | P99 | Speedup vs 02 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 02_benchmark | 10,000 | 2.226 | 2.208 | 2.341 | 2.458 | 1.00× |
| 03_cuda_preprocess | 10,000 | 1.530 | 1.521 | 1.595 | 1.703 | 1.45× |

Speedup = pooled P50 of 02 / pooled P50 of the version, calculated before rounding. Version 02 measures the baseline CPU path; it is not itself a preprocessing optimization. Version 01 is a single-pass demonstration with no warmup and drawing included in postprocessing, so its timings are excluded.

Per-image Pipeline results (5,000 samples per image and version):

| Version | Image | Mean | P50 | P99 |
| --- | --- | ---: | ---: | ---: |
| 02_benchmark | bus.jpg | 2.252 | 2.259 | 2.457 |
| 02_benchmark | zidane.jpg | 2.200 | 2.183 | 2.462 |
| 03_cuda_preprocess | bus.jpg | 1.524 | 1.519 | 1.626 |
| 03_cuda_preprocess | zidane.jpg | 1.537 | 1.523 | 1.744 |

Stage means:

| Version | Preprocess | Infer | Postprocess |
| --- | ---: | ---: | ---: |
| 02_benchmark | 0.603 | 1.622 | 0.0014 |
| 03_cuda_preprocess | 0.255 | 1.274 | 0.0014 |

In 02, input H2D is included in Infer. In 03, raw-image H2D is included in Preprocess together with the CUDA kernel and stream synchronization. Infer includes output D2H and synchronization in both versions. The individual Infer columns therefore have different boundaries; use Pipeline to compare overall latency. Pipeline P50 is measured directly, not obtained by adding stage percentiles.

## Conditions and scope

- RTX 3060; driver 525.147.05; Linux 6.5.0-rt6; GCC 11.4.0; CUDA 11.8.89; TensorRT 8.6.1; OpenCV 3.4.20.
- Release build for the local GPU (`native`, sm_86); the same YOLO26n FP16 engine with FP32 I/O, batch 1, input 640 × 640.
- Inputs: `bus.jpg` (810 × 1080) and `zidane.jpg` (1280 × 720), processed in path order.
- Five rounds. Each image is decoded once per run, then processed for 200 warmup iterations and 1,000 measured iterations. Each round starts a new process per version, with engine/context/stream/device buffers reused within that process.
- Sequential version order: 02→03, 03→02, 02→03, 03→02, 02→03. Versions were not run concurrently.
- CPU steady-clock durations, with GPU stream synchronization before timing boundaries. Initialization, decoding, warmup, drawing, saving, logging, and CSV writing are excluded.
- Desktop GPU; clocks and power were not locked and background desktop activity was not isolated. Per-run GPU snapshots are included in `summary.json`.
- This is a warmed, repeated-input sample benchmark, not a full-dataset benchmark, application throughput measurement, or accuracy evaluation. The 200-iteration warmup is this sample protocol, not the longer publication protocol in the roadmap.

Every run produced five detections for `bus.jpg` and three for `zidane.jpg`, with the same class/order across 02 and 03. For these samples, the largest coordinate difference was 1 pixel and the largest score difference was 0.001953125. All 14 CPU/CUDA preprocessing test cases passed. These checks do not establish full-dataset mAP equivalence.

## Raw records

- [02 timings](02_benchmark.csv) and [03 timings](03_cuda_preprocess.csv): 10,000 measured rows each, with the round number added to the original timing CSV. No samples were trimmed or discarded.
- [Detections](detections.csv): the last iteration's detections for every image, round, and version.
- [Summary and provenance](summary.json): full-precision statistics, per-round results, run order/timestamps, GPU snapshots, and SHA-256 hashes of sources, executables, engine, and images. Source hashes identify the exact measured implementation.

To recompute a headline P50, sort one version's 10,000 `total_ms` values and select the 5,000th value (one-based). P95 and P99 use the 9,500th and 9,900th values.

## Reproduce

From the repository root, using the same model and input images:

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release
ctest --test-dir build/release --output-on-failure
cmake -S docs/benchmarks/legacy-sources/pre-merge-2026-09-14 \
    -B build/legacy-pre-merge -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/legacy-pre-merge -j

repo="$PWD"
for round in 1 2 3 4 5; do
    mkdir -p "results/benchmark/round_$round"
    (
        cd "results/benchmark/round_$round" || exit 1
        if [ "$((round % 2))" -eq 1 ]; then
            versions="yolo_benchmark yolo_cuda_preprocess"
        else
            versions="yolo_cuda_preprocess yolo_benchmark"
        fi
        for version in $versions; do
            "$repo/build/legacy-pre-merge/$version" \
                "$repo/weights/yolo26n.fp16.engine" "$repo/images" 1000 200 || exit 1
        done
    ) || exit 1
done
```

Each round keeps its own generated `results/02_benchmark/` and `results/03_cuda_preprocess/` folders. Running this command again overwrites those experiment outputs. Engine files are GPU/TensorRT-specific: a locally rebuilt engine may have a different hash and different performance.
