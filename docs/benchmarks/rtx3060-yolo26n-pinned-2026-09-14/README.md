# RTX 3060 · Pinned memory comparison

Measured on 2026-09-14. [English README](../../../README.md) · [中文 README](../../../README.zh-CN.md)

Names and timing contracts below describe the original implementations, not new measurements of the merged chapters. Current 01 incorporates old 02, current 02 combines old 03 and 07, and current 03 is pinned memory. [Historical snapshots](../legacy-sources/pre-merge-2026-09-14/README.md) preserve the old sources and output paths; original CSVs and hashes remain unchanged. See the [numbering map](../../chapter-numbering.md).

## Result: no latency improvement over 03 in this sample

All durations are milliseconds. Each version has 12,000 measured samples, pooled across six rounds and two images. Percentiles use nearest rank over the original `total_ms` values; no samples were trimmed or discarded.

| Version | Mean | Pipeline P50 | P95 | P99 | Speedup vs 02 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 02_benchmark | 2.223 | 2.215 | 2.299 | 2.374 | 1.00× |
| 03_cuda_preprocess | 1.518 | 1.517 | 1.536 | 1.574 | 1.46× |
| 04_pinned_memory | 1.556 | 1.567 | 1.586 | 1.618 | 1.41× |

04 is **0.97×** as fast as 03 by this P50 definition: its median latency is **3.3% higher**, not lower. The input CPU staging copy is included. This does not imply that pinned memory is always slower; it shows no overall latency benefit for this synchronous repeated-image implementation on this machine. Stage timings do not isolate DMA bandwidth from staging/kernel/synchronization costs.

Version 01 is not benchmarked: it is a single-pass demo without warmup and includes drawing in postprocessing. Version 02 measures the baseline CPU path; warmup itself is not an optimization.

This is a fresh, same-session comparison of 02, 03, and 04. The [earlier 02/03 benchmark](../rtx3060-yolo26n-2026-09-14/README.md) is retained separately; its numbers were not mixed into this result.

## What changed in 04

- The CUDA kernel is byte-identical to 03, and the detection algorithm is unchanged.
- Raw-image staging and detection output use reusable `cudaMallocHost` allocations, freed with `cudaFreeHost`.
- Every measured preprocessing call copies the ordinary decoded image into pinned staging, then performs H2D, the CUDA kernel, and stream synchronization. Contiguous images use one CPU copy; non-contiguous images are copied row by row.
- Output D2H writes directly to the pinned output buffer, which postprocessing reads without an intermediate vector copy.
- Pinned/GPU allocations and capacity growth occur outside warmup and measurement. This is not an allocation-free pipeline: postprocessing still constructs a detection vector.
- There is no cross-image overlap, double buffering, or CUDA Graph. Page-locked memory is a transfer-buffer change, not a complete asynchronous pipeline.

[NVIDIA's page-locked memory guidance](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#page-locked-host-memory) describes its transfer uses and resource costs. Do not allocate or pin buffers on every inference simply to enable asynchronous copies.

Stage means:

| Version | Preprocess | Infer | Postprocess |
| --- | ---: | ---: | ---: |
| 02_benchmark | 0.610 | 1.611 | 0.0014 |
| 03_cuda_preprocess | 0.242 | 1.275 | 0.0013 |
| 04_pinned_memory | 0.285 | 1.270 | 0.0013 |

02 includes input H2D in Infer. 03 includes raw-image H2D in Preprocess. 04 includes CPU staging plus raw-image H2D in Preprocess. All three include stream synchronization and output D2H. Compare Pipeline, not differently bounded Infer columns. The Pipeline P50 is measured directly, not obtained by adding stage percentiles.

Per-image Pipeline results (6,000 samples per image and version):

| Version | Image | P50 | P99 |
| --- | --- | ---: | ---: |
| 02_benchmark | bus.jpg | 2.266 | 2.389 |
| 02_benchmark | zidane.jpg | 2.184 | 2.321 |
| 03_cuda_preprocess | bus.jpg | 1.517 | 1.574 |
| 03_cuda_preprocess | zidane.jpg | 1.517 | 1.574 |
| 04_pinned_memory | bus.jpg | 1.561 | 1.615 |
| 04_pinned_memory | zidane.jpg | 1.569 | 1.621 |

## Protocol and checks

- RTX 3060; driver 525.147.05; Linux 6.5.0-rt6; GCC 11.4.0; CUDA 11.8.89; TensorRT 8.6.1; OpenCV 3.4.20.
- Release build, local GPU architecture (`native`, sm_86); same YOLO26n FP16 engine with FP32 I/O, batch 1, input 640 × 640.
- `bus.jpg` (810 × 1080) and `zidane.jpg` (1280 × 720), decoded once per process in path order.
- Six rounds; every image in each round gets 200 warmup iterations and 1,000 measured iterations. A new process starts per version and round; resources are reused within it.
- Each of the six possible version orders is used once: 02→03→04, 03→04→02, 04→02→03, 04→03→02, 03→02→04, 02→04→03. Each version appears twice in each position. Runs are sequential.
- CPU steady-clock measurement includes preprocessing, transfers, inference, and postprocessing, excluding initialization, image decoding, warmup, drawing, file I/O, and logging.
- Desktop GPU; clocks/power were not locked and background desktop activity was not isolated. Per-run GPU snapshots are in `summary.json`.
- This is a warmed repeated-input sample test, not a full-dataset accuracy/throughput benchmark. The 200-iteration warmup is this sample protocol, not the longer publication protocol in the roadmap.

03 and 04 produced **identical detection CSV rows in all six rounds**: five detections for bus, three for zidane. Compared with 02, maximum box-coordinate difference was 1 pixel and maximum score difference was 0.001953125.

Debug and Release builds passed both CTest entries. The 14-case CPU/CUDA reference test passed; the pinned-memory test covered eight image/ROI cases with changed pixels on a second pass, allocation attributes, buffer growth/reuse, transfer equivalence, and invalid inputs. Compute Sanitizer reported zero errors for the pinned-memory test. These checks do not establish full-dataset mAP equivalence.

## Raw records and reproduction

[02 timings](02_benchmark.csv) · [03 timings](03_cuda_preprocess.csv) · [04 timings](04_pinned_memory.csv) · [Detections](detections.csv) · [Summary, source/model hashes, and run metadata](summary.json)

To recompute a headline P50, sort one version's 12,000 `total_ms` values and take the 6,000th value (one-based). P95 and P99 use the 11,400th and 11,880th values. Speedup uses unrounded percentiles.

Run from the repository root with the same engine and images:

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release
ctest --test-dir build/release --output-on-failure
cmake -S docs/benchmarks/legacy-sources/pre-merge-2026-09-14 \
    -B build/legacy-pre-merge -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/legacy-pre-merge -j

repo="$PWD"
round=0
for order in "02 03 04" "03 04 02" "04 02 03" "04 03 02" "03 02 04" "02 04 03"; do
    round=$((round + 1))
    mkdir -p "results/pinned-benchmark/round_$round"
    (
        cd "results/pinned-benchmark/round_$round" || exit 1
        for version in $order; do
            case "$version" in
                02) binary="$repo/build/legacy-pre-merge/yolo_benchmark" ;;
                03) binary="$repo/build/legacy-pre-merge/yolo_cuda_preprocess" ;;
                04) binary="$repo/build/legacy-pre-merge/yolo_pinned_memory" ;;
            esac
            "$binary" \
                "$repo/weights/yolo26n.fp16.engine" "$repo/images" 1000 200 || exit 1
        done
    ) || exit 1
done
```

Each round saves independent per-version output folders. Re-running overwrites those experiment outputs. Engine files are GPU/TensorRT-specific; rebuilding may change tactics, the engine hash, and performance.
