# RTX 3060: execution and concurrency experiments 05–09

Historical report: all chapter numbers below are original numbers; see the [current numbering map](../../chapter-numbering.md). Old 07 used pinned staging and one final synchronization, unlike current 02. The archived runner, CSVs, metadata and numbers remain unchanged. Rerun this cohort using all original binaries and output paths from the [source snapshots](../legacy-sources/pre-merge-2026-09-14/README.md), rather than the renumbered active programs.

This is a separate cohort from the earlier [02–04 pinned-memory experiment](../rtx3060-yolo26n-pinned-2026-09-14/README.md). It reruns 02–04 and every 05–09 control on the same engine, retaining improvements and regressions. Chapter numbers are not a performance ranking.

[Learning notes (中文)](../../notes/README.md) · [Pooled statistics](data/summary.json) · [Environment, commands, hashes, run order and per-run statistics](data/metadata.json)

## Conditions and protocol

- GPU: NVIDIA GeForce RTX 3060, 12 GB; driver 525.147.05.
- Linux 6.5.0-rt6, GCC 11.4.0, CUDA Toolkit 11.8.89, TensorRT 8.6.1.6, OpenCV 3.4.20.
- Release build, C++17 / CUDA17, native SM86; exact CMake cache and executable/source SHA-256 values are in metadata. The snapshot includes uncommitted sources, so the recorded Git HEAD alone does not reproduce it.
- YOLO26n FP16 engine, FP32 linear I/O, batch 1, input `images [1,3,640,640]`, output `output0 [1,300,6]`. This NMS-free output still needs confidence filtering, coordinate restoration and box validation.
- Two original Ultralytics sample images: `bus.jpg` (810×1080) and `zidane.jpg` (1280×720). Confidence threshold 0.25; no changes to weights, engine, resolution or detection rules.
- Three rounds. Each configuration uses 200 warmup and 1,000 measured iterations **per image per round**: 2,000 samples/run, 6,000/configuration, 15 configurations, **90,000 measured samples**.
- Round 1 follows the script's configuration order; round 2 rotates by five then reverses; round 3 rotates by ten. These are three different orders, not every permutation. Programs run sequentially, never as concurrent benchmark processes.
- CUDA module loading was unset. This experiment did not change GPU clock/power settings or control desktop background activity. Before/after GPU telemetry is recorded, not continuous profiling.

Each `data/r<round>_<configuration>/` directory contains the original `timings.csv`, `detections.csv`, `stdout.txt`, and, for 05–09, `throughput.csv`. Logs also preserve setup measurements. Annotated images are excluded from the archived benchmark data.

## Results

| Configuration | Mode | Request P50 (ms) | P95 (ms) | P99 (ms) | Measured images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 02 · CPU baseline | Serial | 2.204 | 2.292 | 2.399 | — |
| 03 · CUDA preprocessing | Serial | 1.515 | 1.539 | 1.576 | — |
| 04 · Pinned memory | Serial | 1.569 | 1.608 | 1.626 | — |
| 05 · 1 slot | Serial | 1.568 | 1.589 | 1.629 | 637.5 |
| 05 · 2 slots | 2 in flight | 2.050 | 2.088 | 2.118 | 975.1 |
| 06 · Graph off | Serial | 1.552 | 1.590 | 1.621 | 642.1 |
| 06 · Graph on | Serial | 1.383 | 1.404 | 1.442 | 724.7 |
| 07 · CPU postprocessing | Serial | 1.565 | 1.582 | 1.611 | 640.0 |
| 07 · GPU postprocessing | Serial | 1.575 | 1.599 | 1.630 | 638.4 |
| 08 · Graph off | Serial | 1.560 | 1.586 | 1.614 | 642.1 |
| 08 · Graph on | Serial | 1.392 | 1.410 | 1.447 | 721.1 |
| 09 · 1 slot, Graph off | Serial | 1.572 | 1.599 | 1.745 | 633.9 |
| 09 · 1 slot, Graph on | Serial | 1.384 | 1.419 | 1.450 | 724.7 |
| 09 · 2 slots, Graph off | 2 in flight | 2.248 | 2.304 | 2.427 | 887.3 |
| 09 · 2 slots, Graph on | 2 in flight | 2.310 | 2.354 | 2.374 | 867.4 |

Request latency is from preparing a decoded CPU image to CPU-usable detections. With two slots, it includes time waiting for GPU execution and result consumption **after entering the slot**, not upstream queueing. A higher-throughput configuration can therefore have higher request latency.

02–04 do not record a separate measurement-window wall clock, so their actual throughput is deliberately left blank. It is not reconstructed as `1000 / P50`.

### What the controls show

- **05:** two slots deliver **1.53×** the single-slot throughput (637.5 → 975.1 images/s), while request P50 rises **30.8%** (1.568 → 2.050 ms).
- **06:** Graph reduces serial P50 **10.9%**, or **1.12×** by off/on latency. Both modes already remove 04's intermediate synchronization; that separate change is not attributed to Graph.
- **07:** GPU postprocessing gives no clear end-to-end gain here. Its pooled P50 is **0.6% higher** than the CPU path; small differences are not evidence of statistical significance. CPU postprocessing/unpacking mean drops from 1.434 to 0.313 μs, which does not imply lower total latency.
- **08:** capturing the full GPU chain reduces P50 **10.8%**, or **1.12×**, against its own Graph-off control.
- **09:** Graph helps the single-slot path, but two slots with Graph achieve **2.2% less throughput** than two slots without Graph. Each of the three rounds shows that direction. The default combination reaches 867.4 images/s, below 05's 975.1 images/s.
- The exact reason the combinations differ still needs a profiler timeline; these results alone do not isolate GPU resource contention, launch scheduling or the postprocessing kernel as the cause.

For additional context, this cohort's serial 02 baseline is **2.204 ms**, 03 is **1.515 ms** (1.45×), and 04 is **1.569 ms** (1.40×). Do not use the older README cohort's 02 value to recompute these ratios.

## Setup is not free

Ranges below come from the three runs' logs. 06/08 report setup per image (six observations per configuration); 09 reports setup for the whole two-image set (three observations). These scopes differ.

| Configuration | Setup range (ms) |
| --- | ---: |
| 06 · Graph off / on, per image | 2.456–3.174 / 3.507–4.148 |
| 08 · Graph off / on, per image | 2.516–3.084 / 3.567–4.488 |
| 09 · 1 slot, Graph off / on, whole set | 7.758–8.640 / 9.591–11.511 |
| 09 · 2 slots, Graph off / on, whole set | 14.357–14.603 / 18.791–21.645 |

Setup includes buffer preparation, a normal prime execution and optional capture/instantiation, not engine loading and not only capture. Steady-state timings exclude it. 09 caches one GraphExec per slot per image: two instances for one slot and four for two slots here. Setup time and cache memory grow with the image set; this is not a bounded production video cache.

## Timing and aggregation

- P50/P95/P99 use nearest-rank on all 6,000 individual samples, not an average of per-image or per-round percentiles. All stage means and percentiles remain available in `summary.json`.
- Throughput is `sum(samples) × 1000 / sum(wall_ms)`; it is neither mean(run FPS) nor reciprocal median latency.
- 05/09 measure a whole round-robin request stream, including pipeline fill, drain and in-memory bookkeeping. 06–08 sum the wall-clock measurement windows for each image.
- All steady-state timings exclude image decoding, drawing, file output, logging, initialization and warmup. **Every measured request in 04–09 includes CPU copying into pinned staging.** 02/03 use pageable host buffers.
- `gpu_ms` spans H2D, CUDA preprocessing, TensorRT, optional GPU postprocessing and D2H. It is not isolated TensorRT kernel time; concurrent work or host submission gaps can affect it.
- CUDA event spans and CPU wall-clock intervals have different boundaries: stage columns need not sum to `total_ms`. 05/09 exclude geometry calculation before staging; 06–08 include it. Event-query bookkeeping also differs slightly across versions. Use **same-executable controls** to isolate a change.
- GPU postprocessing returns a fixed-capacity 7,204-byte structure versus the 7,200-byte raw output; this experiment does **not** reduce D2H bandwidth by transferring only valid boxes.

Repeated warm images favor cache reuse. Two images and three short rounds are a teaching microbenchmark, not a full-dataset evaluation, production latency SLO, sustained-throughput guarantee or statistical significance test. Two images with two slots also give a fixed image-to-slot mapping; alternating three-image correctness cases are tested separately.

## Correctness checks

The independently recomputed sample counts, per-stage means/percentiles, actual-wall-clock FPS and 37 file hashes match the recorded results. For all 42 runs of 03–09, each image's **last recorded detection list** is exactly equal to 04, including class, score, coordinates and order: five boxes for bus, three for zidane.

02 differs from the CUDA preprocessing path on bus: one box has left/right coordinates differing by one pixel, and two scores differ (maximum absolute difference 0.001953125). Detection count/class/order and zidane results match. This is consistent with the documented CPU/CUDA interpolation difference, not evidence of unchanged full-dataset mAP.

Validation also includes:

- Strict Debug (`-Wconversion -Wsign-conversion -Werror`) and Release builds; three GPU component tests pass in both.
- The integration script passes in Debug and Release: 13 configurations × three images × three measured iterations, three single-job/two-slot drain cases, invalid arguments, empty/corrupt folders, CSV fields/counts and actual throughput formulas.
- Integration fixtures contain different dimensions, a tiny odd-sized BMP and a comma in a filename. Last detection lists must match exactly; the GPU postprocessing component additionally tests every candidate, stable ordering, NaN/Inf, integer boundaries, count reuse and output guards.
- GPU postprocessing memcheck: 0 errors; racecheck: 0 hazards. Full Graph initcheck: 0 errors. Two-slot Graph pipeline memcheck: 0 errors. Sanitizer timings are not included in the performance data.

The benchmark CSVs save only the last detection list per image, not detections for every one of the 90,000 samples. Full-dataset accuracy and longer-run reliability validation remain separate work.

## Reproduce

From the repository root, with a locally built compatible engine:

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release -j4
ctest --test-dir build/release --output-on-failure
python3 tests/series_integration.py \
  --bin-dir build/release/src --engine weights/yolo26n.fp16.engine --images images

cmake -S docs/benchmarks/legacy-sources/pre-merge-2026-09-14 \
  -B build/legacy-pre-merge -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/legacy-pre-merge -j4

python3 docs/benchmarks/rtx3060-series-05-09-2026-09-14/run.py \
  --bin-dir build/legacy-pre-merge \
  --output results/series-benchmark --iterations 1000 --warmup 200 --rounds 3
```

The runner uses only Python's standard library and executes configurations sequentially. The output directory must be new; it will not overwrite an existing run. Each executable runs in an isolated temporary working directory, so it does not overwrite another configuration's result files. Engine files are hardware/software-specific: rebuilding on another GPU produces a new experiment, not the same engine.
