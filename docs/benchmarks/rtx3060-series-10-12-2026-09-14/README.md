# RTX 3060 · Versions 10–12 experiments

This is a historical measurement cohort using original chapter numbers. Old 05/06/10/11/12 are now 04/05/08/09/10. Active result directory names have changed, so the frozen runner below must use [archived binaries with the old paths](../legacy-sources/pre-merge-2026-09-14/README.md). Recorded data and hashes remain unchanged; see the [numbering map](../../chapter-numbering.md).

Measured on 2026-09-14. These are separate controlled experiments, not a cumulative optimization stack: 10 and 11 branch from 06's serial CUDA Graph path; 12 compares stream assignment with one TensorRT context.

The main experiment contains **84,000 measured requests across 14 configurations**. Results include unsuccessful or small optimizations, not just the fastest settings.

## Environment and protocol

- NVIDIA GeForce RTX 3060 (12 GB), driver 525.147.05; CUDA 11.8.89; TensorRT 8.6.1.6; OpenCV 3.4.20; GCC 11.4.0.
- Linux 6.5.0-rt6, Release build, CUDA architecture 86. No fast-math change or model/precision change.
- Same YOLO26n FP16 engine with FP32 linear I/O: `[1,3,640,640] → [1,300,6]`, batch 1, confidence threshold 0.25. YOLO26 NMS-free output still undergoes filtering and coordinate restoration.
- Same decoded images: `bus.jpg` (810×1080) and `zidane.jpg` (1280×720). Every measured request uploads pixels; registered input does not skip H2D.
- Three rounds, 200 warmups and 1,000 measured iterations **per image per round**: 6,000 samples per configuration. Programs run sequentially; configuration order rotates by five positions each round and reverses on odd zero-based rounds.
- Request latency runs from CPU staging through completed CPU postprocessing. It excludes engine loading, decoding, per-image setup, warmup, drawing, CSV and image output. Pipeline requests include queueing; throughput includes pipeline fill and drain.
- P50/P95/P99 are nearest-rank percentiles of all 6,000 request samples, not averages of per-run percentiles. FPS is `sum(completed requests) × 1000 / sum(measured wall_ms)`, not `1000 / P50` or an average of run FPS.
- GPU clocks were not locked. Per-run temperatures, power, clocks and utilization are recorded, but desktop activity and thermal variation were not completely controlled. Three repeated-image rounds do not establish statistical significance or production performance.

[Raw per-run data](data/) · [Pooled summary](data/summary.json) · [Environment, commands, run order and SHA-256 hashes](data/metadata.json) · [Runner](run.py).

This cohort includes freshly measured 05/06 references. Do not combine its speedup calculations with the older 01–04 or 05–09 tables.

## Complete request results

All latency columns are milliseconds; FPS is completed images/s.

| Configuration | Mean | P50 | P95 | P99 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 05 · two contexts, two slots (fresh reference) | 2.120 | 2.101 | 2.281 | 2.377 | 942.9 |
| 06 · serial Graph (fresh reference) | 1.433 | 1.420 | 1.531 | 1.622 | 697.5 |
| 10 · original preprocessing | 1.426 | 1.409 | 1.558 | 1.714 | 700.7 |
| 10 · preprocessing maps | 1.421 | 1.404 | 1.560 | 1.679 | 703.3 |
| 11 · pinned staging | 1.419 | 1.404 | 1.546 | 1.648 | 704.6 |
| 11 · registered input | 1.290 | 1.280 | 1.339 | 1.443 | 774.8 |
| 12 · 1 slot, single stream | 1.637 | 1.605 | 1.821 | 1.974 | 610.9 |
| 12 · 1 slot, split streams | 1.657 | 1.621 | 1.858 | 2.010 | 603.4 |
| 12 · 2 slots, single stream | 3.082 | 3.075 | 3.235 | 3.270 | 648.6 |
| 12 · 2 slots, split streams | 2.630 | 2.624 | 2.697 | 2.760 | 759.9 |
| 12 · 3 slots, single stream | 4.583 | 4.573 | 4.775 | 4.861 | 654.1 |
| 12 · 3 slots, split streams | 3.926 | 3.925 | 3.990 | 4.024 | 763.5 |
| 12 · 4 slots, single stream | 6.237 | 6.229 | 6.566 | 6.695 | 640.6 |
| 12 · 4 slots, split streams | 5.251 | 5.248 | 5.347 | 5.402 | 760.9 |

### What the controls show

**10 — precomputed interpolation maps:** P50 decreases only **0.4%**, from 1.409177 to 1.403889 ms. One of three rounds is slightly slower with maps. The average GPU interval decreases by about 6.5 μs, but this does not establish a stable, significant full-request gain. Keep the preprocessing microbenchmark separate from this result.

**11 — registered input:** P50 decreases **8.8%**, from 1.403665 to 1.279959 ms, about **1.10×** latency speedup. Mean CPU staging decreases from 0.121995 to 0.000051 ms. This removes the extra CPU copy into a pinned staging allocation; it is **not GPU zero-copy**, and H2D still runs on every Graph replay. Setup and teardown are excluded above and reported below.

**12 — split transfer streams:** at two slots, FPS increases from 648.594 to 759.927, **1.172×** (17.2%), relative to the same program's single-stream control. Three and four split-stream slots stay near 760 FPS while P50 grows to 3.925 and 5.248 ms. One slot has no cross-request overlap opportunity and is slightly slower when split. The default remains two slots.

12 is **not the throughput winner**: fresh 05 with two contexts reaches 942.882 FPS. 12 keeps only one context and serializes all TensorRT inference on one compute stream; the experiment explores transfer scheduling with a different resource structure. At these image sizes each slot adds 2,772,000 bytes of pinned buffers and 7,687,200 bytes of device buffers, excluding engine/context/driver allocations.

For 12, `gpu_ms` is a start-to-done event span across dependencies and may contain waiting behind other requests. It is not isolated TensorRT execution time and does not necessarily sum with CPU stage times to `total_ms`. No usable Nsight Systems timeline was obtained on this machine; throughput results alone do not verify a particular amount of hardware overlap.

## 10: preprocessing-only microbenchmark

[Source](../../../tests/preprocess_maps_benchmark.cpp) · [All 18 per-round CSV rows](preprocess_micro.csv).

Each shape/mode captures **100 consecutive preprocessing calls** in a Graph, warms up 20 Graph launches, then measures 100 Graph launches with CUDA events and divides each interval by 100. Three rounds alternate mode order: off/on, on/off, off/on.

Each shape/mode therefore has 300 event-interval samples containing 30,000 measured calls; all six combinations contain **180,000 calls**. The following values average the three equal-count means; they are not pooled P50 values.

| Source size | Original mean (μs/call) | Maps mean (μs/call) | Microbenchmark speedup |
| --- | ---: | ---: | ---: |
| 810×1080 | 30.063 | 24.638 | 1.22× |
| 1280×720 | 25.667 | 24.063 | 1.07× |
| 640×640 | 36.686 | 21.029 | 1.74× |

These amortized intervals include Graph node scheduling gaps and possible cache reuse. Synthetic input remains resident on the GPU; H2D, D2H, map generation, Graph construction and model inference are excluded. The square-image 1.74× result is **not** a full inference speedup.

Per-image setup in the full experiment averages **3.963 ms without maps** and **4.175 ms with maps** (ranges 3.638–4.211 and 3.661–4.771 ms). Setup includes resource preparation, prime and Graph creation; this is not an isolated map-build timing.

## 11: setup, teardown and short runs

Six per-image observations per configuration in the main cohort:

| Configuration | Setup mean / range (ms) | Teardown mean / range (ms) |
| --- | --- | --- |
| Pinned staging | 4.128 / 3.487–4.881 | 0.217 / 0.160–0.300 |
| Registered input | 3.065 / 2.661–3.879 | 0.452 / 0.384–0.528 |

Setup includes allocation or registration, a normal prime execution, and Graph capture/instantiation. Teardown includes synchronization and Graph destruction, plus unregistering/releasing the source in the registered path. These are not isolated measurements of `cudaHostRegister` and `cudaHostUnregister`; the registered path's total setup was actually lower here.

Additional three-round experiments use no warmup and only 1 or 10 measured requests per image. Prime and Graph construction still happen during setup. The metric below is:

`(sum(formal-loop wall_ms) + sum(setup_ms) + sum(teardown_ms)) / completed requests`

| Requests per image | Total requests per configuration | Staging mean (ms) | Registered mean (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 6 | 5.562 | 4.748 |
| 10 | 60 | 1.895 | 1.715 |

[1-request runs](short_1/) · [1-request summary](short_1/summary.json) · [10-request runs](short_10/) · [10-request summary](short_10/summary.json).

Both short cases are faster with registration on this machine, but their sample counts are small. They do not justify “registration always wins” or “one-shot registration always loses.” These averages include per-image setup/teardown but exclude engine loading, decoding and output I/O: they are **not process cold-start times**, nor directly comparable to steady-state P50. A camera SDK, external buffer owner or different allocator needs its own lifetime checks and measurements.

## Correctness and validation

- All targets build in Release and strict Debug, including C++ `-Wconversion`, `-Wsign-conversion` and Debug `-Werror`.
- All five CTest component tests pass in Debug and Release. New map tests cover 17 geometry/layout cases, three changing inputs each, Graph replay, pitched memory, ROI, extreme aspect ratios and guard regions. New registered-input tests cover eight layouts, changed pixels on Graph replays, retained source ownership, registration state, and normal/exception cleanup.
- [Advanced integration](../../../tests/advanced_integration.py) passes in Debug and Release: 13 configurations using three fixture images, plus ten partial-ring/single-job pipeline-drain cases and invalid-input checks. The existing 04–09 integration regression also passes in Release.
- Compute Sanitizer memcheck reports **0 errors** for the map component test, registered-input component test, and a four-slot split-stream inference run. Instrumented runs are not included in any performance table.
- All 42 main-run detection CSVs are byte-for-byte equal to the fresh 06 reference (and 05 reference): five bus detections and three zidane detections, including class, confidence, coordinates and order. These CSVs contain the **last recorded detection list per image**, not detections for every one of the 84,000 timed requests. Component tests separately verify updated-pixel tensors exactly.
- An independent recomputation checked sample IDs/counts, all reported stage statistics, wall-time FPS, saved detections and all 50 recorded source/model/image/binary hashes for the main experiment and both short-run archives.

These are sample regressions and memory checks, **not a full-dataset mAP evaluation**, cross-device equivalence proof, or evidence of statistical significance.

## Reproduce

From the repository root, with the same engine already built and images available:

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release -j
ctest --test-dir build/release --output-on-failure

python3 tests/advanced_integration.py \
  --bin-dir build/release/src \
  --engine weights/yolo26n.fp16.engine --images images

cmake -S docs/benchmarks/legacy-sources/pre-merge-2026-09-14 \
  -B build/legacy-pre-merge -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/legacy-pre-merge -j

python3 docs/benchmarks/rtx3060-series-10-12-2026-09-14/run.py \
  --bin-dir build/legacy-pre-merge \
  --output results/advanced-benchmark --iterations 1000 --warmup 200 --rounds 3

python3 docs/benchmarks/rtx3060-series-10-12-2026-09-14/run.py \
  --bin-dir build/legacy-pre-merge \
  --configs 11_r0 11_r1 --iterations 1 --warmup 0 --rounds 3 \
  --output results/registered-short-1

python3 docs/benchmarks/rtx3060-series-10-12-2026-09-14/run.py \
  --bin-dir build/legacy-pre-merge \
  --configs 11_r0 11_r1 --iterations 10 --warmup 0 --rounds 3 \
  --output results/registered-short-10

./build/release/src/preprocess_maps_benchmark > results/preprocess-micro.csv
```

The runner requires a **new output directory** and runs configurations sequentially in isolated working directories. It preserves raw CSVs, program logs, commands, hashes and pooled summaries. Do not run multiple GPU benchmarks simultaneously. Building a new engine on another GPU changes the experiment; record its hash and remeasure.

Current reading notes (Chinese): [08 — maps](../../notes/08_preprocess_maps.md) · [09 — registered input](../../notes/09_registered_input.md) · [10 — split streams](../../notes/10_split_streams.md). These correspond to old 10/11/12 in this report.
