# YOLO26n Grouped TopK — RTX 3060

Grouped selection reduced **engine GPU P50 by 3.95%** and **pipeline P50 by 3.47%** in this experiment. A separate identical-cache-copy check reproduced a **4.25% engine GPU reduction**. This is a small model-graph improvement, not a new C++ runtime or a claim of dataset-wide accuracy equivalence.

[中文原理与实验笔记（当前第 11 篇）](../../notes/11_grouped_topk.md) · [Graph rewrite](../../../tools/grouped_topk.py) · [CPU tests](../../../tests/test_grouped_topk.py)

This historical report retains its original chapter labels: Grouped TopK was 13, and the registered-input runtime was 11 (now 09). See the [numbering map](../../chapter-numbering.md). The frozen runner expects the old result path and therefore uses the archived runtime in the reproduction commands below; raw measurements, scripts and hashes are unchanged.

## Change under test

Only the two detection-tail TopK nodes change. Each selection is split into eight contiguous groups; each group retains 300 entries, then a second selection chooses the global 300 and maps indices back. The two selection widths are 8,400 and 24,000. Keeping 300 **per group**, rather than dividing the quota between groups, preserves the exact top-score selection.

This follows the grouped-selection idea in Ultralytics' [`Detect._grouped_topk`](https://docs.ultralytics.com/reference/nn/modules/head/#ultralytics.nn.modules.head.Detect._grouped_topk). The installed exporter is 8.4.137: its static `format="engine"` path selects eight groups, while the existing `format="onnx"` export uses one. Instead of re-exporting the whole model, the experiment rewrites the existing ONNX to isolate this change.

The graph grows from 445 to 467 nodes, and from two to four TopK operators. All 443 unrelated nodes, all 204 initializers, their order, model IO and metadata remain byte-identical. No retraining, NMS, custom plugin, reduced output count or weight change is involved. The original ONNX and default engine are untouched.

## Environment and timing boundaries

- RTX 3060 12 GB; NVIDIA driver 525.147.05; CUDA 11.8; TensorRT 8.6.1.6; C++ OpenCV 3.4.20.
- Static batch 1, FP32 linear IO: `images[1,3,640,640] → output0[1,300,6]`; FP16-enabled engine build, workspace 1,024 MiB, builder optimization level 3.
- Two images, `bus.jpg` and `zidane.jpg`, repeated after warmup. Engine, input, runtime and runner hashes, plus the commands, are in [metadata](data/metadata.json); model hashes are in the [structure audit](evidence/engine_structure.json). No GPU clock or power setting was changed; per-run telemetry is retained.
- **Engine GPU:** `trtexec`, one inference stream, CUDA Graph, 500 ms warmup, 1,000 measured iterations, `--duration=0 --noDataTransfers`. The true preprocessed tensor is uploaded before timing. Reported times are GPU computation, not host request latency; `trtexec` can queue successive requests.
- **Pipeline:** [original 11 registered-input runtime](../legacy-sources/pre-merge-2026-09-14/src/11_registered_input/main.cpp), Graph enabled, `REGISTERED=1`, 200 warmups then 1,000 measurements per image. Wall-clock request latency includes staging/geometry, H2D, CUDA preprocessing, TensorRT, D2H, synchronization and CPU postprocessing. It excludes image decoding, registration/Graph setup, teardown, drawing and file IO. This is a repeated decoded-image workload, not a live-camera test with registration on every frame.
- Formal measurements have no layer profiler. FPS is measured samples divided by summed loop wall time, including loop/statistics overhead; it is not `1000 / P50`.

## Main experiment: 48,000 measurements

Both graphs were built twice, in order **control A → grouped A → grouped B → control B**, using a sequentially updated shared timing cache. Three timing rounds alternate order; exact order and commands are saved by [run.py](run.py). Each engine has 6,000 engine-GPU and 6,000 pipeline measurements.

| Build | Engine GPU P50 / P95 (ms) | Pipeline P50 / P95 (ms) | Actual pipeline FPS |
| --- | ---: | ---: | ---: |
| Control A | 1.077270 / 1.087400 | 1.281318 / 1.326529 | 777.98 |
| Grouped A | 1.039920 / 1.045840 | 1.237226 / 1.289294 | 802.07 |
| Grouped B | 1.039000 / 1.042180 | 1.238582 / 1.289965 | 800.74 |
| Control B | 1.085330 / 1.091310 | 1.283227 / 1.337706 | 772.90 |

The following statistics merge **individual samples**, not medians, across the two builds. Each cell uses 12,000 measurements; percentiles use nearest rank.

| Metric | Control | Grouped | Change |
| --- | ---: | ---: | ---: |
| Engine GPU P50 | 1.082030 ms | 1.039310 ms | −3.95%; 1.041× |
| Engine GPU P95 | 1.090700 ms | 1.045170 ms | −4.17% |
| Pipeline P50 | 1.282466 ms | 1.237978 ms | −3.47%; 1.036× |
| Pipeline P95 | 1.332550 ms | 1.289677 ms | −3.22% |
| Actual pipeline throughput | 775.43 FPS | 801.40 FPS | 1.033× |

All three timing rounds separately favor grouped selection. [Per-build statistics](data/summary.json), [pooled statistics](data/pooled_summary.json), [independent audit](data/audit.json), compressed raw GPU timings, pipeline CSVs and complete logs are retained. [summarize.py](summarize.py) independently recomputes the statistics and checks sample IDs, output equality and hashes.

Repeated timings of two images are not 12,000 independent scenes. These results establish a repeatable local improvement, not a confidence interval, a universal speedup or a replacement for previous chapters' benchmark cohorts.

## Build variability and an additional control

The first control has 223 Inspector layer records. The second control and both grouped builds have **identical 216-record public Inspector JSON**. Seven explicit tail-input/output Reformat records disappear even when rebuilding the original graph. Therefore their removal **cannot be credited specifically to Grouped TopK**. The other 215 records and public tactic fields are unchanged. [Structure audit](evidence/engine_structure.json)

The main timing cache evolved between builds; it reduces some build work but does not freeze all compiler decisions. A follow-up therefore starts each graph from its own copy of the **same populated cache bytes**, with otherwise identical build flags. [check_cache.py](check_cache.py) records cache-before/after hashes, engine hashes and commands.

Both follow-up engines expose identical 216-record Inspector JSON. On the same bus input, two rounds run in C/G then G/C order, 1,000 measurements per run:

| Separate fixed-cache check | Control | Grouped |
| --- | ---: | ---: |
| Engine GPU P50 | 1.076050 ms | 1.030330 ms |
| Engine GPU P95 | 1.081850 ms | 1.032470 ms |
| Measurements per graph | 2,000 | 2,000 |

The P50 reduction is **4.25%**. These 4,000 measurements are archived separately and **not pooled into the main 48,000**. [Follow-up statistics](fixed-cache/summary.json) · [Provenance](fixed-cache/metadata.json)

## Where the time changed

Separate instrumented runs of the fixed-cache engines measured the fused detection-tail `ForeignNode[/model.23/Reshape_3.../model.23/Concat_4]` at **0.131632 → 0.0870816 ms average**. This supports locating the improvement inside the compiled detection tail, even though its public layer name and interface are identical.

This is not proof of an individual TopK kernel's speed or a 34% whole-model gain: Inspector does not reveal the entire fused implementation, and layer profiling adds overhead. The two profile files contain 621/624 recorded calls including profiling warmup/initialization, not exactly 500 steady-state samples. Their timing sums must not be substituted for normal GPU latency. The logs include the expected capture-time layer-timing warning, but no failed previous-Graph-launch warning. [Profile evidence inventory](fixed-cache/manifest.json)

## Correctness checks and limits

1. Five CPU test methods cover 18 selection cases at the two actual widths, including concentrated winners, ties, FP16-rounded ties, thresholds, group boundaries and negative scores; additional checks cover minimum group size, invalid contracts, byte preservation and overwrite refusal.
2. Four extracted-selector cases run on TensorRT: unique scores, FP16-rounded ties, more than 300 high-confidence tied positions, and multi-class/0.25-threshold boundaries. These isolated engines use **FP32**, including scores rounded through FP16 where requested. Exact global top-score values, legal/unique position-class pairs and tie handling are checked. All four control/grouped raw outputs match byte-for-byte. [Cases and commands](evidence/selector_gpu.json)
3. **All six complete FP16-enabled engines** produce byte-identical full `[1,300,6]` FP32 output on 21 fixtures: original/flip/scale/brightness/crop/tiled versions of both images, three constant images, noise and a checkerboard. Every image also passes three deterministic Graph replays. The saved CPU detection CSVs match as well. [Main regression](evidence/comparison.json) · [Follow-up regression](fixed-cache/raw_comparison.json)
4. All final outputs saved during the main timing runs agree: 24 engine JSON outputs compared per image, and 12 pipeline detection CSVs. Full-precision binary checks above are stronger than these serialized text-output checks; timing runs do not archive every iteration's full tensor.

No labeled validation dataset or mAP evaluation was run. Other models, shapes, precision modes or TensorRT versions can differ, particularly in which tied candidates survive. Keep the original model available and evaluate the intended dataset before adopting the replacement.

## Reproduce

From the repository root, use a **new** experiment directory. Existing export dependencies and the release build are required; adapt installation paths to the target machine.

The current preprocessing object path below uses new chapter 02; its CUDA preprocessing source is byte-identical to the original 03 kernel. The historical 11 runtime is built from the snapshot so its output paths still match the frozen runner.

```bash
report=docs/benchmarks/rtx3060-grouped-topk-2026-09-14
cmake -S docs/benchmarks/legacy-sources/pre-merge-2026-09-14 \
    -B build/legacy-pre-merge -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/legacy-pre-merge --target yolo_registered_input -j
mkdir -p artifacts
experiment="$(mktemp -d "$PWD/artifacts/grouped-topk-XXXXXX")"
.venv-export/bin/python tools/grouped_topk.py \
    weights/yolo26n.onnx "$experiment/grouped8.onnx" --groups 8
.venv-export/bin/python tests/test_grouped_topk.py -v

# Restore the original build helper into its intended artifacts/<run>/ location.
cp "$report/evidence/scripts/build_one.py" "$experiment/build_one.py"
.venv-export/bin/python "$experiment/build_one.py" weights/yolo26n.onnx control_a
.venv-export/bin/python "$experiment/build_one.py" "$experiment/grouped8.onnx" grouped_a
.venv-export/bin/python "$experiment/build_one.py" "$experiment/grouped8.onnx" grouped_b
.venv-export/bin/python "$experiment/build_one.py" weights/yolo26n.onnx control_b

g++ -std=c++17 -O2 "$report/prepare_input.cpp" \
    build/release/src/CMakeFiles/yolo_cuda_prepost.dir/02_cuda_prepost/preprocess.cu.o \
    -isystem /usr/local/include -isystem /usr/local/cuda-11.8/include \
    -L/usr/local/lib -lopencv_imgcodecs -lopencv_core \
    -L/usr/local/cuda-11.8/lib64 -lcudart \
    -Wl,-rpath,/usr/local/cuda-11.8/lib64 -o "$experiment/prepare_input"
mkdir "$experiment/inputs"
"$experiment/prepare_input" images/bus.jpg "$experiment/inputs/bus.bin"
"$experiment/prepare_input" images/zidane.jpg "$experiment/inputs/zidane.bin"

.venv-export/bin/python "$report/run.py" --engines \
    "$experiment/control_a.engine" "$experiment/grouped_a.engine" \
    "$experiment/grouped_b.engine" "$experiment/control_b.engine" \
    --inputs "$experiment/inputs" --output "$experiment/data" \
    --runtime build/legacy-pre-merge/yolo_registered_input

.venv-export/bin/python "$report/check_cache.py" \
    --control weights/yolo26n.onnx --grouped "$experiment/grouped8.onnx" \
    --cache "$experiment/shared.cache" --input "$experiment/inputs/bus.bin" \
    --output "$experiment/fixed_cache"
```

[Evidence manifest](evidence/manifest.json) lists original source and stored hashes. Its six scripts are byte-preserved historical experiment helpers, not relocated standalone tools; several depend on their original `artifacts/<run>/` position or recorded paths. Raw output archives, fixture recipes and selector references support inspection without committing models, engines, input tensors or image copies. The reusable graph rewrite and CPU/selector test generators live in `tools/` and `tests/`; no extra inference executable is required.
