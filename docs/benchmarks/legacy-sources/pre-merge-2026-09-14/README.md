# Source snapshots before chapter consolidation

These are byte-preserved source snapshots for historical benchmark interpretation, not additional active chapters. The current series combines old 01+02 into new 01 and old 03+07 into new 02, then uses consecutive chapter numbers 01–11. See the [complete numbering map](../../../chapter-numbering.md).

| Historical implementation | Current reading entry | Important difference |
| --- | --- | --- |
| [01 baseline](src/01_baseline/main.cpp) | [01 deployment and benchmarking](../../../notes/01_baseline.md) | The old single-pass demo included drawing in postprocessing time. |
| [02 benchmark](src/02_benchmark/main.cpp) | [01 deployment and benchmarking](../../../notes/01_baseline.md) | Its measurement loop is retained, but the current default is one iteration, zero warmups. |
| [03 CUDA preprocessing](src/03_cuda_preprocess/) | [02 CUDA preprocessing/postprocessing](../../../notes/02_cuda_prepost.md) | Current 02 adds an optional GPU postprocessor; CPU postprocessing remains the default. |
| [07 GPU postprocessing](src/07_gpu_postprocess/) | [02 CUDA preprocessing/postprocessing](../../../notes/02_cuda_prepost.md) | Old 07 used pinned staging and one final wait; current 02 does not inherit those changes. |

[Historical 07 notes](07_gpu_postprocess.md) retain that implementation's explanation. Old CSV labels, timings, engine hashes, recorded commands and source hashes remain historical evidence: they have not been renamed into new measurements. This snapshot also retains original 04/05/06/08/09/10/11/12 sources with their original result paths, allowing historical runners to keep working. All 29 source files preserve their pre-renumbering bytes; see [SHA256SUMS](SHA256SUMS).

From this directory, `sha256sum -c SHA256SUMS` verifies the snapshots. To build all twelve historical executables separately from the active project, run from the repository root:

```bash
cmake -S docs/benchmarks/legacy-sources/pre-merge-2026-09-14 \
    -B build/legacy-pre-merge -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native \
    -DTensorRT_ROOT=/usr/local/TensorRT
cmake --build build/legacy-pre-merge -j
```

Executables are placed directly under `build/legacy-pre-merge/`, using the old names and old result directory names. Supply `--bin-dir build/legacy-pre-merge` to historical benchmark scripts, or `--runtime build/legacy-pre-merge/yolo_registered_input` for the Grouped TopK runner. Adapt other hard-coded paths in a separate copy when necessary. Engine, input, precision and measurement conditions must still match the report. Rebuilding this source is not a promise of identical engine tactics, GPU clocks or absolute timing.

For new work use the root project's current targets. No compatibility executables with retired names are built by the active CMake project.
