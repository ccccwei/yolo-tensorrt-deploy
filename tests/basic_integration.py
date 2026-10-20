#!/usr/bin/env python3
"""CLI, CSV and exact-mode regression for the merged 01 and 02; stdlib only.

All fixtures and outputs use a temporary directory. This checks default counts,
warmup exclusion, image output and ordered detections, not latency or dataset mAP.
CPU preprocessing is not assumed to be bitwise equal to CUDA preprocessing.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import series_integration as check


VERSIONS = {
    "01": ("yolo_baseline", "01_baseline"),
    "02": ("yolo_cuda_prepost", "02_cuda_prepost"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True,
                        help="YOLO26 FP32 I/O [1,300,6] engine")
    parser.add_argument("--images", type=Path, required=True,
                        help="Directory containing bus.jpg and zidane.jpg")
    args = parser.parse_args()
    binaries, engine, images = (path.resolve() for path in (args.bin_dir, args.engine, args.images))
    check.require(engine.is_file(), f"Missing engine: {engine}")
    for binary, _ in VERSIONS.values():
        check.require((binaries / binary).is_file(), f"Missing executable: {binaries / binary}")
    for name in ("bus.jpg", "zidane.jpg"):
        check.require((images / name).is_file(), f"Missing fixture source: {images / name}")
    check.VERSIONS = VERSIONS

    with tempfile.TemporaryDirectory(prefix="yolo-basic-integration-") as directory:
        root = Path(directory)
        fixtures = root / "images"
        fixtures.mkdir()
        shutil.copyfile(images / "bus.jpg", fixtures / 'bus,"fixture.jpg')
        shutil.copyfile(images / "zidane.jpg", fixtures / "zidane.jpg")
        check.write_blank_bmp(fixtures / "blank.bmp")
        # Unrelated files must not count as input images.
        (fixtures / "ignore.txt").write_text("not an input image\n", encoding="utf-8")
        filenames = ['blank.bmp', 'bus,"fixture.jpg', 'zidane.jpg']

        configs = [
            ("01-default", "01", [], 1, 0),
            ("01-iterations-only", "01", ["2"], 2, 0),
            ("01-repeat-warmup", "01", ["3", "2"], 3, 2),
            ("02-default", "02", [], 200, 50),
            ("02-cpu-post", "02", ["3", "1", "0"], 3, 1),
            ("02-gpu-post", "02", ["3", "1", "1"], 3, 1),
        ]
        references = {}
        for label, version, arguments, iterations, warmup in configs:
            work = root / label
            result = check.run_program(binaries, engine, fixtures, work, version, arguments)
            stdout = (work / "stdout.txt").read_text(encoding="utf-8")
            counts = (f"Images: {len(filenames)} | Iterations per image: {iterations}"
                      f" | Warmup per image: {warmup}")
            check.require(counts in stdout, f"{label}: missing or incorrect configured counts")
            detections = check.validate_results(result, version, filenames, iterations)
            if version in references:
                check.compare_detections(detections, references[version], label)
            else:
                references[version] = detections
            if version == "02":
                other = result.parent / ("cpu_post" if result.name == "gpu_post" else "gpu_post")
                check.require(not other.exists(), f"{label}: wrote results into the other mode")
            check.require(not (result / "throughput.csv").exists(),
                          f"{label}: synchronous chapter unexpectedly changed CSV contract")
            print(f"PASS {label}: {len(filenames) * iterations} measured rows, "
                  f"warmup={warmup} excluded, exact same-version detections and image outputs", flush=True)

        empty, corrupt = root / "empty", root / "corrupt"
        empty.mkdir()
        corrupt.mkdir()
        (corrupt / "broken.jpg").write_bytes(b"not JPEG")
        mixed = root / "mixed"
        mixed.mkdir()
        valid_name = 'bus,"fixture.jpg'
        shutil.copyfile(fixtures / valid_name, mixed / valid_name)
        # Sort the corrupt image first to prove failure does not stop the loop.
        (mixed / "broken.jpg").write_bytes(b"not JPEG")
        for version in VERSIONS:
            invalid = [
                ["0", "0"], ["-1", "0"], ["1", "-1"], ["1x", "0"],
                ["1e309", "0"], ["1", "1e309"], ["2147483648", "0"],
                ["1", "2147483648"], ["1.5", "0"], ["1", "0", "0", "unexpected"],
            ]
            if version == "01":
                invalid += [["1", "0", "0"]]
            else:
                invalid += [["1", "0", flag] for flag in ("2", "-1", "1x", "1e309", "0.5")]
            for index, arguments in enumerate(invalid):
                check.run_program(binaries, engine, fixtures, root / f"bad-{version}-{index}",
                                  version, arguments, success=False)
            for label, folder in (("empty", empty), ("corrupt", corrupt)):
                check.run_program(binaries, engine, folder, root / f"{label}-{version}",
                                  version, ["1", "0"], success=False)
            result = check.run_program(binaries, engine, mixed, root / f"mixed-{version}",
                                       version, ["1", "0"], success=False)
            detections = check.validate_results(result, version, [valid_name], 1)
            check.compare_detections(detections, {valid_name: references[version][valid_name]},
                                     f"mixed-{version}")
            check.require(not (result / "broken.jpg").exists(),
                          f"mixed-{version}: saved an annotated image for corrupt input")
            print(f"PASS {version}: {len(invalid)} invalid argument cases, empty/corrupt folders", flush=True)
            print(f"PASS {version}: mixed folder continues after corrupt input, "
                  "valid-image-only CSV/output, nonzero non-crash exit", flush=True)
    print("All basic integration tests passed. Temporary fixtures and outputs removed.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"FAIL {error}") from error
