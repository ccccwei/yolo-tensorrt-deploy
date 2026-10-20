#!/usr/bin/env python3
"""Small GPU integration regression: independent 02–07 chapters, stdlib only.

Run from any directory. Models and executables are supplied by the caller; all
fixtures and program outputs live in a temporary directory. This checks results
and CSV contracts, not benchmark performance or dataset-level detection accuracy.
"""

import argparse
import csv
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


VERSIONS = {
    "02": ("yolo_cuda_prepost", "02_cuda_prepost"),
    "03": ("yolo_pinned_memory", "03_pinned_memory"),
    "04": ("yolo_async_pipeline", "04_async_pipeline"),
    "05": ("yolo_cuda_graph", "05_cuda_graph"),
    "06": ("yolo_full_graph", "06_full_graph"),
    "07": ("yolo_graph_pipeline", "07_graph_pipeline"),
}
CONFIGS = [
    ("03-baseline", "03", []),
    ("04-slots1", "04", ["1"]), ("04-slots2", "04", ["2"]),
    ("05-graph0", "05", ["0"]), ("05-graph1", "05", ["1"]),
    ("02-post0", "02", ["0"]), ("02-post1", "02", ["1"]),
    ("06-graph0", "06", ["0"]), ("06-graph1", "06", ["1"]),
    ("07-slots1-graph0", "07", ["1", "0"]),
    ("07-slots1-graph1", "07", ["1", "1"]),
    ("07-slots2-graph0", "07", ["2", "0"]),
    ("07-slots2-graph1", "07", ["2", "1"]),
]
TIMING_FIELDS = ["staging_ms", "gpu_ms", "postprocess_ms", "total_ms"]
BASELINE_FIELDS = ["preprocess_ms", "infer_ms", "postprocess_ms", "total_ms"]
DETECTION_FIELDS = ["image", "class_id", "score", "left", "top", "right", "bottom"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_blank_bmp(path):
    """31 × 19 BGR image: exercises odd geometry and BMP row padding."""
    width, height = 31, 19
    stride = (width * 3 + 3) & ~3
    row = bytes([114, 114, 114]) * width + bytes(stride - width * 3)
    pixels = row * height
    header = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 0, 0, 0, 0)
    path.write_bytes(header + dib + pixels)


def read_csv(path, fields):
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames == fields, f"Unexpected CSV header in {path}: {reader.fieldnames}")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"Malformed CSV row in {path}")
    return rows


def number(value, label, positive=False):
    parsed = float(value)
    require(math.isfinite(parsed) and (parsed > 0 if positive else parsed >= 0),
            f"Invalid {label}: {value}")
    return parsed


def validate_results(result, version, filenames, iterations):
    fields = BASELINE_FIELDS if version in ("01", "02", "03") else TIMING_FIELDS
    timings = read_csv(result / "timings.csv", ["image", "iteration"] + fields)
    expected = {(name, iteration) for name in filenames for iteration in range(1, iterations + 1)}
    actual = [(row["image"], int(row["iteration"])) for row in timings]
    require(len(actual) == len(expected) and set(actual) == expected,
            f"Missing, duplicate or unexpected timing rows in {result}")
    for row in timings:
        for field in fields:
            number(row[field], field, positive=field == "total_ms")

    detections = {name: [] for name in filenames}
    for row in read_csv(result / "detections.csv", DETECTION_FIELDS):
        name = row["image"]
        require(name in detections, f"Unexpected detection filename: {name}")
        class_id = int(row["class_id"])
        score = number(row["score"], "score")
        left, top, right, bottom = (int(row[field]) for field in DETECTION_FIELDS[3:])
        require(class_id >= 0 and score >= 0.25 and 0 <= left < right and 0 <= top < bottom,
                f"Invalid detection: {row}")
        detections[name].append((class_id, score, left, top, right, bottom))
    for name in filenames:
        annotated = result / name
        require(annotated.is_file() and annotated.stat().st_size > 0,
                f"Missing annotated image: {annotated}")

    if version not in ("01", "02", "03"):
        rows = read_csv(result / "throughput.csv", ["samples", "wall_ms", "fps"])
        require(len(rows) == 1, f"Expected one throughput summary in {result}")
        row = rows[0]
        samples = int(row["samples"])
        wall_ms = number(row["wall_ms"], "wall_ms", positive=True)
        fps = number(row["fps"], "fps", positive=True)
        require(samples == len(expected), f"Throughput sample count mismatch in {result}")
        require(math.isclose(fps, samples * 1000.0 / wall_ms, rel_tol=1e-7),
                f"FPS must be samples / measured wall time in {result}")
    return detections


def compare_detections(actual, expected, label):
    require(actual.keys() == expected.keys(), f"{label}: image sets differ")
    for name, wanted in expected.items():
        got = actual[name]
        require(len(got) == len(wanted),
                f"{label}: {name}: expected {len(wanted)} detections, got {len(got)}")
        for index, (left, right) in enumerate(zip(got, wanted)):
            require(left == right,
                    f"{label}: {name}: ordered detection {index} differs: {left} != {right}")


def run_program(bin_dir, engine, folder, work, version, arguments, success=True):
    work.mkdir()
    binary, output = VERSIONS[version]
    command = [str(bin_dir / binary), str(engine), str(folder), *arguments]
    completed = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=180, check=False)
    (work / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    # A signal or crash is not a successful negative-argument test.
    passed = completed.returncode == 0 if success else completed.returncode > 0
    require(passed, f"Unexpected exit {completed.returncode}: {' '.join(command)}\n{completed.stdout[-8000:]}")
    result = work / "results" / output
    if version == "02":
        gpu_post = len(arguments) >= 3 and arguments[2] == "1"
        result /= "gpu_post" if gpu_post else "cpu_post"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True, help="Usually build/release/src")
    parser.add_argument("--engine", type=Path, required=True, help="YOLO26 FP32 I/O [1,300,6] engine")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing bus.jpg and zidane.jpg")
    args = parser.parse_args()
    bin_dir, engine, images = (path.resolve() for path in (args.bin_dir, args.engine, args.images))
    require(engine.is_file(), f"Missing engine: {engine}")
    for binary, _ in VERSIONS.values():
        require((bin_dir / binary).is_file(), f"Missing executable: {bin_dir / binary}")
    for name in ("bus.jpg", "zidane.jpg"):
        require((images / name).is_file(), f"Missing fixture source: {images / name}")

    with tempfile.TemporaryDirectory(prefix="yolo-series-integration-") as temporary:
        root = Path(temporary)
        fixtures = root / "images"
        fixtures.mkdir()
        shutil.copyfile(images / "bus.jpg", fixtures / "bus,fixture.jpg")
        shutil.copyfile(images / "zidane.jpg", fixtures / "zidane.jpg")
        write_blank_bmp(fixtures / "blank.bmp")
        filenames = sorted(path.name for path in fixtures.iterdir())
        reference = None
        for label, version, switches in CONFIGS:
            result = run_program(bin_dir, engine, fixtures, root / label, version, ["3", "1", *switches])
            detections = validate_results(result, version, filenames, 3)
            if reference is None:
                reference = detections
            else:
                compare_detections(detections, reference, label)
            print(f"PASS {label}: 9 measured jobs, CSV contract, exact ordered detections", flush=True)

        # Fewer jobs than slots: exercise drain without ever reusing a busy slot.
        single = root / "single"
        single.mkdir()
        shutil.copyfile(fixtures / "bus,fixture.jpg", single / "bus,fixture.jpg")
        single_reference = {"bus,fixture.jpg": reference["bus,fixture.jpg"]}
        drain_configs = [("04-drain", "04", ["2"]),
                         ("07-drain-graph0", "07", ["2", "0"]),
                         ("07-drain-graph1", "07", ["2", "1"])]
        for label, version, switches in drain_configs:
            result = run_program(bin_dir, engine, single, root / label, version, ["1", "0", *switches])
            compare_detections(validate_results(result, version, list(single_reference), 1), single_reference, label)
            print(f"PASS {label}: 1 job, 2 slots, no warmup", flush=True)

        empty = root / "empty"
        empty.mkdir()
        corrupt = root / "corrupt"
        corrupt.mkdir()
        (corrupt / "broken,fixture.jpg").write_bytes(b"not a JPEG image\n")
        for version in VERSIONS:
            invalid = [["0", "0"], ["1", "-1"], ["1x", "0"], ["2147483648", "0"],
                       ["1", "0", "1", "1", "unexpected"]]
            if version in ("04", "07"):
                invalid.extend([["1", "0", "0"], ["1", "0", "3"]])
            if version in ("02", "05", "06"):
                invalid.extend([["1", "0", "2"], ["1", "0", "-1"]])
            if version == "07":
                invalid.extend([["1", "0", "2", "2"], ["1", "0", "2", "-1"]])
            for index, arguments in enumerate(invalid):
                run_program(bin_dir, engine, fixtures, root / f"{version}-invalid-{index}",
                            version, arguments, success=False)
            for label, folder in (("empty", empty), ("corrupt", corrupt)):
                run_program(bin_dir, engine, folder, root / f"{version}-{label}",
                            version, ["1", "0"], success=False)
            print(f"PASS {version}: {len(invalid)} invalid argument cases, empty/corrupt image folders", flush=True)
    print("All series integration tests passed. Temporary fixtures and outputs removed.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"FAIL {error}") from error
