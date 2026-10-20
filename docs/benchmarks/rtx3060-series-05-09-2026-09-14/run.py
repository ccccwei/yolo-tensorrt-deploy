#!/usr/bin/env python3
"""Reproduce the 02–09 experiment sequentially; Python standard library only."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone


CONFIGS = [
    ("02", "yolo_benchmark", "02_benchmark", []),
    ("03", "yolo_cuda_preprocess", "03_cuda_preprocess", []),
    ("04", "yolo_pinned_memory", "04_pinned_memory", []),
    ("05_s1", "yolo_async_pipeline", "05_async_pipeline", ["1"]),
    ("05_s2", "yolo_async_pipeline", "05_async_pipeline", ["2"]),
    ("06_g0", "yolo_cuda_graph", "06_cuda_graph", ["0"]),
    ("06_g1", "yolo_cuda_graph", "06_cuda_graph", ["1"]),
    ("07_p0", "yolo_gpu_postprocess", "07_gpu_postprocess", ["0"]),
    ("07_p1", "yolo_gpu_postprocess", "07_gpu_postprocess", ["1"]),
    ("08_g0", "yolo_full_graph", "08_full_graph", ["0"]),
    ("08_g1", "yolo_full_graph", "08_full_graph", ["1"]),
    ("09_s1_g0", "yolo_graph_pipeline", "09_graph_pipeline", ["1", "0"]),
    ("09_s1_g1", "yolo_graph_pipeline", "09_graph_pipeline", ["1", "1"]),
    ("09_s2_g0", "yolo_graph_pipeline", "09_graph_pipeline", ["2", "0"]),
    ("09_s2_g1", "yolo_graph_pipeline", "09_graph_pipeline", ["2", "1"]),
]


def capture(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def stats(values):
    values = sorted(values)
    return {"mean": sum(values) / len(values), **{
        f"p{percent}": values[math.ceil(percent / 100 * len(values)) - 1]
        for percent in (50, 95, 99)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path("build/release/src"))
    parser.add_argument("--engine", type=Path, default=Path("weights/yolo26n.fp16.engine"))
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrites a run")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.rounds < 1:
        parser.error("Positive iterations/rounds and nonnegative warmup required")
    repo = Path(__file__).resolve().parents[3]
    binary_dir, engine, images, output = (path.resolve() for path in
                                         (args.bin_dir, args.engine, args.images, args.output))
    output.mkdir(parents=True, exist_ok=False)
    image_paths = sorted(path for path in images.iterdir() if path.suffix.lower() in
                         (".jpg", ".jpeg", ".png", ".bmp") and path.is_file())
    if not image_paths:
        raise RuntimeError("No input images")
    files = sorted(repo.glob("src/0*/*")) + image_paths + [engine, repo / "CMakeLists.txt", repo / "src/CMakeLists.txt", repo / "CMakePresets.json", Path(__file__).resolve()]
    files += sorted({binary_dir / config[1] for config in CONFIGS})
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "iterations_per_image_per_round": args.iterations,
                "warmup_per_image_per_round": args.warmup, "rounds": args.rounds,
                "platform": platform.platform(), "compiler": capture(["c++", "--version"]),
                "cuda": capture(["/usr/local/cuda-11.8/bin/nvcc", "--version"]),
                "git_head": capture(["git", "-C", str(repo), "rev-parse", "HEAD"]),
                "git_status": capture(["git", "-C", str(repo), "status", "--short"]),
                "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "unset"),
                "cmake_cache": (binary_dir.parent / "CMakeCache.txt").read_text(),
                "sha256": {str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path): sha256(path)
                           for path in files if path.is_file()}, "runs": []}
    summary = {}
    gpu_command = ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu", "--format=csv"]
    with tempfile.TemporaryDirectory(prefix="yolo-series-benchmark-") as temporary:
        for round_index in range(args.rounds):
            shift = (round_index * 5) % len(CONFIGS)
            order = CONFIGS[shift:] + CONFIGS[:shift]
            if round_index % 2:
                order = list(reversed(order))
            for label, binary, result_name, switches in order:
                run_name = f"r{round_index + 1}_{label}"
                work = Path(temporary) / run_name
                work.mkdir()
                command = [str(binary_dir / binary), str(engine), str(images),
                           str(args.iterations), str(args.warmup), *switches]
                gpu_before = capture(gpu_command)
                completed = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, timeout=300, check=False)
                if completed.returncode:
                    raise RuntimeError(f"{run_name} failed:\n{completed.stdout}")
                result = work / "results" / result_name
                saved = output / run_name
                saved.mkdir()
                for path in result.glob("*.csv"):
                    shutil.copyfile(path, saved / path.name)
                (saved / "stdout.txt").write_text(completed.stdout)
                timings = rows(saved / "timings.csv")
                expected = {(path.name, i) for path in image_paths for i in range(1, args.iterations + 1)}
                if len(timings) != len(expected) or {(r["image"], int(r["iteration"])) for r in timings} != expected:
                    raise RuntimeError(f"{run_name}: unexpected sample IDs")
                accumulated = summary.setdefault(label, {"samples": 0, "stages_ms": {}, "runs": [],
                                                          "wall_ms": 0.0, "throughput_samples": 0})
                accumulated["samples"] += len(timings)
                for field in timings[0]:
                    if field.endswith("_ms"):
                        values = [float(row[field]) for row in timings]
                        if not all(math.isfinite(value) and value >= 0 for value in values):
                            raise RuntimeError(f"{run_name}: invalid timing")
                        accumulated["stages_ms"].setdefault(field, []).extend(values)
                run = {"name": run_name, "command": command, "gpu_before": gpu_before,
                       "gpu_after": capture(gpu_command), "total_ms": stats([float(r["total_ms"]) for r in timings])}
                if label[:2] >= "05" and not (saved / "throughput.csv").is_file():
                    raise RuntimeError(f"{run_name}: missing measured throughput")
                if (saved / "throughput.csv").exists():
                    throughput = rows(saved / "throughput.csv")[0]
                    count, wall, fps = int(throughput["samples"]), float(throughput["wall_ms"]), float(throughput["fps"])
                    if count != len(timings) or wall <= 0 or not math.isclose(fps, count * 1000 / wall, rel_tol=1e-7):
                        raise RuntimeError(f"{run_name}: inconsistent throughput")
                    accumulated["wall_ms"] += wall
                    accumulated["throughput_samples"] += count
                    run["fps"] = fps
                accumulated["runs"].append(run_name)
                metadata["runs"].append(run)
                print(f"{run_name}: {len(timings)} samples, P50 {run['total_ms']['p50']:.3f} ms"
                      + (f", {run['fps']:.1f} images/s" if "fps" in run else ""), flush=True)
    for item in summary.values():
        item["stages_ms"] = {field: stats(values) for field, values in item["stages_ms"].items()}
        item["fps"] = item["throughput_samples"] * 1000 / item["wall_ms"] if item["wall_ms"] else None
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved raw CSV, logs, metadata and pooled summary: {output}", flush=True)


if __name__ == "__main__":
    main()
