#!/usr/bin/env python3
"""Measure current chapters against 01 in one cohort; Python standard library only."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


CONFIGS = [
    ("01", "yolo_baseline", "01_baseline", []),
    ("02", "yolo_cuda_prepost", "02_cuda_prepost/cpu_post", ["0"]),
    ("03", "yolo_pinned_memory", "03_pinned_memory", []),
    ("04", "yolo_async_pipeline", "04_async_pipeline", ["2"]),
    ("05", "yolo_cuda_graph", "05_cuda_graph", ["1"]),
    ("06", "yolo_full_graph", "06_full_graph", ["1"]),
    ("07", "yolo_graph_pipeline", "07_graph_pipeline", ["2", "1"]),
    ("08", "yolo_preprocess_maps", "08_preprocess_maps", ["1"]),
    ("09", "yolo_registered_input", "09_registered_input", ["1"]),
    ("10", "yolo_split_streams", "10_split_streams", ["2", "1"]),
    ("11", "yolo_registered_input", "09_registered_input", ["1"]),
]


def capture(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def statistics(values):
    ordered = sorted(values)
    return {"samples": len(values), "mean_ms": sum(values) / len(values), **{
        f"p{p}_ms": ordered[math.ceil(p / 100 * len(ordered)) - 1] for p in (50, 95, 99)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--grouped-engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, default=Path("build/release/src"))
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.rounds < 1:
        parser.error("iterations/rounds must be positive; warmup must be nonnegative")
    repo = Path(__file__).resolve().parents[3]
    binary_dir, engine, grouped, images, output = (p.resolve() for p in
        (args.bin_dir, args.engine, args.grouped_engine, args.images, args.output))
    image_files = sorted(p for p in images.iterdir() if p.is_file()
                         and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    files = [engine, grouped, *image_files, Path(__file__).resolve(),
             repo / "CMakeLists.txt", repo / "src/CMakeLists.txt", repo / "CMakePresets.json"]
    files += sorted({binary_dir / config[1] for config in CONFIGS})
    files += sorted(p for p in (repo / "src").glob("[0-9][0-9]*/*") if p.is_file())
    if not image_files or not all(p.is_file() for p in files):
        raise RuntimeError("Missing images, engine, source, or executable")
    output.mkdir(parents=True, exist_ok=False)
    gpu_command = ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,"
                   "power.draw,clocks.sm,clocks.mem,utilization.gpu", "--format=csv"]
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "iterations_per_image": args.iterations, "warmup_per_image": args.warmup,
                "rounds": args.rounds, "platform": platform.platform(),
                "configs": CONFIGS, "engine": str(engine), "grouped_engine": str(grouped),
                "cmake_cache": (binary_dir.parent / "CMakeCache.txt").read_text(),
                "sha256": {str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p):
                           hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "runs": []}
    expected = {(p.name, i) for p in image_files for i in range(1, args.iterations + 1)}
    values = {key: [] for key, *_ in CONFIGS}
    with tempfile.TemporaryDirectory(prefix="yolo-unified-baseline-") as temp:
        for round_index in range(args.rounds):
            shift = round_index * 4 % len(CONFIGS)
            order = CONFIGS[shift:] + CONFIGS[:shift]
            if round_index % 2:
                order = list(reversed(order))
            for key, binary, result_dir, switches in order:
                name = f"r{round_index + 1}_{key}"
                work, saved = Path(temp) / name, output / name
                work.mkdir()
                saved.mkdir()
                command = [str(binary_dir / binary), str(grouped if key == "11" else engine),
                           str(images), str(args.iterations), str(args.warmup), *switches]
                before_gpu = capture(gpu_command)
                result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=300)
                (saved / "stdout.txt").write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"{name} failed; see {saved / 'stdout.txt'}")
                for path in (work / "results" / result_dir).glob("*.csv"):
                    shutil.copyfile(path, saved / path.name)
                rows = read_rows(saved / "timings.csv")
                if len(rows) != len(expected) or {(r["image"], int(r["iteration"])) for r in rows} != expected:
                    raise RuntimeError(f"{name}: unexpected sample count or IDs")
                totals = [float(r["total_ms"]) for r in rows]
                if not all(math.isfinite(v) and v > 0 for v in totals):
                    raise RuntimeError(f"{name}: invalid timing")
                detections = read_rows(saved / "detections.csv")
                if not detections or any(r["image"] not in {p.name for p in image_files} for r in detections):
                    raise RuntimeError(f"{name}: missing or invalid detection output")
                values[key].extend(totals)
                run = {"name": name, "command": command, "statistics": statistics(totals),
                       "gpu_before": before_gpu, "gpu_after": capture(gpu_command)}
                metadata["runs"].append(run)
                (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"{name}: {len(totals)} samples, P50={run['statistics']['p50_ms']:.6f} ms", flush=True)
    summary = {key: statistics(times) for key, times in values.items()}
    for key, item in summary.items():
        item["speedup_vs_01"] = summary["01"]["p50_ms"] / item["p50_ms"]
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
