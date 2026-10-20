#!/usr/bin/env python3
"""Independently audit raw benchmark records and pool samples, not percentiles."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


LABELS = ("control_a", "control_b", "grouped_a", "grouped_b")
STAGES = ("staging_ms", "gpu_ms", "postprocess_ms", "total_ms")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, fields):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(fields), f"Unexpected CSV schema: {path}")
        return list(reader)


def stats(values):
    ordered = sorted(values)
    require(ordered and all(math.isfinite(value) and value >= 0 for value in ordered), "Invalid timings")
    return {"n": len(ordered), "mean": math.fsum(ordered) / len(ordered),
            **{f"p{p}": ordered[(p * len(ordered) + 99) // 100 - 1] for p in (50, 95, 99)}}


def empty():
    return {"gpu": [], "pipeline": {key: [] for key in STAGES}, "wall_ms": []}


def combine(items):
    result = empty()
    for item in items:
        result["gpu"].extend(item["gpu"])
        result["wall_ms"].extend(item["wall_ms"])
        for key in STAGES:
            result["pipeline"][key].extend(item["pipeline"][key])
    return result


def summarize(item):
    wall = math.fsum(item["wall_ms"])
    return {"engine_gpu_ms": stats(item["gpu"]),
            "pipeline_ms": {key: stats(values) for key, values in item["pipeline"].items()},
            "pipeline_fps": len(item["pipeline"]["total_ms"]) * 1000 / wall,
            "pipeline_wall_ms": wall}


def compare_numbers(expected, actual, path="summary"):
    if isinstance(expected, dict):
        require(expected.keys() == actual.keys(), f"Different keys: {path}")
        return max((compare_numbers(value, actual[key], path + "." + key)
                    for key, value in expected.items()), default=0.0)
    require(math.isclose(expected, actual, rel_tol=1e-11, abs_tol=1e-11),
            f"Summary mismatch {path}: {expected} != {actual}")
    return abs(expected - actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, help="Destination for two new JSON files; defaults to --data")
    parser.add_argument("--skip-external-hashes", action="store_true",
                        help="Offline reanalysis only: explicitly skip unavailable local engines/runtime/inputs")
    args = parser.parse_args()
    data, out = args.data.resolve(), (args.output or args.data).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not any((out / name).exists() for name in ("pooled_summary.json", "audit.json")), "Outputs must be new")
    metadata = json.loads((data / "metadata.json").read_text())
    runs = {run["name"]: run for run in metadata["runs"]}
    require(len(runs) == len(metadata["runs"]) == 36, "Expected 36 unique runs")
    expected_names = {f"r{r}_{label}_{image}_engine" for r in range(1, 4)
                      for label in LABELS for image in ("bus", "zidane")}
    expected_names |= {f"r{r}_{label}_pipeline11" for r in range(1, 4) for label in LABELS}
    require(runs.keys() == expected_names, "Missing or extra runs")
    actual_folders = {path.name for path in data.iterdir() if path.is_dir() and path.name.startswith("r")}
    require(actual_folders == expected_names, "Unexpected archived run directories")
    validated_external = {}
    for name, expected in metadata["sha256"].items():
        if not args.skip_external_hashes:
            actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            require(actual == expected, f"External input changed: {name}")
            validated_external[name] = actual

    pieces = {(label, r): empty() for label in LABELS for r in range(1, 4)}
    output_reference, detection_reference = {}, None
    checks, warnings = [], {}
    for round_number in range(1, 4):
        for label in LABELS:
            item = pieces[label, round_number]
            for image in ("bus", "zidane"):
                name = f"r{round_number}_{label}_{image}_engine"
                folder, command = data / name, runs[name]["command"]
                require(all(flag in command for flag in ("--noDataTransfers", "--useCudaGraph", "--infStreams=1",
                    "--warmUp=500", "--duration=0", "--iterations=1000")), f"Wrong engine protocol: {name}")
                with gzip.open(folder / "times.json.gz", "rt") as stream:
                    trace = json.load(stream)
                require(len(trace) == 1000, f"Wrong engine sample count: {name}")
                require(all(all(math.isfinite(value) for value in row.values()) and row["computeMs"] > 0
                            and row["h2dMs"] == row["d2hMs"] == 0
                            and row["latencyMs"] == row["computeMs"] for row in trace), f"Invalid engine events: {name}")
                require(all(a["startComputeMs"] < b["startComputeMs"] for a, b in zip(trace, trace[1:])),
                        f"Unordered engine trace: {name}")
                item["gpu"].extend(row["computeMs"] for row in trace)
                tensors = json.loads((folder / "output.json").read_text())
                require(len(tensors) == 1 and tensors[0]["name"] == "output0"
                        and tensors[0]["dimensions"] == "1x300x6", f"Wrong exported output: {name}")
                values = tensors[0]["values"]
                require(len(values) == 1800 and all(math.isfinite(value) for value in values), f"Bad output: {name}")
                require(image not in output_reference or values == output_reference[image], f"Engine output changed: {name}")
                output_reference[image] = values
                checks.append({"run": name, "samples": 1000, "engine_gpu_ms": stats([row["computeMs"] for row in trace])})

            name = f"r{round_number}_{label}_pipeline11"
            folder = data / name
            require(runs[name]["command"][-3:] == ["1000", "200", "1"], f"Wrong pipeline protocol: {name}")
            trace = read_csv(folder / "timings.csv", ("image", "iteration", *STAGES))
            require(len(trace) == 2000, f"Wrong pipeline sample count: {name}")
            identities = {(row["image"], int(row["iteration"])) for row in trace}
            require(identities == {(image, iteration) for image in ("bus.jpg", "zidane.jpg")
                                   for iteration in range(1, 1001)}, f"Missing/duplicate sample IDs: {name}")
            for key in STAGES:
                values = [float(row[key]) for row in trace]
                require(all(math.isfinite(value) and value >= 0 for value in values), f"Invalid pipeline stage: {name}/{key}")
                item["pipeline"][key].extend(values)
            throughput = read_csv(folder / "throughput.csv", ("samples", "wall_ms", "fps"))
            require(len(throughput) == 1 and int(throughput[0]["samples"]) == 2000, f"Wrong throughput count: {name}")
            wall = float(throughput[0]["wall_ms"])
            require(math.isfinite(wall) and wall > 0 and math.isclose(float(throughput[0]["fps"]), 2000000 / wall,
                    rel_tol=1e-7), f"Wrong wall-clock FPS: {name}")
            require(wall >= math.fsum(float(row["total_ms"]) for row in trace), f"Wall shorter than measured requests: {name}")
            item["wall_ms"].append(wall)
            setup = read_csv(folder / "setup.csv", ("image", "setup_ms", "teardown_ms"))
            require(len(setup) == 2 and {row["image"] for row in setup} == {"bus.jpg", "zidane.jpg"}
                    and all(math.isfinite(float(row[key])) and float(row[key]) > 0 for row in setup
                            for key in ("setup_ms", "teardown_ms")), f"Invalid setup timing: {name}")
            detections = (folder / "detections.csv").read_bytes()
            require(detection_reference is None or detections == detection_reference, f"Pipeline detections changed: {name}")
            detection_reference = detections
            records = read_csv(folder / "detections.csv", ("image", "class_id", "score", "left", "top", "right", "bottom"))
            require(sum(row["image"] == "bus.jpg" for row in records) == 5
                    and sum(row["image"] == "zidane.jpg" for row in records) == 3, f"Unexpected detection counts: {name}")
            checks.append({"run": name, "samples": 2000, "pipeline_ms": {key: stats(item["pipeline"][key]) for key in STAGES},
                           "pipeline_wall_ms": wall, "pipeline_fps": 2000000 / wall})

    for name, run in runs.items():
        require(run["returncode"] == 0, f"Nonzero process status: {name}")
        log = (data / name / "stdout.txt").read_text()
        require("[E]" not in log, f"Error log: {name}")
        warnings[name] = [line for line in log.splitlines() if "[W]" in line or "[TensorRT]" in line]
    by_engine = {label: summarize(combine(pieces[label, r] for r in range(1, 4))) for label in LABELS}
    summary_error = compare_numbers(json.loads((data / "summary.json").read_text()), by_engine)
    pooled = {variant: summarize(combine(pieces[f"{variant}_{build}", r] for build in ("a", "b") for r in range(1, 4)))
              for variant in ("control", "grouped")}
    for item in pooled.values():
        require(item["engine_gpu_ms"]["n"] == item["pipeline_ms"]["total_ms"]["n"] == 12000, "Wrong pooled count")
    pooled["method"] = "Merge individual A/B samples, then nearest-rank percentiles; FPS=sum(samples)*1000/sum(wall_ms)"
    pooled["comparison"] = {
        "engine_p50_latency_reduction_percent": (1 - pooled["grouped"]["engine_gpu_ms"]["p50"] / pooled["control"]["engine_gpu_ms"]["p50"]) * 100,
        "pipeline_p50_latency_reduction_percent": (1 - pooled["grouped"]["pipeline_ms"]["total_ms"]["p50"] / pooled["control"]["pipeline_ms"]["total_ms"]["p50"]) * 100,
        "pipeline_fps_speedup": pooled["grouped"]["pipeline_fps"] / pooled["control"]["pipeline_fps"],
    }
    audit = {
        "status": "passed", "engine_runs": 24, "pipeline_runs": 12,
        "engine_samples": 24000, "pipeline_samples": 24000, "total_samples": 48000,
        "summary_recomputed": by_engine, "maximum_summary_numeric_difference": summary_error,
        "run_checks": checks, "warnings_by_run": warnings,
        "last_engine_outputs_equal_per_image": True, "last_pipeline_detection_csvs_byte_identical": True,
        "external_hashes_skipped": args.skip_external_hashes, "verified_external_sha256": validated_external,
        "pipeline_timing_scope": {
            "staging_ms": "CPU geometry calculation; REGISTERED=1 does not copy to pinned staging",
            "gpu_ms": "CUDA event interval around Graph H2D + CUDA preprocessing + TensorRT + D2H, not model-only",
            "postprocess_ms": "CPU confidence filtering, coordinate recovery and invalid-box rejection; no drawing",
            "total_ms": "Host-clock request start through completed CPU postprocess, including launch/synchronization overhead",
            "wall_ms": "Measured loop, including timing-query and record bookkeeping; excludes warmup/setup/teardown, decode, drawing, file IO",
        },
        "limitations": [
            "Engine-only uses GPU-resident fixed input and trtexec pipelined Graph submission; not host request latency or end-to-end FPS",
            "Only final exported output per engine run and final detection list per image are archived, not output on every timed request",
            "Engine output JSON uses text-rounded scores; the separate 21-image binary-output audit is the bitwise correctness evidence",
            "48k measurements repeat two images; this is not 48k independent inputs, labeled accuracy or statistical significance",
            "Two builds per variant used an evolving cache; control_a and control_b Inspector layouts differ, so retain separate build summaries",
            "No locked GPU clocks or controlled-background guarantee; GPU telemetry and run order remain in metadata",
        ],
        "rounds": {str(r): {variant: summarize(combine(pieces[f"{variant}_{b}", r] for b in ("a", "b")))
                            for variant in ("control", "grouped")} for r in range(1, 4)},
        "archive_sha256": {str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(data.rglob("*")) if path.is_file()},
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for name, contents in (("pooled_summary.json", pooled), ("audit.json", audit)):
        with (out / name).open("x") as stream:
            json.dump(contents, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"pooled": pooled, "audit": {key: audit[key] for key in (
        "status", "total_samples", "maximum_summary_numeric_difference", "last_engine_outputs_equal_per_image",
        "last_pipeline_detection_csvs_byte_identical")}}, indent=2))


if __name__ == "__main__":
    main()
