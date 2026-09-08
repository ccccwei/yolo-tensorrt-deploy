#!/usr/bin/env python3
"""Collect reproducible build and GPU environment metadata as JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def run_command(command: Sequence[str]) -> dict[str, Any]:
    executable = command[0]
    if os.sep not in executable:
        resolved = shutil.which(executable)
        if resolved is None:
            return {
                "available": False,
                "command": list(command),
                "returncode": None,
                "stdout": "",
                "stderr": f"{executable} was not found in PATH",
            }
        command = [resolved, *command[1:]]
    elif not Path(executable).is_file():
        return {
            "available": False,
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{executable} does not exist",
        }

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": True,
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }

    return {
        "available": True,
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def resolve_tensorrt_root(explicit_root: str | None) -> Path | None:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    environment_root = os.environ.get("TensorRT_ROOT")
    if environment_root:
        return Path(environment_root).expanduser().resolve()

    for candidate in ("/usr/local/TensorRT", "/opt/TensorRT"):
        path = Path(candidate)
        if path.is_dir():
            return path.resolve()
    return None


def read_tensorrt_version(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"available": False, "header": None, "version": None}

    header = root / "include" / "NvInferVersion.h"
    if not header.is_file():
        return {"available": False, "header": str(header), "version": None}

    values: dict[str, int] = {}
    pattern = re.compile(r"^#define\s+NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD)\s+(\d+)")
    for line in header.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1).lower()] = int(match.group(2))

    required = ("major", "minor", "patch", "build")
    if not all(part in values for part in required):
        return {"available": False, "header": str(header), "version": None}

    version = ".".join(str(values[part]) for part in required)
    return {"available": True, "header": str(header), "version": version}


def query_gpus() -> dict[str, Any]:
    def optional_int(value: str) -> int | None:
        try:
            return int(value)
        except ValueError:
            return None

    def optional_float(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    fields = (
        "index,name,uuid,compute_cap,driver_version,memory.total,"
        "pstate,power.limit,temperature.gpu"
    )
    result = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    if result["returncode"] == 0:
        for row in csv.reader(result["stdout"].splitlines(), skipinitialspace=True):
            if len(row) != 9:
                continue
            gpus.append(
                {
                    "index": optional_int(row[0]),
                    "name": row[1],
                    "uuid": row[2],
                    "compute_capability": row[3],
                    "driver_version": row[4],
                    "memory_total_mib": optional_float(row[5]),
                    "performance_state": row[6],
                    "power_limit_w": optional_float(row[7]),
                    "temperature_c": optional_float(row[8]),
                }
            )
    return {"command": result, "devices": gpus}


def collect_environment(tensorrt_root: str | None) -> dict[str, Any]:
    root = resolve_tensorrt_root(tensorrt_root)
    trtexec = str(root / "bin" / "trtexec") if root is not None else "trtexec"

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "source": {
            "git_commit": run_command(["git", "rev-parse", "HEAD"]),
            "git_status": run_command(["git", "status", "--short"]),
        },
        "build_tools": {
            "cmake": run_command(["cmake", "--version"]),
            "cxx": run_command(["c++", "--version"]),
            "nvcc": run_command(["nvcc", "--version"]),
        },
        "tensorrt": {
            "root": str(root) if root is not None else None,
            "headers": read_tensorrt_version(root),
            "trtexec": run_command([trtexec, "--version"]),
        },
        "gpu": query_gpus(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensorrt-root",
        help="TensorRT installation prefix; defaults to TensorRT_ROOT or common paths",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of standard output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = json.dumps(
        collect_environment(args.tensorrt_root),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output is None:
        print(document)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
