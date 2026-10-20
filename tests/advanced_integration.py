#!/usr/bin/env python3
"""Exact-result and lifecycle integration checks for 08–10; stdlib only."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import series_integration as check


VERSIONS = {
    "03": ("yolo_pinned_memory", "03_pinned_memory"),
    "08": ("yolo_preprocess_maps", "08_preprocess_maps"),
    "09": ("yolo_registered_input", "09_registered_input"),
    "10": ("yolo_split_streams", "10_split_streams"),
}
CONFIGS = [("03", [])]
CONFIGS += [(version, [str(mode)]) for version in ("08", "09") for mode in (0, 1)]
CONFIGS += [("10", [str(slots), str(split)]) for slots in range(1, 5) for split in (0, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    binaries, engine, images = (path.resolve() for path in (args.bin_dir, args.engine, args.images))
    check.require(engine.is_file(), f"Missing engine: {engine}")
    for binary, _ in VERSIONS.values():
        check.require((binaries / binary).is_file(), f"Missing binary: {binary}")
    check.VERSIONS = VERSIONS
    with tempfile.TemporaryDirectory(prefix="yolo-advanced-integration-") as directory:
        root = Path(directory)
        fixtures = root / "images"
        fixtures.mkdir()
        shutil.copyfile(images / "bus.jpg", fixtures / 'bus,"fixture.jpg')
        shutil.copyfile(images / "zidane.jpg", fixtures / "zidane.jpg")
        check.write_blank_bmp(fixtures / "blank.bmp")
        filenames = sorted(path.name for path in fixtures.iterdir())
        reference = None
        for version, switches in CONFIGS:
            label = version + "-" + "-".join(switches)
            result = check.run_program(binaries, engine, fixtures, root / label,
                                       version, ["3", "1", *switches])
            detections = check.validate_results(result, version, filenames, 3)
            if reference is None:
                reference = detections
            else:
                check.compare_detections(detections, reference, label)
            if version == "09":
                setup = check.read_csv(result / "setup.csv", ["image", "setup_ms", "teardown_ms"])
                check.require(len(setup) == 3 and {r["image"] for r in setup} == set(filenames), "Bad setup records")
                for row in setup:
                    for field in ("setup_ms", "teardown_ms"):
                        check.number(row[field], field, positive=True)
            print(f"PASS {label}: 9 jobs, exact detections, valid CSV and throughput", flush=True)

        # Four slots, three images: vary the drain offset and last image-to-slot mapping.
        for iterations in (1, 2, 4, 5):
            for split in (0, 1):
                label = f"10-drain-{iterations}-{split}"
                result = check.run_program(binaries, engine, fixtures, root / label, "10",
                                           [str(iterations), "0", "4", str(split)])
                detections = check.validate_results(result, "10", filenames, iterations)
                check.compare_detections(detections, reference, label)
        single = root / "single"
        single.mkdir()
        name = 'bus,"fixture.jpg'
        shutil.copyfile(fixtures / name, single / name)
        for split in (0, 1):
            label = f"10-single-{split}"
            result = check.run_program(binaries, engine, single, root / label, "10", ["1", "0", "4", str(split)])
            check.compare_detections(check.validate_results(result, "10", [name], 1), {name: reference[name]}, label)
        print("PASS 10: ten drain cases, including one job / four slots", flush=True)

        empty, corrupt = root / "empty", root / "corrupt"
        empty.mkdir()
        corrupt.mkdir()
        (corrupt / "broken.jpg").write_bytes(b"not JPEG")
        for version in ("08", "09", "10"):
            invalid = [["0", "0"], ["1", "-1"], ["1x", "0"], ["2147483648", "0"],
                       ["1", "0", "1", "1", "unexpected"]]
            if version == "10":
                invalid += [["1", "0", "0"], ["1", "0", "5"], ["1", "0", "2", "2"]]
            else:
                invalid += [["1", "0", "2"], ["1", "0", "-1"]]
            for index, arguments in enumerate(invalid):
                check.run_program(binaries, engine, fixtures, root / f"bad-{version}-{index}",
                                  version, arguments, success=False)
            for label, folder in (("empty", empty), ("corrupt", corrupt)):
                check.run_program(binaries, engine, folder, root / f"{label}-{version}",
                                  version, ["1", "0"], success=False)
            print(f"PASS {version}: invalid arguments and empty/corrupt folders", flush=True)
    print("All advanced integration tests passed.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"FAIL {error}") from error
