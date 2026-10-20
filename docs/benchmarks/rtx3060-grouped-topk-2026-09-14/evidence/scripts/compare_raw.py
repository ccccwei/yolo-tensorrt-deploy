"""CPU-only exact audit of four freshly built engines on the same image fixtures."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LABELS = ("control_a", "control_b", "grouped_a", "grouped_b")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(label):
    directory = ROOT / ("raw_" + label)
    raw = {}
    hashes = {}
    for path in sorted(directory.glob("*.bin")):
        value = np.fromfile(path, dtype=np.float32).reshape(300, 6)
        if not np.isfinite(value).all():
            raise AssertionError(f"Nonfinite output: {path}")
        if np.any(value[:, 4] < 0) or np.any(value[:, 4] > 1):
            raise AssertionError(f"Invalid confidence: {path}")
        if not np.array_equal(value[:, 5], value[:, 5].astype(np.int64)):
            raise AssertionError(f"Nonintegral class: {path}")
        if np.any(value[:, 5] < 0) or np.any(value[:, 5] >= 80):
            raise AssertionError(f"Invalid class: {path}")
        if np.any(value[:-1, 4] < value[1:, 4]):
            raise AssertionError(f"Unsorted scores: {path}")
        raw[path.name[:-4]] = value
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    if len(raw) != 21:
        raise AssertionError(f"Expected 21 image outputs: {label}")
    detections = {name: [] for name in raw}
    csv_path = directory / "detections.csv"
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            detections[row["image"]].append(row)
    hashes[str(csv_path.relative_to(ROOT))] = sha256(csv_path)
    return raw, detections, hashes


def multiset(value):
    return Counter(row.tobytes() for row in value)


def without_score(rows):
    return [{key: value for key, value in row.items() if key != "score"} for row in rows]


def compare(left, right, data):
    raw_left, det_left, _ = data[left]
    raw_right, det_right, _ = data[right]
    if raw_left.keys() != raw_right.keys():
        raise AssertionError("Different input filename sets")
    images = []
    for name, a in raw_left.items():
        b = raw_right[name]
        da, db = det_left[name], det_right[name]
        sa, sb = np.sort(a[:, 4]), np.sort(b[:, 4])
        row_multiset_a, row_multiset_b = multiset(a), multiset(b)
        images.append({
            "image": name,
            "raw_bytes_identical": a.tobytes() == b.tobytes(),
            "equal_float_elements": int(np.count_nonzero(a == b)),
            "equal_aligned_rows": int(np.count_nonzero(np.all(a == b, axis=1))),
            "raw_row_multiset_equal": row_multiset_a == row_multiset_b,
            "raw_common_rows_multiset": sum((row_multiset_a & row_multiset_b).values()),
            "aligned_max_absolute_difference": float(np.abs(a.astype(np.float64) - b).max()),
            "score_order_equal": bool(np.array_equal(a[:, 4], b[:, 4])),
            "score_multiset_equal": sa.tobytes() == sb.tobytes(),
            "sorted_score_max_absolute_difference": float(np.abs(sa.astype(np.float64) - sb).max()),
            "selected_score_duplicates": [int(300 - len(np.unique(x[:, 4]))) for x in (a, b)],
            "raw_confidence_at_least_0_25": [int(np.count_nonzero(x[:, 4] >= .25)) for x in (a, b)],
            "postprocessed_count": [len(da), len(db)],
            "postprocessed_class_order_equal": [x["class_id"] for x in da] == [x["class_id"] for x in db],
            "postprocessed_class_and_integer_box_order_equal": without_score(da) == without_score(db),
            "postprocessed_score_order_equal": [x["score"] for x in da] == [x["score"] for x in db],
            "postprocessed_rows_equal": da == db,
            "postprocessed_rows_left_if_different": da if da != db else [],
            "postprocessed_rows_right_if_different": db if da != db else [],
        })
    return {
        "left": left, "right": right,
        "images": images,
        "totals": {
            "images": len(images),
            "raw_rows_per_engine": len(images) * 300,
            "raw_float_elements_per_engine": len(images) * 1800,
            "byte_identical_images": sum(row["raw_bytes_identical"] for row in images),
            "equal_raw_aligned_rows": sum(row["equal_aligned_rows"] for row in images),
            "equal_row_multiset_images": sum(row["raw_row_multiset_equal"] for row in images),
            "equal_score_multiset_images": sum(row["score_multiset_equal"] for row in images),
            "equal_postprocessed_images": sum(row["postprocessed_rows_equal"] for row in images),
            "postprocessed_detections": [sum(row["postprocessed_count"][i] for row in images) for i in (0, 1)],
            "max_absolute_raw_difference": max(row["aligned_max_absolute_difference"] for row in images),
        },
    }


data = {label: load(label) for label in LABELS}
report = {
    "scope": "21 synthetic/derived image fixtures, four fresh engines, last raw output per image",
    "raw_format": "Little-endian native float32, [1,300,6], binary rather than text-rounded scores",
    "limitations": [
        "Not a labeled dataset or a mAP evaluation",
        "Per-execution program checks three Graph replays internally; only one raw output is archived per image",
        "No tolerance, row reordering or low-confidence filtering is used for the raw equality checks",
        "Postprocessed CSV rows include the production coordinate scaling/clamping and invalid-box rejection",
    ],
    "file_sha256": {name: digest for _, _, hashes in data.values() for name, digest in hashes.items()},
    "comparisons": [compare(a, b, data) for a, b in (
        ("control_a", "control_b"), ("grouped_a", "grouped_b"),
        ("control_a", "grouped_a"), ("control_b", "grouped_b"))],
}
for name in ("compare_raw.py", "infer_raw.cpp", "infer_raw", "image_cases/metadata.json"):
    report["file_sha256"][name] = sha256(ROOT / name)
for label in LABELS:
    report["file_sha256"][label + ".engine"] = sha256(ROOT / (label + ".engine"))
with (ROOT / "comparison.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
for item in report["comparisons"]:
    print(item["left"], "vs", item["right"], item["totals"])
