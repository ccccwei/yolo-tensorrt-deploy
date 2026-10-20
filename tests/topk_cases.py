#!/usr/bin/env python3
"""Deterministic, tie-aware correctness fixtures for the actual YOLO26 selector."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def score_cases(width, k=300, groups=8):
    """One-dimensional TopK inputs, shaped [1, width], with exact FP32 values."""
    if width % groups or width // groups < k:
        raise ValueError("Each group must contain at least k values")
    rng = np.random.default_rng(260013 + width)
    unique = rng.permutation(width).astype(np.float32) / np.float32(width)
    concentrated = np.zeros(width, dtype=np.float32)
    winners = min(k + 1, width)
    concentrated[:winners] = np.arange(winners, dtype=np.float32) + 1
    boundary = unique * np.float32(.5)
    boundary[boundary >= np.sort(boundary)[-min(k + 10, width)]] = np.float32(.75)
    threshold = np.resize(np.array([0.0, np.nextafter(np.float32(.25), np.float32(0)),
                                   .25, np.nextafter(np.float32(.25), np.float32(1)),
                                   .5, 1.0], dtype=np.float32), width)
    edges = np.zeros(width, dtype=np.float32)
    for group in range(groups):
        edges[group * (width // groups)] = np.float32(group + 1)
        edges[(group + 1) * (width // groups) - 1] = np.float32(group + groups + 1)
    cases = {
        "unique": unique,
        "one_group_contains_all_winners": concentrated,
        "tied_boundary": boundary,
        "all_equal": np.full(width, .75, dtype=np.float32),
        "all_zero": np.zeros(width, dtype=np.float32),
        "fp16_rounded_ties": unique.astype(np.float16).astype(np.float32),
        "threshold_neighbors": threshold,
        "group_edges": edges,
        "negative_scores": unique - np.float32(2),
    }
    return {name: value.reshape(1, width) for name, value in cases.items()}


def assert_selection(scores, values, indices, k=300):
    """Require correct sorted score multiset and source membership, allowing ties."""
    scores, values, indices = (np.asarray(item) for item in (scores, values, indices))
    if scores.ndim != 2 or values.shape != (scores.shape[0], k) or indices.shape != values.shape:
        raise AssertionError("Invalid selection shapes")
    if not np.issubdtype(indices.dtype, np.integer) or not np.isfinite(scores).all():
        raise AssertionError("Indices must be integers and scores finite")
    if not np.isfinite(values).all() or np.any(indices < 0) or np.any(indices >= scores.shape[1]):
        raise AssertionError("Invalid selected values or indices")
    expected = np.sort(scores, axis=1)[:, -k:][:, ::-1]
    if not np.array_equal(values, expected):
        raise AssertionError("Selected scores differ from the exact global TopK multiset/order")
    if not np.array_equal(values, np.take_along_axis(scores, indices, axis=1)):
        raise AssertionError("Selected score does not match its source index")
    for row, selected in zip(scores, indices):
        if len(np.unique(selected)) != k:
            raise AssertionError("Duplicate selected indices")
        if len(np.unique(row)) == len(row):
            canonical = np.argsort(row)[-k:][::-1]
            if not np.array_equal(selected, canonical):
                raise AssertionError("Unique scores must select identical original indices")


def detection_cases():
    """Yield [1, 8400, 80] scores; no activation is needed at this graph cut."""
    anchors, classes, k = 8400, 80, 300
    rng = np.random.default_rng(260013)
    count = anchors * classes
    # Exact binary fractions remain unique in FP32 throughout this range.
    unique = rng.permutation(count).astype(np.float32).reshape(anchors, classes) / np.float32(2**20)
    yield "unique", unique[np.newaxis].copy()
    yield "fp16_rounded_ties", unique.astype(np.float16).astype(np.float32)[np.newaxis].copy()
    tied = np.zeros((anchors, classes), dtype=np.float32)
    # More than max_det tied high-confidence positions, including multiple classes
    # at one position and group boundaries. Either tied index set is acceptable.
    tied[:k + 80, :3] = np.float32(.75)
    tied[1049:1052, 5:7] = np.float32(.75)
    yield "high_confidence_ties", tied[np.newaxis].copy()
    threshold = np.zeros((anchors, classes), dtype=np.float32)
    threshold[:40, :3] = np.float32(.5)
    threshold[40:80, :2] = np.nextafter(np.float32(.25), np.float32(1))
    threshold[80:110, :2] = np.float32(.25)
    threshold[110:140, :2] = np.nextafter(np.float32(.25), np.float32(0))
    yield "threshold_multiclass", threshold[np.newaxis].copy()


def encoded_boxes():
    """Unique integer XYXY coordinates encode the position (also exact in FP16)."""
    anchor = np.arange(8400, dtype=np.float32)
    x, y = anchor % 128, np.floor(anchor / 128)
    return np.stack((x, y, x + 1, y + 1), axis=1)[np.newaxis]


def validate_detections(scores, output):
    """Validate [1,300,6] actual selector output against the full source score grid."""
    output = np.asarray(output, dtype=np.float32)
    if output.shape != (1, 300, 6) or not np.isfinite(output).all():
        raise AssertionError("Expected finite [1,300,6] output")
    anchor = (output[..., 1] * 128 + output[..., 0]).astype(np.int64)
    category = output[..., 5].astype(np.int64)
    if np.any(anchor < 0) or np.any(anchor >= 8400) or np.any(category < 0) or np.any(category >= 80):
        raise AssertionError("Invalid decoded anchor/class")
    if not np.array_equal(output[..., 5], category):
        raise AssertionError("Non-integral output class")
    expected_boxes = encoded_boxes()[0, anchor[0]].reshape(1, 300, 4)
    if not np.array_equal(output[..., :4], expected_boxes):
        raise AssertionError("Output box does not match its original anchor")
    flat = scores.reshape(1, -1)
    assert_selection(flat, output[..., 4], anchor * 80 + category)
    return {
        "rows": 300,
        "above_threshold": int(np.count_nonzero(output[..., 4] >= np.float32(.25))),
        "distinct_anchor_class_pairs": int(len(np.unique(anchor * 80 + category))),
        "minimum_selected_score": float(output[..., 4].min()),
    }


def extract_selector(source, target):
    """Cut the existing model at decoded boxes/scores; retain its actual tail nodes."""
    import onnx

    names = ("/model.23/Split_output_0", "/model.23/Split_output_1")
    model = onnx.shape_inference.infer_shapes(onnx.load(source))
    inferred = {value.name: value for value in model.graph.value_info}
    for name, shape in zip(names, ((1, 8400, 4), (1, 8400, 80))):
        value = inferred[name]
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Expected FP32 selector input: {name}")
        if tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim) != shape:
            raise ValueError(f"Unexpected selector input shape: {name}")
    selector = onnx.utils.Extractor(model).extract_model(list(names), ["output0"])
    rename = dict(zip(names, ("boxes", "scores")))
    for node in selector.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = rename.get(name, name)
    for value in (*selector.graph.input, *selector.graph.value_info):
        value.name = rename.get(value.name, value.name)
    onnx.checker.check_model(selector)
    with target.open("xb") as stream:
        stream.write(selector.SerializeToString())
    return selector


def prepare(args):
    """Export CPU-verified tails and identical input bytes, never overwriting a run."""
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.op_run import OpRun

    # ONNX 1.22's np.choose-based reference GatherElements cannot address >64
    # rows on NumPy 2. This is the equivalent gather, not a replacement selector.
    class GatherElements(OpRun):
        op_domain = ""

        def _run(self, data, indices, axis=0):
            return (np.take_along_axis(data, indices, axis=axis),)

    args.output.mkdir(parents=True, exist_ok=False)
    models = {
        label: extract_selector(source, args.output / f"{label}.onnx")
        for label, source in (("control", args.control), ("grouped", args.grouped))
    }
    evaluators = {label: ReferenceEvaluator(model, new_ops=[GatherElements])
                  for label, model in models.items()}
    boxes = encoded_boxes()
    boxes.tofile(args.output / "boxes.bin")
    report = {
        "scope": "Actual ONNX tail after Split: decoded boxes + sigmoid scores -> output0",
        "required_precision": "FP32 selector build (do not enable --fp16); ties include FP16-rounded source scores",
        "input_shapes": {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]},
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (args.control, args.grouped)},
        "cases": {},
    }
    for name, scores in detection_cases():
        case = args.output / name
        case.mkdir()
        scores.tofile(case / "scores.bin")
        outputs = {}
        case_report = {}
        for label, evaluator in evaluators.items():
            output = evaluator.run(None, {"boxes": boxes, "scores": scores})[0]
            outputs[label] = output
            case_report[label] = validate_detections(scores, output)
            np.save(case / f"{label}.reference.npy", output)
        case_report["reference_rows_identical"] = bool(np.array_equal(outputs["control"], outputs["grouped"]))
        if name == "unique" and not case_report["reference_rows_identical"]:
            raise AssertionError("Unique-score outputs must match exactly")
        report["cases"][name] = case_report
        print(f"PASS CPU selector: {name}; identical rows={case_report['reference_rows_identical']}")
    report["fixture_sha256"] = {
        str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.rglob("*")) if path.is_file()
    }
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")


def check_output(args):
    scores = np.fromfile(args.scores, dtype=np.float32).reshape(1, 8400, 80)
    if args.output.suffix == ".npy":
        output = np.load(args.output, allow_pickle=False)
    else:
        output = np.fromfile(args.output, dtype=np.float32).reshape(1, 300, 6)
    print(json.dumps(validate_detections(scores, output), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("prepare", help="Extract both actual tails and verify four CPU fixtures")
    export.add_argument("--control", type=Path, required=True)
    export.add_argument("--grouped", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    check = commands.add_parser("check", help="Validate raw FP32 or NPY GPU output; JSON loses exact score bits")
    check.add_argument("--scores", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else check_output)(args)


if __name__ == "__main__":
    main()
