"""Audit serialized ONNX preservation and the public TensorRT Inspector records."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import onnx


ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parents[1]
LABELS = ("control_a", "control_b", "grouped_a", "grouped_b")
TARGETS = {"/model.23/TopK", "/model.23/TopK_1"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha256(path):
    return digest(path.read_bytes())


def compare_layers(left, right, inspectors):
    a, b = inspectors[left], inspectors[right]
    la, lb = ({layer["Name"]: layer for layer in value["Layers"]} for value in (a, b))
    if len(la) != len(a["Layers"]) or len(lb) != len(b["Layers"]):
        raise AssertionError("Duplicate layer names")
    common = sorted(la.keys() & lb.keys())
    changed = []
    for name in common:
        fields = {key: {"left": la[name].get(key), "right": lb[name].get(key)}
                  for key in la[name].keys() | lb[name].keys() if la[name].get(key) != lb[name].get(key)}
        if fields:
            changed.append({"name": name, "fields": fields})
    return {
        "left": left, "right": right,
        "all_inspector_records_equal": a == b,
        "layer_counts": [len(la), len(lb)],
        "exact_common_records": sum(la[name] == lb[name] for name in common),
        "common_record_order_equal": [x["Name"] for x in a["Layers"] if x["Name"] in lb]
                                      == [x["Name"] for x in b["Layers"] if x["Name"] in la],
        "bindings_equal": a["Bindings"] == b["Bindings"],
        "deleted_records": [la[name] for name in sorted(la.keys() - lb.keys())],
        "added_records": [lb[name] for name in sorted(lb.keys() - la.keys())],
        "changed_records": changed,
        "changed_public_tactic_records": [name for name in common
                                            if la[name].get("TacticName") != lb[name].get("TacticName")
                                            or la[name].get("TacticValue") != lb[name].get("TacticValue")],
    }


original_path, grouped_path = ROOT / "weights/yolo26n.onnx", ARTIFACT / "grouped8.onnx"
original, grouped = (onnx.load(path) for path in (original_path, grouped_path))
original_nodes = {node.name: node for node in original.graph.node}
grouped_nodes = {node.name: node for node in grouped.graph.node}
preserved = [name for name, node in original_nodes.items()
             if name not in TARGETS and name in grouped_nodes
             and node.SerializeToString() == grouped_nodes[name].SerializeToString()]
if len(preserved) != len(original.graph.node) - 2:
    raise AssertionError("An unrelated ONNX node changed")
if [node.name for node in original.graph.node if node.name not in TARGETS] != [
        node.name for node in grouped.graph.node if node.name in preserved]:
    raise AssertionError("Unrelated ONNX node order changed")
initializers = [{"name": item.name, "sha256": digest(item.SerializeToString())}
                for item in original.graph.initializer]
if [item.SerializeToString() for item in original.graph.initializer] != [
        item.SerializeToString() for item in grouped.graph.initializer]:
    raise AssertionError("ONNX initializer/order changed")
left, right = deepcopy(original), deepcopy(grouped)
del left.graph.node[:]
del right.graph.node[:]
if left.SerializeToString() != right.SerializeToString():
    raise AssertionError("ONNX non-node model metadata, IO or initializer changed")

inspectors = {label: json.loads((ARTIFACT / (label + ".layers.json")).read_text()) for label in LABELS}
builds = {label: json.loads((ARTIFACT / (label + ".build.json")).read_text()) for label in LABELS}
for label, build in builds.items():
    source = original_path if label.startswith("control") else grouped_path
    if build["source_sha256"] != sha256(source) or build["engine_sha256"] != sha256(ARTIFACT / (label + ".engine")):
        raise AssertionError("Build source/engine hashes changed")
flags = [[arg for arg in builds[label]["command"] if not arg.startswith(
    ("--onnx=", "--saveEngine=", "--exportLayerInfo="))] for label in LABELS]
if not all(flag == flags[0] for flag in flags):
    raise AssertionError("Different build flags")

sources = [ROOT / name for name in (
    "tools/grouped_topk.py", "tests/test_grouped_topk.py", "tests/topk_cases.py",
    "src/06_cuda_graph/main.cpp", "src/06_cuda_graph/preprocess.cu", "src/CMakeLists.txt",
    "CMakeLists.txt", "CMakePresets.json", "weights/yolo26n.onnx", "images/bus.jpg", "images/zidane.jpg")]
sources += [grouped_path]
sources += [ARTIFACT / name for name in (
    "inspect_structure.py", "build_one.py", "image_cases.py", "infer_raw.cpp", "infer_raw",
    "compare_raw.py", "comparison.json", "run_selector.py", "selector_gpu.json", "image_cases/metadata.json")]
sources += sorted((ARTIFACT / "image_cases").glob("*.png"))
sources += [ARTIFACT / (label + suffix) for label in LABELS
            for suffix in (".build.json", ".layers.json", ".engine")]
report = {
    "onnx": {
        "original_nodes": len(original.graph.node), "grouped_nodes": len(grouped.graph.node),
        "original_topk_nodes": sum(node.op_type == "TopK" for node in original.graph.node),
        "grouped_topk_nodes": sum(node.op_type == "TopK" for node in grouped.graph.node),
        "replaced_nodes": sorted(TARGETS),
        "preserved_unrelated_nodes_byte_exact": len(preserved),
        "preserved_unrelated_node_order": True,
        "non_node_model_proto_byte_exact": True,
        "preserved_initializers_byte_exact": len(initializers),
        "initializer_sha256": initializers,
    },
    "engines": {label: {
        "layer_records": len(inspectors[label]["Layers"]),
        "layer_types": dict(Counter(layer["LayerType"] for layer in inspectors[label]["Layers"])),
        "bindings": inspectors[label]["Bindings"],
        "engine_size_bytes": (ARTIFACT / (label + ".engine")).stat().st_size,
    } for label in LABELS},
    "comparisons": [compare_layers(a, b, inspectors) for a, b in (
        ("control_a", "control_b"), ("grouped_a", "grouped_b"),
        ("control_a", "grouped_a"), ("control_b", "grouped_b"))],
    "builds": {
        "same_flags_excluding_model_and_output_paths": True,
        "common_flags": flags[0],
        "order_by_start": [{"label": label, "started_unix": builds[label]["started_unix"],
                             "cache_before": builds[label]["cache_before"],
                             "cache_after": builds[label]["cache_after"],
                             "wall_s": builds[label]["wall_s"]}
                            for label in sorted(LABELS, key=lambda name: builds[name]["started_unix"])],
    },
    "interpretation": [
        "control_a has 223 layer records; control_b, grouped_a and grouped_b have identical 216-record public Inspector JSON",
        "The only common record change is ForeignNode inputs/outputs; seven explicit Reformat records disappear",
        "All other 215 records, their relative order, bindings and public tactic fields are unchanged",
        "The same original ONNX rebuild also removed the seven Reformats: the removal is not uniquely attributable to grouped TopK",
        "A sequentially evolving timing cache and build conditions are confounders, not evidence of a specific cause",
        "Inspector does not expose the complete ForeignNode implementation; equal records do not prove equal compiled code or kernel counts",
    ],
    "file_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
}
with (ARTIFACT / "engine_structure.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print("ONNX: 443 unrelated nodes, 204 initializers and non-node protobuf preserved byte-exact")
for item in report["comparisons"]:
    print(item["left"], "vs", item["right"], "layer counts", item["layer_counts"],
          "exact common", item["exact_common_records"], "changed", len(item["changed_records"]),
          "deleted", len(item["deleted_records"]), "all JSON equal", item["all_inspector_records_equal"])
