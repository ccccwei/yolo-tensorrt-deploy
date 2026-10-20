#!/usr/bin/env python3
"""Replace the two static YOLO26 TopK selections with exact grouped selections.

This is a bounded experiment for the repository's batch-1, 640px, 80-class
ONNX export. Scores are preserved; equal-score indices may differ by backend.
The original model, all unrelated nodes, and every initializer are untouched.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


TARGETS = {"/model.23/TopK": 8400, "/model.23/TopK_1": 24000}
K = 300


def grouped_nodes(node, width, groups):
    """Build Ultralytics' group-local TopK, merge TopK, and index remapping."""
    if groups < 1 or width % groups or width // groups < K:
        raise ValueError("groups must divide the selection width, with >= 300 items per group")
    if groups == 1:
        return [deepcopy(node)]

    prefix = node.name + "/grouped/"
    nodes = []

    def constant(name, value):
        output = prefix + name
        nodes.append(helper.make_node(
            "Constant", [], [output], name=output + "/Constant",
            value=numpy_helper.from_array(np.asarray(value, dtype=np.int64))))
        return output

    def operation(op, name, inputs, outputs=None, **attributes):
        outputs = outputs or [prefix + name]
        nodes.append(helper.make_node(op, inputs, outputs, name=prefix + name + "/" + op,
                                      **attributes))
        return outputs

    shape = constant("shape", [1, groups, width // groups])
    flat_shape = constant("flat_shape", [1, groups * K])
    group_size = constant("group_size", [width // groups])
    grouped = operation("Reshape", "reshape", [node.input[0], shape])[0]
    values, local = operation("TopK", "local", [grouped, node.input[1]],
                              [prefix + "local_values", prefix + "local_indices"],
                              axis=2, largest=1, sorted=1)
    values = operation("Reshape", "flat_values", [values, flat_shape])[0]
    local = operation("Reshape", "flat_indices", [local, flat_shape])[0]
    _, winners = operation("TopK", "merge", [values, node.input[1]],
                            [node.output[0], prefix + "winners"], axis=1, largest=1, sorted=1)
    group = operation("Div", "group", [winners, node.input[1]])[0]
    offset = operation("Mul", "offset", [group, group_size])[0]
    index = operation("GatherElements", "index", [local, winners], axis=1)[0]
    operation("Add", "global_indices", [offset, index], [node.output[1]])
    return nodes


def _shapes_and_constants(model):
    """Resolve the exporter's constant Expand shape on a validation-only copy."""
    check = deepcopy(model)
    constants = {x.name: numpy_helper.to_array(x) for x in check.graph.initializer}
    for node in check.graph.node:
        if (node.domain == "" and node.op_type in
                {"Constant", "ConstantOfShape", "Mul", "Equal", "Where"}
                and len(node.output) == 1 and all(x in constants for x in node.input)):
            value = ReferenceEvaluator(node, opsets={"": 16}).run(
                None, {x: constants[x] for x in node.input})[0]
            constants[node.output[0]] = value
            node.CopyFrom(helper.make_node(
                "Constant", [], list(node.output), name=node.name,
                value=numpy_helper.from_array(value)))
    inferred = onnx.shape_inference.infer_shapes(check, strict_mode=True)
    shapes = {x.name: tuple(x.dims) for x in inferred.graph.initializer}
    shapes.update({x.name: tuple(d.dim_value if d.HasField("dim_value") else None
                           for d in x.type.tensor_type.shape.dim)
                   for x in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info)})
    return shapes, constants


def rewrite_model(model, groups=8):
    """Return a validated copy; groups=1 is the unchanged-graph control."""
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
        raise ValueError("groups must be a positive integer")
    if [(x.domain, x.version) for x in model.opset_import] != [("", 16)]:
        raise ValueError("expected the repository's default-domain opset 16 export")
    if any(x.data_location == TensorProto.EXTERNAL for x in model.graph.initializer):
        raise ValueError("external-data models are not supported by this experiment")
    onnx.checker.check_model(model, full_check=True)
    shapes, constants = _shapes_and_constants(model)
    if ([(x.name, shapes[x.name], x.type.tensor_type.elem_type) for x in model.graph.input]
            != [("images", (1, 3, 640, 640), TensorProto.FLOAT)] or
            [(x.name, shapes[x.name], x.type.tensor_type.elem_type) for x in model.graph.output]
            != [("output0", (1, 300, 6), TensorProto.FLOAT)]):
        raise ValueError("expected FP32 images[1,3,640,640] -> output0[1,300,6]")
    targets = [x for x in model.graph.node if x.op_type == "TopK"]
    if (len(targets) != 2 or {x.name for x in targets} != set(TARGETS) or
            sum(x.name in TARGETS for x in model.graph.node) != 2):
        raise ValueError("expected exactly /model.23/TopK and /model.23/TopK_1")
    for node in targets:
        width = TARGETS[node.name]
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (node.domain or len(node.input) != 2 or len(node.output) != 2 or
                attributes != {"axis": 1, "largest": 1, "sorted": 1}):
            raise ValueError(f"{node.name}: expected axis=1, largest=1, sorted=1")
        k = constants.get(node.input[1])
        if k is None or k.dtype != np.int64 or k.shape != (1,) or k[0] != K:
            raise ValueError(f"{node.name}: expected constant int64 k=[300]")
        if shapes.get(node.input[0]) != (1, width):
            raise ValueError(f"{node.name}: expected selection input [1,{width}]")
        if width % groups or width // groups < K:
            raise ValueError(f"groups={groups} must divide {width}, with >= 300 items per group")

    result = deepcopy(model)
    if groups != 1:
        existing = {x.name for x in model.graph.node}
        existing.update(x.name for x in model.graph.initializer)
        existing.update(x.name for x in (*model.graph.input, *model.graph.output))
        existing.update(output for x in model.graph.node for output in x.output)
        nodes = []
        for node in model.graph.node:
            if node.name not in TARGETS:
                nodes.append(node)
                continue
            replacement = grouped_nodes(node, TARGETS[node.name], groups)
            for new in replacement:
                if new.name in existing or any(x in existing and x not in node.output for x in new.output):
                    raise ValueError(f"generated name collision: {new.name}")
            nodes.extend(replacement)
        del result.graph.node[:]
        result.graph.node.extend(nodes)
    onnx.checker.check_model(result, full_check=True)
    return result


def rewrite_file(input_path, output_path, groups=8):
    """Save exclusively to a new file; never overwrite a model or symlink."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve() or output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output must be a new path: {output_path}")
    model = onnx.load(input_path, load_external_data=False)
    result = rewrite_model(model, groups)
    with output_path.open("xb") as stream:
        stream.write(result.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()
    rewrite_file(args.input, args.output, args.groups)
    print(f"{args.input} -> {args.output} (groups={args.groups})")


if __name__ == "__main__":
    main()
