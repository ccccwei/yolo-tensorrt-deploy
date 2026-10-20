#!/usr/bin/env python3
"""CPU-only graph surgery and tie-aware exact selection tests (ONNX + NumPy)."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.grouped_topk import TARGETS, grouped_nodes, rewrite_file, rewrite_model
from topk_cases import assert_selection, score_cases


class GatherElements(OpRun):
    """ONNX 1.22's reference np.choose path fails for >64 elements on NumPy 2."""

    op_domain = ""

    def _run(self, data, indices, axis=0):
        return (np.take_along_axis(data, indices, axis=axis),)


def selector(width, groups):
    node = helper.make_node("TopK", ["scores", "k"], ["values", "indices"],
                            name="selection", axis=1, largest=1, sorted=1)
    graph = helper.make_graph(
        grouped_nodes(node, width, groups), "isolated_selector",
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 300]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 300])],
        [numpy_helper.from_array(np.array([300], dtype=np.int64), "k")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model, full_check=True)
    return ReferenceEvaluator(model, new_ops=[GatherElements])


def fixture(widths=(8400, 24000), k=300):
    """Small valid graph with the production IO/TopK contracts; no model weights."""
    nodes = []
    initializers = [numpy_helper.from_array(np.array([k], dtype=np.int64), "k")]
    for index, (name, width) in enumerate(zip(TARGETS, widths)):
        score = f"scores_{index}"
        initializers.append(numpy_helper.from_array(np.arange(width, dtype=np.float32)[None], score))
        nodes.append(helper.make_node("TopK", [score, "k"], [name + "_values", name + "_indices"],
                                      name=name, axis=1, largest=1, sorted=1))
    initializers.append(numpy_helper.from_array(np.zeros((1, 300, 6), dtype=np.float32), "unrelated"))
    nodes.append(helper.make_node("Identity", ["unrelated"], ["output0"], name="unrelated_identity"))
    graph = helper.make_graph(nodes, "fixture",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 300, 6])],
                              initializers)
    graph.doc_string = "Unrelated graph metadata must survive unchanged."
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], producer_name="test_fixture")


class GroupedTopKTest(unittest.TestCase):
    def test_numeric_cases(self):
        for width in TARGETS.values():
            plain, grouped = selector(width, 1), selector(width, 8)
            for name, scores in score_cases(width).items():
                with self.subTest(width=width, case=name):
                    plain_values, plain_indices = plain.run(None, {"scores": scores})
                    values, indices = grouped.run(None, {"scores": scores})
                    assert_selection(scores, values, indices)
                    assert_selection(scores, plain_values, plain_indices)
                    np.testing.assert_array_equal(values, plain_values)
                    if name == "unique":
                        np.testing.assert_array_equal(indices, plain_indices)

    def test_minimum_group_size(self):
        for width, groups in ((300, 1), (2400, 8)):
            scores = np.arange(width, dtype=np.float32)[None]
            values, indices = selector(width, groups).run(None, {"scores": scores})
            assert_selection(scores, values, indices)
        with self.assertRaises(ValueError):
            selector(2392, 8)  # 299 items per group cannot retain 300 winners.

    def test_preservation(self):
        model = fixture()
        before = model.SerializeToString()
        result = rewrite_model(model)
        self.assertEqual(before, model.SerializeToString())
        self.assertEqual(before, rewrite_model(model, 1).SerializeToString())
        self.assertEqual([x.SerializeToString() for x in model.graph.initializer],
                         [x.SerializeToString() for x in result.graph.initializer])
        original_nodes = {x.name: x.SerializeToString() for x in model.graph.node if x.name not in TARGETS}
        result_nodes = {x.name: x.SerializeToString() for x in result.graph.node}
        for name, value in original_nodes.items():
            self.assertEqual(value, result_nodes[name])
        # Everything except graph.node (including IO, metadata and initializers) is identical.
        left, right = deepcopy(model), deepcopy(result)
        del left.graph.node[:]
        del right.graph.node[:]
        self.assertEqual(left.SerializeToString(), right.SerializeToString())
        outputs = {name for node in result.graph.node for name in node.output}
        for node in model.graph.node:
            self.assertTrue(set(node.output).issubset(outputs))

    def test_invalid_contracts(self):
        for widths, k in (((8399, 24000), 300), ((8400, 24001), 300), ((8400, 24000), 299)):
            with self.subTest(widths=widths, k=k), self.assertRaises(ValueError):
                rewrite_model(fixture(widths, k))
        for groups in (0, -1, 7, 32, 100, True, 8.0):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                rewrite_model(fixture(), groups)
        for change in ("opset", "name", "largest", "extra_topk", "duplicate_name", "collision"):
            model = fixture()
            if change == "opset":
                model.opset_import[0].version = 17
            elif change == "name":
                model.graph.node[0].name = "unexpected"
            elif change == "largest":
                next(a for a in model.graph.node[0].attribute if a.name == "largest").i = 0
            elif change == "extra_topk":
                extra = deepcopy(model.graph.node[0])
                extra.name = "extra_topk"
                extra.output[:] = ["extra_values", "extra_indices"]
                model.graph.node.append(extra)
            elif change == "duplicate_name":
                model.graph.node[-1].name = "/model.23/TopK"
            else:
                model.graph.node[-1].name = "/model.23/TopK/grouped/shape/Constant"
            with self.subTest(change=change), self.assertRaises(ValueError):
                rewrite_model(model)

    def test_never_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="grouped-topk-test-") as directory:
            source, output = Path(directory) / "source.onnx", Path(directory) / "output.onnx"
            onnx.save(fixture(), source)
            before = source.read_bytes()
            rewrite_file(source, output)
            saved = output.read_bytes()
            for destination in (source, output):
                with self.assertRaises(FileExistsError):
                    rewrite_file(source, destination)
            alias = Path(directory) / "alias.onnx"
            alias.symlink_to(source)
            with self.assertRaises(FileExistsError):
                rewrite_file(source, alias)
            self.assertEqual(before, source.read_bytes())
            self.assertEqual(saved, output.read_bytes())
            bad_output = Path(directory) / "invalid.onnx"
            with self.assertRaises(ValueError):
                rewrite_file(source, bad_output, 7)
            self.assertFalse(bad_output.exists())


if __name__ == "__main__":
    unittest.main()
