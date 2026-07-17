# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from solar.analysis.fusion import FusionPlanner
from solar.analysis.orojenesis import OrojenesisRunner, select_capacity_point
from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.rocm.architecture import MemoryLevel
from solar.verification import EinsumGraphExecutor


def _start(name: str, shape: list[int], dtype: str = "torch.float32") -> dict:
    return {
        "type": "start",
        "semantic_op": {
            "kind": "input",
            "target": "input",
            "arguments": [],
            "kwargs": {},
        },
        "tensor_names": {"inputs": [], "outputs": [name]},
        "tensor_shapes": {"inputs": [], "outputs": [shape]},
        "tensor_dtypes": {"inputs": [], "outputs": [dtype]},
        "connections": {"inputs": [], "outputs": []},
    }


def _aten(
    target: str,
    inputs: list[str],
    outputs: list[str],
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    *,
    arguments: list[dict] | None = None,
    kwargs: dict | None = None,
    effects: dict | None = None,
) -> dict:
    return {
        "type": target,
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "einsum_equation": "A->A",
        "semantic_op": {
            "kind": "aten",
            "target": target,
            "overload": "default",
            "arguments": arguments or [{"tensor": i} for i in range(len(inputs))],
            "kwargs": kwargs or {},
            "effects": effects
            or {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {"inputs": inputs, "outputs": outputs},
        "tensor_shapes": {"inputs": input_shapes, "outputs": output_shapes},
        "tensor_dtypes": {
            "inputs": ["torch.float32"] * len(inputs),
            "outputs": ["torch.float32"] * len(outputs),
        },
        "connections": {"inputs": [], "outputs": []},
    }


def test_executor_uses_explicit_softmax_dim() -> None:
    graph = {
        "schema_version": 3,
        "layers": {
            "input": _start("x", [2, 3]),
            "softmax": _aten(
                "softmax",
                ["x"],
                ["y"],
                [[2, 3]],
                [[2, 3]],
                kwargs={"dim": {"value": 0}},
            ),
        },
    }
    value = torch.randn(2, 3)
    torch.testing.assert_close(
        EinsumGraphExecutor(graph)(value), torch.softmax(value, dim=0)
    )


def test_executor_supports_multiple_outputs() -> None:
    graph = {
        "schema_version": 3,
        "layers": {
            "input": _start("x", [4]),
            "split": _aten(
                "split",
                ["x"],
                ["left", "right"],
                [[4]],
                [[2], [2]],
                arguments=[{"tensor": 0}, {"value": 2}],
            ),
        },
    }
    value = torch.arange(4, dtype=torch.float32)
    actual = EinsumGraphExecutor(graph)(value)
    expected = torch.split(value, 2)
    assert len(actual) == 2
    torch.testing.assert_close(actual, expected)


def test_unknown_operation_is_not_a_copy_fallback() -> None:
    with pytest.raises(ValueError, match="Unsupported operation"):
        EinsumAnalyzer().get_einsum_op(
            "definitely_unknown", TensorShapes(inputs=[[4]], outputs=[[4]])
        )


def test_fusion_records_barriers_and_capacity_spill() -> None:
    input_layer = _start("x", [1024])
    input_layer["connections"]["outputs"] = ["add"]
    add = _aten("add", ["x", "x"], ["a"], [[1024], [1024]], [[1024]])
    add["connections"] = {"inputs": ["input"], "outputs": ["relu"]}
    relu = _aten("relu", ["a"], ["b"], [[1024]], [[1024]])
    relu["connections"] = {"inputs": ["add"], "outputs": ["scatter"]}
    scatter = _aten(
        "scatter",
        ["b"],
        ["c"],
        [[1024]],
        [[1024]],
        effects={
            "mutates": [0],
            "aliases": [],
            "atomic": True,
            "opaque_library_call": False,
        },
    )
    scatter["connections"] = {"inputs": ["relu"], "outputs": []}
    graph = {
        "schema_version": 3,
        "layers": {"input": input_layer, "add": add, "relu": relu, "scatter": scatter},
    }
    result = FusionPlanner(graph).plan(
        [MemoryLevel("lds", "compute_unit", 1024, source="test")]
    )
    assert any(item["legal"] for item in result["decisions"])
    assert any(item["reason"] in {"mutation", "atomic"} for item in result["decisions"])
    fused_region = next(
        region for region in result["regions"] if "add" in region["layers"]
    )
    assert fused_region["capacity"]["lds"]["spill_bytes_lower_bound"] > 0


def test_orojenesis_parser_keeps_pareto_points(tmp_path: Path) -> None:
    source = tmp_path / "oaves.csv"
    source.write_text(
        "64,2,100,map,None,10\n64,3,80,map,None,10\n128,4,70,map,None,10\n256,3,90,map,None,10\n"
    )
    curve = OrojenesisRunner.parse_curve(source, word_bytes=2)
    assert [item["buffer_bytes"] for item in curve] == [64, 128]
    assert select_capacity_point(curve, 100)["dram_bytes"] == 160
