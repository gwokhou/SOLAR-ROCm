# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# torchview is a lazily exposed vendored module and PyTorch's generated
# functional bindings do not carry signatures that pylint can infer.
# pylint: disable=no-name-in-module,not-callable

from pathlib import Path

import pytest
import torch
from torch import nn

from solar.analysis.fusion import FusionPlanner
from solar.analysis.orojenesis import OrojenesisRunner, select_capacity_point
from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.einsum import PyTorchToEinsum
from solar.graph.torchview_processor import TorchviewProcessor
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
    assert fused_region["capacity"]["lds"]["capacity_pressure_bytes"] > 0
    assert fused_region["prefetched_bytes"] == fused_region["fused_bytes"]


def test_view_alias_is_a_fusion_barrier() -> None:
    input_layer = _start("x", [2, 2])
    input_layer["connections"]["outputs"] = ["view"]
    view = _aten("view", ["x"], ["y"], [[2, 2]], [[4]], kwargs={"shape": [4]})
    view["semantic_op"]["effects"]["aliases"] = [
        {"output": 0, "input": 0, "conditional": False}
    ]
    view["connections"] = {"inputs": ["input"], "outputs": ["relu"]}
    relu = _aten("relu", ["y"], ["z"], [[4]], [[4]])
    relu["connections"] = {"inputs": ["view"], "outputs": []}
    graph = {
        "schema_version": 3,
        "layers": {"input": input_layer, "view": view, "relu": relu},
    }
    result = FusionPlanner(graph).plan([])
    decision = next(item for item in result["decisions"] if item["producer"] == "view")
    assert decision == {
        "producer": "view",
        "consumer": "relu",
        "legal": False,
        "reason": "observable_alias",
    }


def test_orojenesis_parser_keeps_pareto_points(tmp_path: Path) -> None:
    source = tmp_path / "oaves.csv"
    source.write_text(
        "64,2,100,map,None,10\n64,3,80,map,None,10\n128,4,70,map,None,10\n256,3,90,map,None,10\n"
    )
    curve = OrojenesisRunner.parse_curve(source, word_bytes=2)
    assert [item["buffer_bytes"] for item in curve] == [64, 128]
    assert select_capacity_point(curve, 100)["dram_bytes"] == 160


def test_strict_trace_preserves_structured_call_semantics(tmp_path: Path) -> None:
    from solar._vendor import torchview

    class StructuredModel(nn.Module):
        def forward(self, value, index):
            normalized = torch.softmax(value, dim=0)
            viewed = normalized.view(2, 6)
            return torch.gather(viewed, 1, index)

    model = StructuredModel().eval()
    value = torch.randn(3, 4)
    index = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    traced = torchview.draw_graph(
        model,
        input_data=[value, index],
        save_graph=False,
        expand_nested=True,
        depth=float("inf"),
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=False,
        strict=True,
        collect_attributes=True,
    )
    TorchviewProcessor().process_graph(traced, str(tmp_path), "structured", model)
    graph = PyTorchToEinsum(strict=True).convert(
        tmp_path / "pytorch_graph.yaml",
        tmp_path,
        copy_graph=False,
        enable_rename=False,
    )
    assert graph is not None
    gather = next(
        layer for layer in graph["layers"].values() if layer["type"] == "gather"
    )
    assert gather["semantic_op"]["arguments"] == [
        {"tensor": 0},
        {"value": 1},
        {"tensor": 1},
    ]
    assert gather["tensor_dtypes"]["inputs"] == ["torch.float32", "torch.int64"]
    torch.testing.assert_close(
        EinsumGraphExecutor(graph)(value, index), model(value, index)
    )


def _strict_graph(model: nn.Module, inputs: list[torch.Tensor], path: Path) -> dict:
    from solar._vendor import torchview

    traced = torchview.draw_graph(
        model.eval(),
        input_data=inputs,
        device="cpu",
        save_graph=False,
        expand_nested=True,
        depth=float("inf"),
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=False,
        strict=True,
        collect_attributes=True,
    )
    TorchviewProcessor().process_graph(traced, str(path), "p0_graph", model)
    graph = PyTorchToEinsum(strict=True).convert(
        path / "pytorch_graph.yaml",
        path,
        copy_graph=False,
        enable_rename=False,
    )
    assert graph is not None
    return graph


def test_strict_whole_graph_layer_norm_is_executable(tmp_path: Path) -> None:
    model = nn.LayerNorm(4)
    value = torch.randn(2, 4)
    graph = _strict_graph(model, [value], tmp_path)
    assert any(layer["type"] == "layer_norm" for layer in graph["layers"].values())
    torch.testing.assert_close(
        EinsumGraphExecutor(graph)(value, model.weight, model.bias), model(value)
    )


def test_tensor_t_descriptor_preserves_square_transpose(tmp_path: Path) -> None:
    class SquareTranspose(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.T

    model = SquareTranspose()
    value = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    graph = _strict_graph(model, [value], tmp_path)
    transpose = next(
        layer for layer in graph["layers"].values() if layer["type"] == "__get__"
    )
    assert transpose["einsum_equation"] == "AB->BA"
    assert transpose["semantic_op"]["target"] == "transpose"
    torch.testing.assert_close(EinsumGraphExecutor(graph)(value), model(value))


def test_strict_whole_graph_embedding_is_executable(tmp_path: Path) -> None:
    model = nn.Embedding(17, 4)
    index = torch.tensor([[0, 3, 5], [16, 2, 1]], dtype=torch.int64)
    graph = _strict_graph(model, [index], tmp_path)
    embedding = next(
        layer for layer in graph["layers"].values() if layer["type"] == "embedding"
    )
    assert embedding["tensor_dtypes"]["inputs"] == [
        "torch.int64",
        "torch.float32",
    ]
    torch.testing.assert_close(
        EinsumGraphExecutor(graph)(index, model.weight), model(index)
    )


def test_strict_whole_graph_conv2d_preserves_call_parameters(tmp_path: Path) -> None:
    model = nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1, bias=True)
    value = torch.randn(1, 2, 7, 7)
    graph = _strict_graph(model, [value], tmp_path)
    convolution = next(
        layer for layer in graph["layers"].values() if layer["type"] == "conv2d"
    )
    assert convolution["semantic_op"]["kind"] == "aten"
    assert convolution["force_aten_semantics"] is True
    torch.testing.assert_close(
        EinsumGraphExecutor(graph)(value, model.weight, model.bias), model(value)
    )


def test_strict_whole_graph_attention_preserves_mask_scale_and_causal(
    tmp_path: Path,
) -> None:
    class Attention(nn.Module):
        def forward(self, query, key, value, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=0.25,
            )

    model = Attention()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 5)
    mask = torch.tensor(
        [[[[True, True, False], [True, True, True], [False, True, True]]]]
    )
    inputs = [query, key, value, mask]
    graph = _strict_graph(model, inputs, tmp_path)
    attention = next(
        layer
        for layer in graph["layers"].values()
        if layer["type"] == "scaled_dot_product_attention"
    )
    assert attention["semantic_op"]["kind"] == "aten"
    assert attention["semantic_op"]["kwargs"]["scale"] == {"value": 0.25}
    torch.testing.assert_close(EinsumGraphExecutor(graph)(*inputs), model(*inputs))


def test_out_of_place_scatter_is_atomic_but_not_a_mutation(tmp_path: Path) -> None:
    class Scatter(nn.Module):
        def forward(self, value, index, source):
            return torch.scatter(value, 0, index, source)

    model = Scatter()
    value = torch.zeros(4, 2)
    index = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    source = torch.randn(2, 2)
    graph = _strict_graph(model, [value, index, source], tmp_path)
    scatter = next(
        layer for layer in graph["layers"].values() if layer["type"] == "scatter"
    )
    assert scatter["semantic_op"]["effects"]["mutates"] == []
    assert scatter["semantic_op"]["effects"]["atomic"] is True
    executor_inputs = [value.clone(), index.clone(), source.clone()]
    actual = EinsumGraphExecutor(graph)(*executor_inputs)
    torch.testing.assert_close(actual, model(value, index, source))
    torch.testing.assert_close(executor_inputs[0], value)


def test_in_place_operation_preserves_mutation_semantics(tmp_path: Path) -> None:
    class InPlaceAdd(nn.Module):
        def forward(self, value, update):
            return value.add_(update)

    model = InPlaceAdd()
    trace_inputs = [torch.randn(4), torch.randn(4)]
    graph = _strict_graph(model, [item.clone() for item in trace_inputs], tmp_path)
    operation = next(
        layer
        for layer in graph["layers"].values()
        if layer.get("mutates_inputs") is True
    )
    assert operation["semantic_op"]["target"] == "add"
    assert operation["semantic_op"]["effects"]["mutates"] == [0]
    executor_inputs = [item.clone() for item in trace_inputs]
    expected_inputs = [item.clone() for item in trace_inputs]
    expected = model(*expected_inputs)
    actual = EinsumGraphExecutor(graph)(*executor_inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(executor_inputs[0], expected_inputs[0])


def test_quantize_dequantize_semantics_are_executable() -> None:
    quantize = _aten(
        "quantize_per_tensor",
        ["x"],
        ["qx"],
        [[4]],
        [[4]],
        arguments=[
            {"tensor": 0},
            {"value": 0.1},
            {"value": 0},
            {"dtype": "qint8"},
        ],
    )
    quantize["tensor_dtypes"]["outputs"] = ["torch.qint8"]
    quantize["connections"] = {"inputs": ["input"], "outputs": ["dequantize"]}
    dequantize = _aten(
        "dequantize", ["qx"], ["y"], [[4]], [[4]], arguments=[{"tensor": 0}]
    )
    dequantize["tensor_dtypes"]["inputs"] = ["torch.qint8"]
    dequantize["connections"] = {"inputs": ["quantize"], "outputs": []}
    graph = {
        "schema_version": 3,
        "layers": {
            "input": _start("x", [4]),
            "quantize": quantize,
            "dequantize": dequantize,
        },
    }
    value = torch.tensor([-0.25, -0.05, 0.15, 0.35])
    expected = torch.quantize_per_tensor(value, 0.1, 0, torch.qint8).dequantize()
    torch.testing.assert_close(EinsumGraphExecutor(graph)(value), expected)


def test_strict_whole_graph_cat_preserves_tensor_list_and_dim(tmp_path: Path) -> None:
    class Concatenate(nn.Module):
        def forward(self, left, right):
            return torch.cat([left, right], dim=1)

    model = Concatenate()
    inputs = [torch.randn(2, 3), torch.randn(2, 4)]
    graph = _strict_graph(model, inputs, tmp_path)
    cat = next(layer for layer in graph["layers"].values() if layer["type"] == "cat")
    assert cat["semantic_op"]["arguments"][0] == [
        {"tensor": 0},
        {"tensor": 1},
    ]
    torch.testing.assert_close(EinsumGraphExecutor(graph)(*inputs), model(*inputs))


def test_strict_whole_graph_dtype_conversion_is_executable(tmp_path: Path) -> None:
    class Convert(nn.Module):
        def forward(self, value):
            return value.to(dtype=torch.float16)

    model = Convert()
    value = torch.randn(2, 3)
    graph = _strict_graph(model, [value], tmp_path)
    conversion = next(
        layer for layer in graph["layers"].values() if layer["type"] == "to"
    )
    assert conversion["semantic_op"]["kwargs"]["dtype"] == {"dtype": "float16"}
    torch.testing.assert_close(EinsumGraphExecutor(graph)(value), model(value))


def test_strict_whole_graph_view_records_observable_alias(tmp_path: Path) -> None:
    class View(nn.Module):
        def forward(self, value):
            return value.view(2, 6)

    model = View()
    value = torch.randn(3, 4)
    graph = _strict_graph(model, [value], tmp_path)
    view = next(layer for layer in graph["layers"].values() if layer["type"] == "view")
    assert view["semantic_op"]["effects"]["aliases"] == [
        {"output": 0, "input": 0, "conditional": False}
    ]
    actual = EinsumGraphExecutor(graph)(value)
    torch.testing.assert_close(actual, model(value))
    assert actual.untyped_storage()._cdata == value.untyped_storage()._cdata
