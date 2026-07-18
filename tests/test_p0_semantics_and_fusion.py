# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

# torchview is a lazily exposed vendored module and PyTorch's generated
# functional bindings do not carry signatures that pylint can infer.
# pylint: disable=no-name-in-module,not-callable

from pathlib import Path

import pytest
import torch
from torch import nn

from solar.analysis.fusion import FusionPlanner
from solar.analysis.orojenesis import (
    OrojenesisError,
    OrojenesisRunner,
    compose_multi_einsum_curve,
    compose_multi_einsum_region_curve,
    find_multi_einsum_chains,
    find_multi_einsum_regions,
    multi_einsum_mapper_role,
    multi_einsum_problem,
    multi_einsum_region_problem,
    parse_multi_mapping_records,
    select_capacity_point,
)
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


def _matmul(
    left: str,
    right: str,
    output: str,
    m_size: int,
    k_size: int,
    n_size: int,
) -> dict:
    return {
        "type": "matmul",
        "is_real_einsum": True,
        "is_einsum_supportable": True,
        "einsum_equation": "MK,KN->MN",
        "semantic_op": {
            "kind": "einsum",
            "target": "matmul",
            "equation": "MK,KN->MN",
            "arguments": [{"tensor": 0}, {"tensor": 1}],
            "kwargs": {},
            "effects": {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {"inputs": [left, right], "outputs": [output]},
        "tensor_shapes": {
            "inputs": [[m_size, k_size], [k_size, n_size]],
            "outputs": [[m_size, n_size]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float32", "torch.float32"],
            "outputs": ["torch.float32"],
        },
        "connections": {"inputs": [], "outputs": []},
    }


def _mapping_row(
    *,
    buffer_bytes: int,
    mapping: str,
    weight_util: int,
    input_util: int,
    output_util: int,
    weight_accesses: int,
    input_accesses: int,
    output_accesses: int,
) -> str:
    row: list[object] = [0] * 24
    row[0] = buffer_bytes
    row[1] = 1
    row[2] = weight_accesses + input_accesses + output_accesses
    row[3] = mapping
    row[5] = 1
    row[6] = weight_util
    row[10] = input_util
    row[11] = output_util
    row[21] = weight_accesses
    row[22] = input_accesses
    row[23] = output_accesses
    return ",".join(str(value) for value in row) + "\n"


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


def test_multi_einsum_chain_and_compatible_tile_curve(tmp_path: Path) -> None:
    assert [multi_einsum_mapper_role(index, 5) for index in range(5)] == [
        "first",
        "second",
        "middle",
        "middle",
        "last",
    ]
    first_mapper = OrojenesisRunner.multi_mapper_config(2, role="first")
    second_last_mapper = OrojenesisRunner.multi_mapper_config(2, role="second_last")
    assert first_mapper["mapspace_constraints"][3] == {
        "target": "MainMemory",
        "type": "temporal",
        "permutation": "KNM",
    }
    assert second_last_mapper["mapspace_constraints"][3]["permutation"] == "NKM"
    assert first_mapper["mapspace_constraints"][4]["permutation"] == "MNK"

    layers = {
        "start_x": _start("x", [4, 8]),
        "start_w1": _start("w1", [8, 16]),
        "start_w2": _start("w2", [16, 2]),
        "first": _matmul("x", "w1", "hidden", 4, 8, 16),
        "second": _matmul("hidden", "w2", "output", 4, 16, 2),
    }
    layers["first"]["connections"] = {
        "inputs": ["start_x", "start_w1"],
        "outputs": ["second"],
    }
    layers["second"]["connections"] = {
        "inputs": ["first", "start_w2"],
        "outputs": [],
    }
    assert find_multi_einsum_chains(layers) == [["first", "second"]]
    problem = multi_einsum_problem(
        [(layer_id, layers[layer_id]) for layer_id in ("first", "second")]
    )
    assert problem["chain"]["layers"][0]["output"] == "hidden"
    assert problem["chain"]["layers"][1]["input"] == "hidden"
    fusion = FusionPlanner(
        {"schema_version": 3, "layers": layers},
        multi_einsum_chains=[["first", "second"]],
    ).plan([])
    assert fusion["decisions"] == [
        {
            "producer": "first",
            "consumer": "second",
            "legal": True,
            "reason": "verified_multi_einsum_chain",
        }
    ]
    assert fusion["regions"][0]["layers"] == ["first", "second"]

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        _mapping_row(
            buffer_bytes=64,
            mapping="first-map",
            weight_util=16,
            input_util=8,
            output_util=32,
            weight_accesses=10,
            input_accesses=20,
            output_accesses=30,
        )
    )
    second.write_text(
        _mapping_row(
            buffer_bytes=96,
            mapping="second-map",
            weight_util=8,
            input_util=32,
            output_util=4,
            weight_accesses=11,
            input_accesses=30,
            output_accesses=12,
        )
    )
    curve = compose_multi_einsum_curve([[first], [second]], row_tiles=[1], word_bytes=2)
    assert curve == [
        {
            "buffer_bytes": 160,
            "operational_intensity": 2 / 106,
            "dram_accesses_words": 53.0,
            "dram_bytes": 106.0,
            "row_tile": 1,
            "mappings": ["first-map", "second-map"],
        }
    ]

    malformed = tmp_path / "malformed.csv"
    row = first.read_text().split(",")
    row[2] = str(float(row[2]) + 1)
    malformed.write_text(",".join(row))
    with pytest.raises(OrojenesisError, match="access fields disagree"):
        parse_multi_mapping_records(malformed, word_bytes=2)

    branched = copy.deepcopy(layers)
    branched["observer"] = _aten("relu", ["hidden"], ["observed"], [[4, 16]], [[4, 16]])
    assert find_multi_einsum_chains(branched) == []

    mixed_dtype = copy.deepcopy(layers)
    mixed_dtype["second"]["tensor_dtypes"] = {
        "inputs": ["torch.bfloat16", "torch.bfloat16"],
        "outputs": ["torch.bfloat16"],
    }
    assert find_multi_einsum_chains(mixed_dtype) == []

    embedded = copy.deepcopy(layers)
    embedded["observer"] = _aten("relu", ["output"], ["observed"], [[4, 2]], [[4, 2]])
    assert find_multi_einsum_chains(embedded) == []


def _region_einsum(
    equation: str,
    input_names: list[str],
    output_name: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
    *,
    dtype: str = "torch.float16",
) -> dict:
    return {
        "type": "einsum",
        "semantic_op": {
            "kind": "einsum",
            "target": "einsum",
            "equation": equation,
            "arguments": [{"tensor": index} for index in range(len(input_names))],
            "kwargs": {},
            "effects": {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {"inputs": input_names, "outputs": [output_name]},
        "tensor_shapes": {"inputs": input_shapes, "outputs": [output_shape]},
        "tensor_dtypes": {
            "inputs": [dtype] * len(input_names),
            "outputs": [dtype],
        },
        "connections": {"inputs": [], "outputs": []},
    }


def test_extended_multi_einsum_regions_cover_layout_batch_and_fanout(
    tmp_path: Path,
) -> None:
    layout_layers = {
        "x": _start("x", [4, 8], "torch.float16"),
        "w1": _start("w1", [16, 8], "torch.float16"),
        "w2": _start("w2", [16, 2], "torch.float16"),
        "first": _region_einsum(
            "MK,NK->MN", ["x", "w1"], "hidden", [[4, 8], [16, 8]], [4, 16]
        ),
        "view": _aten(
            "view",
            ["hidden"],
            ["viewed"],
            [[4, 16]],
            [[4, 16]],
            kwargs={"shape": [4, 16]},
        ),
        "second": _region_einsum(
            "MK,KN->MN", ["viewed", "w2"], "output", [[4, 16], [16, 2]], [4, 2]
        ),
    }
    layout_layers["view"]["tensor_dtypes"] = {
        "inputs": ["torch.float16"],
        "outputs": ["torch.float16"],
    }
    layout_layers["view"]["semantic_op"]["effects"]["aliases"] = [
        {"output": 0, "input": 0, "conditional": False}
    ]
    layout_layers["first"]["connections"] = {
        "inputs": ["x", "w1"],
        "outputs": ["view"],
    }
    layout_layers["view"]["connections"] = {
        "inputs": ["first"],
        "outputs": ["second"],
    }
    layout_layers["second"]["connections"] = {
        "inputs": ["view", "w2"],
        "outputs": [],
    }
    regions = find_multi_einsum_regions(layout_layers)
    assert len(regions) == 1
    layout = multi_einsum_region_problem(regions[0])
    assert layout["kind"] == "linear_matmul_with_axis_maps"
    assert layout["edges"][0]["bridges"] == ["view"]
    assert layout["edges"][0]["axis_map"] == [0, 1]

    fusion = FusionPlanner(
        {"schema_version": 3, "layers": layout_layers},
        multi_einsum_chains=layout["physical_paths"],
        verified_view_nodes=["view"],
    ).plan([])
    assert {"first", "view", "second"}.issubset(set(fusion["regions"][0]["layers"]))

    batch_layers = {
        "x": _start("x", [2, 4, 8], "torch.float16"),
        "w1": _start("w1", [8, 16], "torch.float16"),
        "w2": _start("w2", [16, 2], "torch.float16"),
        "first": _region_einsum(
            "BMK,KN->BMN",
            ["x", "w1"],
            "hidden",
            [[2, 4, 8], [8, 16]],
            [2, 4, 16],
        ),
        "view": _aten(
            "view",
            ["hidden"],
            ["viewed"],
            [[2, 4, 16]],
            [[8, 16]],
            kwargs={"shape": [8, 16]},
        ),
        "second": _region_einsum(
            "MK,KN->MN", ["viewed", "w2"], "output", [[8, 16], [16, 2]], [8, 2]
        ),
    }
    batch_layers["view"]["tensor_dtypes"] = {
        "inputs": ["torch.float16"],
        "outputs": ["torch.float16"],
    }
    batch_layers["view"]["semantic_op"]["effects"]["aliases"] = [
        {"output": 0, "input": 0, "conditional": False}
    ]
    batch = find_multi_einsum_regions(batch_layers)
    assert len(batch) == 1
    assert batch[0]["kind"] == "broadcast_batch_linear_matmul"
    assert batch[0]["nodes"][0]["m"] == 8

    fanout_layers = {
        "x": _start("x", [4, 8], "torch.float16"),
        "w0": _start("w0", [8, 16], "torch.float16"),
        "w1": _start("w1", [16, 2], "torch.float16"),
        "w2": _start("w2", [16, 3], "torch.float16"),
        "root": _region_einsum(
            "MK,KN->MN", ["x", "w0"], "hidden", [[4, 8], [8, 16]], [4, 16]
        ),
        "left": _region_einsum(
            "MK,KN->MN", ["hidden", "w1"], "left_out", [[4, 16], [16, 2]], [4, 2]
        ),
        "right": _region_einsum(
            "MK,KN->MN", ["hidden", "w2"], "right_out", [[4, 16], [16, 3]], [4, 3]
        ),
    }
    fanout = find_multi_einsum_regions(fanout_layers)
    assert len(fanout) == 1
    assert fanout[0]["kind"] == "matmul_fanout_tree"
    assert fanout[0]["schedule"] == ["root", "left", "right"]

    raw_paths: dict[str, list[Path]] = {}
    row_tiles: dict[str, list[int]] = {}
    for index, node in enumerate(fanout[0]["nodes"]):
        node_id = node["id"]
        path = tmp_path / f"{node_id}.csv"
        path.write_text(
            _mapping_row(
                buffer_bytes=64 + index,
                mapping=f"{node_id}-map",
                weight_util=int(node["k"]) * int(node["n"]) * 2,
                input_util=int(node["k"]) * 2,
                output_util=int(node["n"]) * 2,
                weight_accesses=10 + index,
                input_accesses=20,
                output_accesses=30 + index,
            )
        )
        raw_paths[node_id] = [path]
        row_tiles[node_id] = [1]
    curve = compose_multi_einsum_region_curve(
        fanout[0], raw_paths, row_tiles_by_node=row_tiles, word_bytes=2
    )
    assert curve[0]["mappings"] == ["root-map", "left-map", "right-map"]
    assert curve[0]["dram_accesses_words"] == 10 + 20 + 11 + 31 + 12 + 32

    conditional = copy.deepcopy(layout_layers)
    conditional["view"]["semantic_op"]["effects"]["aliases"][0]["conditional"] = True
    assert find_multi_einsum_regions(conditional) == []

    missing_alias = copy.deepcopy(layout_layers)
    missing_alias["view"]["semantic_op"]["effects"]["aliases"] = []
    assert find_multi_einsum_regions(missing_alias) == []

    transposed = copy.deepcopy(layout_layers)
    transposed["view"] = _aten(
        "transpose",
        ["hidden"],
        ["viewed"],
        [[4, 16]],
        [[16, 4]],
        arguments=[{"tensor": 0}, {"value": 0}, {"value": 1}],
    )
    transposed["view"]["tensor_dtypes"] = {
        "inputs": ["torch.float16"],
        "outputs": ["torch.float16"],
    }
    transposed["view"]["semantic_op"]["effects"]["aliases"] = [
        {"output": 0, "input": 0, "conditional": False}
    ]
    transposed["second"] = _region_einsum(
        "MK,KN->MN", ["viewed", "w2"], "output", [[16, 4], [4, 2]], [16, 2]
    )
    transposed["w2"] = _start("w2", [4, 2], "torch.float16")
    transpose_regions = find_multi_einsum_regions(transposed)
    assert len(transpose_regions) == 1
    assert transpose_regions[0]["edges"][0]["axis_map"] == [1, 0]


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


def test_variance_dim_overload_is_preserved_and_executable(tmp_path: Path) -> None:
    class Variance(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.var(value, dim=-1, keepdim=True, unbiased=False)

    model = Variance()
    value = torch.randn(3, 5)
    graph = _strict_graph(model, [value], tmp_path)
    variance = next(
        layer for layer in graph["layers"].values() if layer["type"] == "var"
    )
    assert variance["semantic_op"]["overload"] == "dim"
    torch.testing.assert_close(EinsumGraphExecutor(graph)(value), model(value))


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
