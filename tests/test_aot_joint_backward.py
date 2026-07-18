# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import yaml
from torch import nn

from solar.graph.backward_processor import BackwardProcessor


def test_backward_processor_exports_verified_aot_joint_graph(tmp_path) -> None:
    model = nn.Linear(4, 2)
    inputs = [torch.randn(3, 4)]
    target = torch.randn(3, 2)
    graph = BackwardProcessor().extract_backward_graph(
        model,
        inputs,
        lambda output, expected: torch.nn.functional.mse_loss(output, expected),
        target,
        str(tmp_path),
        "linear",
    )
    assert graph is not None
    assert graph["schema_version"] == 3
    assert graph["joint_graph"] is True
    assert graph["graph_signature"]["gradients_to_parameters"]
    assert graph["graph_signature"]["gradients_to_user_inputs"]
    assert any(layer.get("phase") == "backward" for layer in graph["layers"].values())
    assert graph["outputs"] == graph["graph_signature"]["joint_outputs"]
    assert any(
        (layer.get("semantic_op") or {}).get("effects", {}).get("aliases")
        for layer in graph["layers"].values()
    )
    assert (tmp_path / "joint_graph.yaml").is_file()
    assert yaml.safe_load((tmp_path / "joint_graph.yaml").read_text()) == graph


class _NormResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value) + value


class _ScatterRoute(nn.Module):
    def forward(self, value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value).scatter_add(0, index, value)


class _BufferMutation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("accumulator", torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.accumulator.add_(value.detach().sum())
        return value * 2


@torch.no_grad()
def _target_like(model: nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    return torch.randn_like(model(*inputs))


@torch.no_grad()
def _integer_route() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [2, 2, 2],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )


def test_aot_joint_graph_executes_softmax_backward_from_yaml(tmp_path) -> None:
    model = nn.Softmax(dim=-1)
    inputs = [torch.randn(3, 4)]
    graph = BackwardProcessor().extract_backward_graph(
        model,
        inputs,
        lambda output, expected: torch.nn.functional.mse_loss(output, expected),
        _target_like(model, inputs),
        str(tmp_path),
        "softmax_backward",
    )
    targets = {layer["semantic_op"]["target"] for layer in graph["layers"].values()}
    assert "_softmax_backward_data" in targets
    assert graph["graph_signature"]["gradients_to_user_inputs"]


def test_aot_joint_graph_executes_norm_residual_backward_from_yaml(tmp_path) -> None:
    model = _NormResidual()
    inputs = [torch.randn(3, 4)]
    graph = BackwardProcessor().extract_backward_graph(
        model,
        inputs,
        lambda output, expected: torch.nn.functional.mse_loss(output, expected),
        _target_like(model, inputs),
        str(tmp_path),
        "norm_residual_backward",
    )
    assert graph["graph_signature"]["gradients_to_parameters"]
    assert graph["graph_signature"]["gradients_to_user_inputs"]
    assert any(layer["type"] == "identity" for layer in graph["layers"].values())


def test_aot_joint_graph_executes_moe_scatter_backward_from_yaml(tmp_path) -> None:
    model = _ScatterRoute()
    inputs = [torch.randn(6, 3), _integer_route()]
    graph = BackwardProcessor().extract_backward_graph(
        model,
        inputs,
        lambda output, expected: torch.nn.functional.mse_loss(output, expected),
        _target_like(model, inputs),
        str(tmp_path),
        "moe_scatter_backward",
    )
    targets = {layer["semantic_op"]["target"] for layer in graph["layers"].values()}
    assert "scatter_add" in targets
    assert "gather" in targets


def test_aot_joint_graph_preserves_functionalized_buffer_mutation(tmp_path) -> None:
    model = _BufferMutation()
    inputs = [torch.randn(3, 4)]
    graph = BackwardProcessor().extract_backward_graph(
        model,
        inputs,
        lambda output, expected: torch.nn.functional.mse_loss(output, expected),
        _target_like(model, inputs),
        str(tmp_path),
        "buffer_mutation_backward",
    )
    mutations = graph["graph_signature"]["buffers_to_mutate"]
    assert set(mutations.values()) == {"model.accumulator"}
    assert set(mutations).issubset(graph["outputs"])
    assert graph["graph_signature"]["gradients_to_user_inputs"]
