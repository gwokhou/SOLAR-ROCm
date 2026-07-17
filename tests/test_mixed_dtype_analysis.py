# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.perf.perf_model import EinsumGraphPerfModel


def _matmul_layer(name: str, dtype: str) -> dict:
    return {
        "type": "matmul",
        "einsum_equation": "MK,KN->MN",
        "is_real_einsum": True,
        "tensor_names": {
            "inputs": [f"{name}_a", f"{name}_b"],
            "outputs": [f"{name}_output"],
        },
        "tensor_types": {"inputs": ["input", "input"], "outputs": ["output"]},
        "tensor_shapes": {"inputs": [[2, 2], [2, 2]], "outputs": [[2, 2]]},
        "connections": {
            "inputs": [f"{name}_start_a", f"{name}_start_b"],
            "outputs": [],
        },
        "tensor_dtypes": {
            "inputs": [f"torch.{dtype}", f"torch.{dtype}"],
            "outputs": [f"torch.{dtype}"],
        },
    }


def _start(name: str, consumer: str, dtype: str) -> dict:
    return {
        "type": "start",
        "is_real_einsum": False,
        "tensor_names": {"inputs": [], "outputs": [name]},
        "tensor_types": {"inputs": [], "outputs": ["input"]},
        "tensor_shapes": {"inputs": [], "outputs": [[2, 2]]},
        "connections": {"inputs": [], "outputs": [consumer]},
        "tensor_dtypes": {"inputs": [], "outputs": [f"torch.{dtype}"]},
    }


def test_mixed_dtype_bytes_and_compute_precision_are_artifact_authoritative(
    tmp_path,
):
    graph = {
        "layers": {
            "half_start_a": _start("half_a", "half", "float16"),
            "half_start_b": _start("half_b", "half", "float16"),
            "half": _matmul_layer("half", "float16"),
            "full_start_a": _start("full_a", "full", "float32"),
            "full_start_b": _start("full_b", "full", "float32"),
            "full": _matmul_layer("full", "float32"),
        }
    }
    graph_path = tmp_path / "einsum_graph.yaml"
    graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

    analysis = EinsumGraphAnalyzer().analyze_graph(
        graph_path, tmp_path / "analysis", precision="bf16", copy_graph=False
    )
    assert analysis is not None
    assert analysis["total"]["macs_by_precision"] == {"fp16": 8, "fp32": 8}
    assert analysis["total"]["fused_bytes"] == 72

    perf = EinsumGraphPerfModel().predict(
        tmp_path / "analysis" / "analysis.yaml",
        tmp_path / "perf",
        arch_config="RX_9060_XT",
        precision="fp8",
        copy_analysis=False,
    )
    assert perf is not None
    assert perf["arch"]["throughput_precision"] == "mixed"
    assert perf["fused"]["memory_bytes"] == 72
    assert perf["workload"]["memory_accounting"] == "per_tensor_dtype"
