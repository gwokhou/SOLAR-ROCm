# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
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
    assert (tmp_path / "joint_graph.yaml").is_file()
