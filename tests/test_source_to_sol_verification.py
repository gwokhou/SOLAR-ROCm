# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.benchmark.models import BenchmarkSpec
from solar.einsum import ConversionError, PyTorchToEinsum, annotate_semantics
from solar.verification import (
    VerificationError,
    create_verification_artifact,
    replay_verification_artifact,
)


def _write_reference(path: Path) -> None:
    path.write_text(
        "import torch\n"
        "def get_inputs(workload, device):\n"
        "    g = torch.Generator(device=device).manual_seed(workload['seed'])\n"
        "    n = workload['n']\n"
        "    return (torch.randn((n, n), generator=g, device=device), "
        "torch.randn((n, n), generator=g, device=device))\n"
        "def run(a, b): return torch.mm(a, b)\n"
    )


def _graph(n: int = 3) -> dict:
    dtype = "torch.float32"
    return annotate_semantics(
        {
            "layers": {
                "start_a": {
                    "type": "start",
                    "tensor_names": {"inputs": [], "outputs": ["a"]},
                    "tensor_shapes": {"inputs": [], "outputs": [[n, n]]},
                    "tensor_dtypes": {"inputs": [], "outputs": [dtype]},
                },
                "start_b": {
                    "type": "start",
                    "tensor_names": {"inputs": [], "outputs": ["b"]},
                    "tensor_shapes": {"inputs": [], "outputs": [[n, n]]},
                    "tensor_dtypes": {"inputs": [], "outputs": [dtype]},
                },
                "matmul": {
                    "type": "matmul",
                    "einsum_equation": "MK,KN->MN",
                    "elementwise_op": "mul",
                    "reduction_op": "add",
                    "is_real_einsum": True,
                    "is_einsum_supportable": True,
                    "tensor_names": {"inputs": ["a", "b"], "outputs": ["output"]},
                    "tensor_types": {
                        "inputs": ["input", "input"],
                        "outputs": ["output"],
                    },
                    "tensor_shapes": {
                        "inputs": [[n, n], [n, n]],
                        "outputs": [[n, n]],
                    },
                    "tensor_dtypes": {
                        "inputs": [dtype, dtype],
                        "outputs": [dtype],
                    },
                    "connections": {"inputs": ["start_a", "start_b"], "outputs": []},
                },
            }
        },
        strict=True,
    )


def _create_attestation(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    reference = tmp_path / "reference.py"
    graph_path = tmp_path / "einsum_graph.yaml"
    verification = tmp_path / "verification.yaml"
    _write_reference(reference)
    graph_path.write_text(yaml.safe_dump(_graph(), sort_keys=False))
    artifact = create_verification_artifact(
        reference_path=reference,
        reference_entry_point="run",
        input_factory_name="get_inputs",
        graph_path=graph_path,
        workload_name="tiny",
        workload_parameters={"n": 3},
        output_path=verification,
        atol=1e-5,
        rtol=1e-5,
    )
    return reference, graph_path, verification, artifact


def test_in_toto_artifact_is_replayable_and_hash_bound(tmp_path: Path):
    reference, graph_path, _, artifact = _create_attestation(tmp_path)
    assert artifact["_type"] == "https://in-toto.io/Statement/v1"
    assert len(artifact["predicate"]["cases"]) == 9
    replay_verification_artifact(
        artifact,
        reference_path=reference,
        graph_path=graph_path,
        workload_name="tiny",
        workload_parameters={"n": 3},
        atol=1e-5,
        rtol=1e-5,
    )


def test_replay_rejects_reference_or_graph_tampering(tmp_path: Path):
    reference, graph_path, _, artifact = _create_attestation(tmp_path)
    reference.write_text(reference.read_text().replace("torch.mm", "torch.add"))
    with pytest.raises(VerificationError, match="reference SHA-256"):
        replay_verification_artifact(
            artifact,
            reference_path=reference,
            graph_path=graph_path,
            workload_name="tiny",
            workload_parameters={"n": 3},
            atol=1e-5,
            rtol=1e-5,
        )


def test_schema_v3_benchmark_requires_and_replays_verification(tmp_path: Path):
    reference, graph_path, verification, _ = _create_attestation(tmp_path)
    graph_digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    analysis_path = tmp_path / "analysis.yaml"
    analysis = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path / "derived",
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
    )
    assert analysis is not None
    analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    benchmark = {
        "schema_version": 3,
        "name": "trusted",
        "precision": "fp32",
        "reference": {
            "source": reference.name,
            "entry_point": "run",
            "input_factory": "get_inputs",
        },
        "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "cache_policy": "cold",
        "workloads": [
            {
                "name": "tiny",
                "status": "compatible",
                "parameters": {"n": 3},
                "analysis": {
                    "path": analysis_path.name,
                    "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                    "source_graph": graph_path.name,
                    "source_graph_sha256": graph_digest,
                },
                "verification": {
                    "path": verification.name,
                    "sha256": hashlib.sha256(verification.read_bytes()).hexdigest(),
                },
            }
        ],
    }
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(yaml.safe_dump(benchmark, sort_keys=False))
    loaded = BenchmarkSpec.load(benchmark_path)
    assert loaded.workloads[0].verification is not None
    benchmark["workloads"][0].pop("verification")
    benchmark_path.write_text(yaml.safe_dump(benchmark, sort_keys=False))
    with pytest.raises(ValueError, match="require.*replayable verification"):
        BenchmarkSpec.load(benchmark_path)


def test_strict_conversion_and_analysis_fail_closed(tmp_path: Path):
    incomplete = {
        "layers": {
            "unknown": {
                "type": "unknown",
                "einsum_equation": "",
                "is_einsum_supportable": False,
                "tensor_shapes": {"inputs": [[2]], "outputs": [[2]]},
            }
        }
    }
    with pytest.raises(ConversionError, match="unsupported operation"):
        PyTorchToEinsum._validate_exact_graph(incomplete)
    path = tmp_path / "einsum_graph.yaml"
    path.write_text(yaml.safe_dump(incomplete))
    with pytest.raises(
        ValueError, match="strict analysis requires executable semantics"
    ):
        EinsumGraphAnalyzer().analyze_graph(path, tmp_path / "out", strict=True)
