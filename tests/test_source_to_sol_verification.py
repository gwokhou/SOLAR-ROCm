# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from solar.analysis.graph_analyzer import (
    EinsumGraphAnalyzer,
    contraction_operands_are_graph_external,
)
from solar.analysis.orojenesis import OROJENESIS_COMMIT, OrojenesisRunner
from solar.benchmark.models import BenchmarkSpec, canonical_hash
from solar.einsum import ConversionError, PyTorchToEinsum, annotate_semantics
from solar.rocm import ArchitectureProfile
from solar.verification import (
    VerificationError,
    create_verification_artifact,
    replay_verification_artifact,
)
from solar.verification.einsum import _assert_close


class _DeterministicOrojenesisRunner:
    """Small evidence-producing stand-in for the pinned external binary."""

    def __init__(self, minimum_accesses: int = 80):
        self.minimum_accesses = minimum_accesses

    def run_layer(self, layer, output_dir, *, word_bits):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        problem = OrojenesisRunner.problem_for_layer(layer)
        dimensions = list(problem["problem"]["shape"]["dimensions"])
        spaces = [item["name"] for item in problem["problem"]["shape"]["data-spaces"]]
        documents = {
            "problem.yaml": problem,
            "architecture.yaml": OrojenesisRunner.architecture(word_bits),
            "mapper.yaml": OrojenesisRunner.mapper_config(dimensions, spaces),
        }
        for name, value in documents.items():
            (output_dir / name).write_text(yaml.safe_dump(value, sort_keys=False))
        curve_path = output_dir / "timeloop-mapper.oaves.csv"
        curve_path.write_text(
            f"64,2,{self.minimum_accesses + 20},map,None,10\n"
            f"128,4,{self.minimum_accesses},map,None,10\n"
        )
        evidence_files = {
            name: {
                "path": name,
                "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
            }
            for name in documents
        }
        evidence_files["curve"] = {
            "path": curve_path.name,
            "sha256": hashlib.sha256(curve_path.read_bytes()).hexdigest(),
        }
        return {
            "solver": "NVlabs/timeloop oaves_keep_max",
            "commit": OROJENESIS_COMMIT,
            "word_bits": word_bits,
            "curve": OrojenesisRunner.parse_curve(
                curve_path, word_bytes=word_bits // 8
            ),
            "evidence_files": evidence_files,
        }


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


def test_attestation_uses_official_nonfinite_and_ratio_policy() -> None:
    import torch

    negative_inf = torch.tensor([float("-inf"), 1.0])
    with pytest.raises(VerificationError, match="non-finite"):
        _assert_close(negative_inf, negative_inf, 0.0, 0.0)
    _assert_close(
        negative_inf,
        negative_inf,
        0.0,
        0.0,
        allow_negative_inf=True,
    )
    actual = torch.ones(100)
    expected = actual.clone()
    actual[-1] = 10.0
    _assert_close(
        actual,
        expected,
        0.0,
        0.0,
        required_matched_ratio=0.99,
    )
    with pytest.raises(VerificationError, match="exceeds cap"):
        _assert_close(
            actual,
            expected,
            0.0,
            0.0,
            required_matched_ratio=0.99,
            max_error_cap=5.0,
        )


def test_schema_v3_benchmark_requires_and_replays_verification(tmp_path: Path):
    reference, graph_path, verification, _ = _create_attestation(tmp_path)
    graph_digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    analysis_path = tmp_path / "derived" / "analysis.yaml"
    analysis = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path / "derived",
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(),
        require_orojenesis=True,
    )
    assert analysis is not None
    assert analysis["total"]["io_lower_bound_bytes"] > analysis["total"]["fused_bytes"]
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
                    "path": str(analysis_path.relative_to(tmp_path)),
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
    architecture_identity = ArchitectureProfile.load("RX_9060_XT").to_dict()
    architecture_identity.pop("source", None)
    assert loaded.workloads[0].analysis is not None
    assert loaded.workloads[0].analysis.architecture_hash == canonical_hash(
        architecture_identity
    )
    curve_path = (
        analysis_path.parent / "orojenesis" / "matmul" / ("timeloop-mapper.oaves.csv")
    )
    original_curve = curve_path.read_text()
    curve_path.write_text("128,4,1,map,None,10\n")
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        BenchmarkSpec.load(benchmark_path)
    curve_path.write_text(original_curve)
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


def test_orojenesis_curve_controls_formal_io_and_time_bound(tmp_path: Path) -> None:
    graph_path = tmp_path / "einsum_graph.yaml"
    graph_path.write_text(yaml.safe_dump(_graph(32), sort_keys=False))
    low = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path / "low",
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(5_000),
        require_orojenesis=True,
    )
    high = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path / "high",
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(20_000),
        require_orojenesis=True,
    )
    assert low is not None and high is not None
    assert high["total"]["io_lower_bound_bytes"] > low["total"]["io_lower_bound_bytes"]
    assert high["total"]["lower_bound_seconds"] > low["total"]["lower_bound_seconds"]
    assert high["metadata"]["orojenesis"]["formal_coverage"] == {
        "applicable_layers": 1,
        "total_layers": 1,
    }


def test_solver_applicability_traces_only_unconditional_aliases() -> None:
    graph = _graph()
    matmul = graph["layers"]["matmul"]
    transpose = {
        "type": "transpose",
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "tensor_names": {"inputs": ["b"], "outputs": ["bt"]},
        "tensor_shapes": {"inputs": [[3, 3]], "outputs": [[3, 3]]},
        "tensor_dtypes": {
            "inputs": ["torch.float32"],
            "outputs": ["torch.float32"],
        },
        "module_args": {
            "call_arguments": [{"tensor": 0}, {"value": 0}, {"value": 1}],
            "call_kwargs": {},
        },
        "connections": {"inputs": ["start_b"], "outputs": ["matmul"]},
    }
    graph["layers"]["transpose"] = transpose
    matmul["tensor_names"]["inputs"][1] = "bt"
    graph = annotate_semantics(graph, strict=True)
    assert contraction_operands_are_graph_external(matmul, graph["layers"])

    graph["layers"]["transpose"]["semantic_op"]["effects"]["aliases"][0][
        "conditional"
    ] = True
    assert not contraction_operands_are_graph_external(matmul, graph["layers"])
