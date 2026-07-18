# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the SOL-ExecBench P1 closure."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

from solar.analysis.resources import (
    RESOURCE_MODEL_VERSION,
    ResourceClassificationError,
    classify_layer_resources,
)
from solar.benchmark.calibration import ResourceProbe, RocmCalibrator
from solar.benchmark.evaluator import RocmEvaluator
from solar.benchmark.models import (
    CompatibilityArtifact,
    WorkloadSpec,
    canonical_hash,
)
from solar.benchmark.official_corpus import OfficialCorpusManifest
from solar.benchmark.timing import TimingPolicy, TimingStatistics
from solar.cli.build_source_to_sol import _extract_graph
from solar.rocm import ArchitectureProfile
from solar.rocm.environment import Capability, RocmEnvironment
from solar.verification.einsum import EinsumGraphExecutor

_ROOT = Path(__file__).resolve().parents[1]


def _aten_layer(
    target: str,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
    input_dtypes: list[str],
    output_dtypes: list[str],
    *,
    arguments: list[object] | None = None,
    kwargs: dict[str, object] | None = None,
    effects: dict[str, object] | None = None,
) -> dict[str, object]:
    layer: dict[str, object] = {
        "type": target,
        "tensor_names": {
            "inputs": [f"input_{index}" for index in range(len(input_shapes))],
            "outputs": [f"output_{index}" for index in range(len(output_shapes))],
        },
        "tensor_shapes": {"inputs": input_shapes, "outputs": output_shapes},
        "tensor_dtypes": {"inputs": input_dtypes, "outputs": output_dtypes},
        "semantic_op": {
            "kind": "aten",
            "target": target,
            "arguments": (
                arguments
                if arguments is not None
                else [{"tensor": index} for index in range(len(input_shapes))]
            ),
            "kwargs": kwargs or {},
            "effects": effects
            or {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
    }
    return layer


def test_resource_model_counts_distinct_amd_pipelines() -> None:
    atomic = _aten_layer(
        "index_add",
        [[4, 8], [6], [6, 8]],
        [[4, 8]],
        ["torch.bfloat16", "torch.int64", "torch.bfloat16"],
        ["torch.bfloat16"],
        effects={
            "mutates": [0],
            "aliases": [],
            "atomic": True,
            "opaque_library_call": False,
        },
    )
    scan = _aten_layer(
        "cumsum",
        [[32]],
        [[32]],
        ["torch.float32"],
        ["torch.float32"],
        arguments=[{"tensor": 0}],
        kwargs={"dim": {"value": 0}},
    )
    reduction = _aten_layer(
        "sum",
        [[4, 8]],
        [[4]],
        ["torch.float32"],
        ["torch.float32"],
        arguments=[{"tensor": 0}],
        kwargs={"dim": {"value": 1}},
    )
    conversion = _aten_layer(
        "to",
        [[32]],
        [[32]],
        ["torch.float32"],
        ["torch.float16"],
    )

    assert classify_layer_resources(
        atomic, macs=0, fallback_precision="fp16", strict=True
    )["work"] == {"atomic": {"bf16": 48}}
    assert classify_layer_resources(
        scan, macs=0, fallback_precision="fp16", strict=True
    )["work"] == {"scan_sort": {"fp32": 32}}
    assert classify_layer_resources(
        reduction, macs=0, fallback_precision="fp16", strict=True
    )["work"] == {"reduction": {"fp32": 28}}
    assert classify_layer_resources(
        conversion, macs=0, fallback_precision="fp16", strict=True
    )["work"] == {"conversion": {"fp32->fp16": 32}}


def test_resource_model_rejects_unknown_compute() -> None:
    layer = _aten_layer(
        "unreviewed_op",
        [[8]],
        [[8]],
        ["torch.float32"],
        ["torch.float32"],
    )
    with pytest.raises(ResourceClassificationError, match="no amd_resource_v1 rule"):
        classify_layer_resources(layer, macs=0, fallback_precision="fp32", strict=True)


def _fake_environment() -> RocmEnvironment:
    return RocmEnvironment(
        rocm_version="7.2",
        torch_version="2.11",
        hip_version="7.2",
        device_name="AMD Radeon RX 9060 XT",
        gfx_target="gfx1200",
        pytorch_compute_units=16,
        normalized_compute_units=32,
        total_memory_bytes=17_095_983_104,
        capabilities={"pytorch_rocm": Capability(True, "test")},
    )


def _fake_statistics() -> TimingStatistics:
    return TimingStatistics(
        samples_ms=(1.0,),
        p20_ms=1.0,
        p50_ms=1.0,
        p80_ms=1.0,
        p95_ms=1.0,
        iqr_ms=0.0,
        mean_ms=1.0,
        std_ms=0.0,
        stable=True,
    )


def test_official_calibration_is_complete_and_not_diagnostic(monkeypatch) -> None:
    class FakeTimer:
        def __init__(self, _policy):
            pass

        def measure(self, _operation):
            return _fake_statistics()

    monkeypatch.setattr("solar.benchmark.calibration.AdaptiveTimer", FakeTimer)
    profile = ArchitectureProfile.load(_ROOT / "configs/arch/RX_9060_XT.yaml")
    probes = {
        resource: ResourceProbe(lambda: None, 1.0, resource, mode)
        for resource, mode in {
            "mfma": "fp32->fp32",
            "valu": "fp32",
            "sfu": "fp32",
            "reduction": "fp32",
            "atomic": "fp32",
            "scan_sort": "fp32",
            "conversion": "fp32->fp16",
            "memory": "hbm",
        }.items()
    }
    artifact = RocmCalibrator(profile, _fake_environment()).calibrate(
        probes,
        timing_profile="official",
        clocks_locked=True,
        clock_levels=("AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK",),
        probe_source_sha256="a" * 64,
    )

    assert artifact.resource_model_version == RESOURCE_MODEL_VERSION
    assert artifact.audit_status == "verified"
    assert artifact.diagnostic_only is False
    assert set(artifact.measured_throughput_per_second) == set(probes)
    assert len(artifact.calibrator_source_sha256) == 64


def test_publishable_calibration_requires_locked_clocks() -> None:
    with pytest.raises(RuntimeError, match="STABLE_PEAK"):
        RocmCalibrator(
            ArchitectureProfile.load(_ROOT / "configs/arch/RX_9060_XT.yaml"),
            _fake_environment(),
        ).calibrate({}, timing_profile="official", clocks_locked=False)


def test_profile_binds_verified_local_resource_evidence() -> None:
    profile = ArchitectureProfile.load(_ROOT / "configs/arch/RX_9060_XT.yaml")
    evidence_path = _ROOT / str(profile.audit_evidence["path"])
    evidence = yaml.safe_load(evidence_path.read_text())

    assert (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        == profile.audit_evidence["sha256"]
    )
    assert evidence["audit_status"] == "verified"
    assert evidence["diagnostic_only"] is False
    assert evidence["clocks_locked"] is True
    assert evidence["gfx_target"] == profile.gfx_target
    assert evidence["resource_model_version"] == RESOURCE_MODEL_VERSION
    assert set(evidence["upper_bound_per_second"]) == set(
        evidence["measured_throughput_per_second"]
    )
    assert all(value <= 1.05 for value in evidence["upper_bound_ratio"].values())


def test_pinned_official_corpus_gate_preserves_incompatibilities() -> None:
    manifest_path = _ROOT / "configs/corpus/RX_9060_XT_SOL_EXECBENCH.yaml"
    manifest = OfficialCorpusManifest.load(manifest_path)
    report = yaml.safe_load(
        (
            _ROOT / "configs/corpus/evidence/RX_9060_XT_SOL_EXECBENCH_audit.yaml"
        ).read_text()
    )

    assert len(manifest.entries) == 10
    assert (
        report["source"]["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert report["gate"] == {
        "terminal_evidence_complete": True,
        "all_compatible_formally_attested": True,
        "passed": True,
    }
    assert report["coverage"]["formal_attested_count"] == 8
    incompatible = [
        result
        for result in report["results"].values()
        if result["compatibility"]["status"] == "incompatible"
    ]
    assert len(incompatible) == 2
    assert all(
        result["compatibility"]["reason_code"] == "unsupported_quantization_format"
        for result in incompatible
    )
    assert all(
        result["compatibility"]["fallbacks_used"] == [] for result in incompatible
    )


def test_compatibility_artifact_rejects_any_fallback(tmp_path: Path) -> None:
    path = tmp_path / "compatibility.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "status": "incompatible",
                "reason_code": "unsupported_quantization_format",
                "fallbacks_used": ["replace_fp8_with_fp16"],
            }
        )
    )
    with pytest.raises(ValueError, match="must not use fallbacks"):
        CompatibilityArtifact.load(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            tmp_path,
        )


def test_tensor_keyword_inputs_are_recovered_without_reordering_source(
    tmp_path: Path,
) -> None:
    import torch

    def reference(destination, source, index):
        output = destination.clone()
        output.index_add_(dim=0, index=index, source=source)
        return output

    inputs = (
        torch.zeros((4, 3), dtype=torch.float32),
        torch.arange(18, dtype=torch.float32).reshape(6, 3),
        torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64),
    )
    graph_path = _extract_graph(
        reference,
        inputs,
        device="cpu",
        output=tmp_path,
        name="keyword_index_add",
    )
    graph = yaml.safe_load(graph_path.read_text())
    assert graph["source_input_indices"] == [0, 2, 1]
    executor_inputs = tuple(inputs[index] for index in graph["source_input_indices"])
    assert torch.equal(EinsumGraphExecutor(graph)(*executor_inputs), reference(*inputs))


def test_publishable_measurement_below_tsol_is_bound_violation(monkeypatch) -> None:
    class FakeTimer:
        def __init__(self, _policy):
            pass

        def measure(self, fn, *, setup, validate, **_kwargs):
            args = setup()
            output = fn(*args)
            validate(args, output)
            return TimingStatistics(
                samples_ms=(0.5,),
                p20_ms=0.5,
                p50_ms=0.5,
                p80_ms=0.5,
                p95_ms=0.5,
                iqr_ms=0.0,
                mean_ms=0.5,
                std_ms=0.0,
                stable=True,
            )

    monkeypatch.setattr("solar.benchmark.evaluator.AdaptiveTimer", FakeTimer)
    evaluator = RocmEvaluator()
    architecture_identity = evaluator.architecture.to_dict()
    architecture_identity.pop("source", None)
    workload = WorkloadSpec(
        name="violates-bound",
        analysis=cast(
            Any,
            SimpleNamespace(
                architecture_hash=canonical_hash(architecture_identity),
                lower_bound_seconds=0.001,
                sha256="a" * 64,
                source_graph_sha256="b" * 64,
            ),
        ),
        verification=cast(Any, SimpleNamespace(sha256="c" * 64)),
    )
    integrity = SimpleNamespace(check=lambda: None)
    result = evaluator._evaluate_workload(  # pylint: disable=protected-access
        workload,
        cast(Any, SimpleNamespace()),
        lambda value: value,
        lambda value: value,
        lambda _parameters, _device: (1.0,),
        TimingPolicy.for_name("official"),
        None,
        None,
        None,
        {},
        True,
        cast(Any, integrity),
    )

    assert result.status == "bound_violation"
    assert result.correct is True
    assert result.sol_score is None
    assert result.bound_audit is not None
    assert result.bound_audit["observed_to_theoretical_ratio"] == 0.5
    assert result.bound_audit["analysis_sha256"] == "a" * 64
