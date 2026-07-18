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

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
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
from solar.benchmark.official_corpus import OfficialCorpusManifest, verify_formal_entry
from solar.benchmark.timing import TimingPolicy, TimingStatistics
from solar.cli.build_official_corpus import problem_build_commands
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


def test_degenerate_reduction_is_explicitly_exempt() -> None:
    layer = _aten_layer(
        "amax",
        [[4, 1]],
        [[4]],
        ["torch.float32"],
        ["torch.float32"],
        kwargs={"dim": {"value": 1}},
    )
    result = classify_layer_resources(
        layer, macs=0, fallback_precision="fp32", strict=True
    )
    assert result["classification"] == "exempt"
    assert result["exemption_reason"] == "degenerate_single_element_reduction"


def _dequantized_matmul_graph(dtype: str) -> dict[str, Any]:
    effects = {
        "mutates": [],
        "aliases": [],
        "atomic": False,
        "opaque_library_call": False,
    }
    layers: dict[str, Any] = {}
    for name, shape in (("a", [8, 16]), ("b", [16, 8])):
        layers[f"start_{name}"] = {
            "type": "start",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {"inputs": [], "outputs": [name]},
            "tensor_shapes": {"inputs": [], "outputs": [shape]},
            "tensor_dtypes": {"inputs": [], "outputs": [dtype]},
            "tensor_types": {"inputs": [], "outputs": ["input"]},
            "connections": {"inputs": [], "outputs": [f"dequant_{name}"]},
            "semantic_op": {
                "kind": "input",
                "target": "input",
                "arguments": [],
                "kwargs": {},
            },
        }
        layers[f"dequant_{name}"] = {
            "type": "to",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {"inputs": [name], "outputs": [f"{name}_fp32"]},
            "tensor_shapes": {"inputs": [shape], "outputs": [shape]},
            "tensor_dtypes": {
                "inputs": [dtype],
                "outputs": ["torch.float32"],
            },
            "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
            "connections": {"inputs": [f"start_{name}"], "outputs": ["matmul"]},
            "semantic_op": {
                "kind": "aten",
                "target": "to",
                "arguments": [{"tensor": 0}],
                "kwargs": {"dtype": {"dtype": "torch.float32"}},
                "effects": effects,
            },
        }
    layers["matmul"] = {
        "type": "matmul",
        "einsum_equation": "MK,KN->MN",
        "is_real_einsum": True,
        "is_einsum_supportable": True,
        "tensor_names": {
            "inputs": ["a_fp32", "b_fp32"],
            "outputs": ["output"],
        },
        "tensor_shapes": {
            "inputs": [[8, 16], [16, 8]],
            "outputs": [[8, 8]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float32", "torch.float32"],
            "outputs": ["torch.float32"],
        },
        "tensor_types": {
            "inputs": ["input", "input"],
            "outputs": ["output"],
        },
        "connections": {"inputs": ["dequant_a", "dequant_b"], "outputs": []},
        "semantic_op": {
            "kind": "einsum",
            "target": "einsum",
            "equation": "MK,KN->MN",
            "arguments": [{"tensor": 0}, {"tensor": 1}],
            "kwargs": {},
            "effects": effects,
        },
    }
    return {"schema_version": 3, "layers": layers, "outputs": ["output"]}


def test_block_scaled_payload_uses_native_ocp_fp8_mfma(tmp_path: Path) -> None:
    graph_path = tmp_path / "einsum_graph.yaml"
    graph_path.write_text(
        yaml.safe_dump(_dequantized_matmul_graph("torch.float8_e4m3fn"))
    )
    result = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path,
        strict=True,
        architecture="RX_9060_XT",
        copy_graph=False,
    )
    assert result is not None
    assert result["total"]["macs_by_precision"] == {"fp8": 1024}
    assert result["total"]["resource_work"]["mfma"] == {"fp8->fp32": 2048}


def test_formal_gfx1200_analysis_rejects_fnuz_tensor(tmp_path: Path) -> None:
    graph_path = tmp_path / "einsum_graph.yaml"
    graph_path.write_text(
        yaml.safe_dump(_dequantized_matmul_graph("torch.float8_e4m3fnuz"))
    )
    with pytest.raises(ValueError, match="architecture-incompatible tensor dtype"):
        EinsumGraphAnalyzer().analyze_graph(
            graph_path,
            tmp_path,
            strict=True,
            architecture="RX_9060_XT",
            copy_graph=False,
        )


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
    probe_modes = [
        ("mfma_fp32", "mfma", "fp32->fp32"),
        ("mfma_fp16", "mfma", "fp16->fp32"),
        ("mfma_bf16", "mfma", "bf16->fp32"),
        ("mfma_fp8", "mfma", "fp8->fp32"),
        ("mfma_int8", "mfma", "int8->int32"),
        ("valu", "valu", "fp32"),
        ("sfu", "sfu", "fp32"),
        ("reduction", "reduction", "fp32"),
        ("atomic", "atomic", "fp32"),
        ("scan_sort", "scan_sort", "fp32"),
        ("conversion", "conversion", "fp32->fp16"),
        ("memory", "memory", "hbm"),
    ]
    probes = {
        name: ResourceProbe(lambda: None, 1.0, resource, mode)
        for name, resource, mode in probe_modes
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
    assert "mfma:fp8->fp32" in artifact.required_resource_modes
    assert "mfma:int8->int32" in artifact.required_resource_modes
    assert artifact.exempt_resource_modes == {
        "mfma:int4->int32": (
            "RDNA4 ISA defines IU4 WMMA, but ROCm 7.2 rocWMMA lists no INT4 "
            "type and PyTorch 2.11 exposes no validated gfx1200 INT4 matrix "
            "API; published peak remains source-only"
        )
    }
    assert artifact.precision_support["fp8"]["hardware"] == ("native_wmma_input_only")
    assert artifact.precision_support["int4"]["calibration"] == "exempt"
    assert len(artifact.calibrator_source_sha256) == 64


def test_publishable_calibration_rejects_missing_native_precision_mode() -> None:
    profile = ArchitectureProfile.load(_ROOT / "configs/arch/RX_9060_XT.yaml")
    probes = {
        resource: ResourceProbe(lambda: None, 1.0, resource, "generic")
        for resource in profile.resource_limits
    }
    probes["memory"] = ResourceProbe(lambda: None, 1.0, "memory", "hbm")
    with pytest.raises(ValueError, match="mfma:fp8->fp32"):
        RocmCalibrator(profile, _fake_environment()).calibrate(
            probes,
            timing_profile="official",
            clocks_locked=True,
            clock_levels=("AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK",),
        )


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
    assert profile.audit_evidence["schema_version"] == 3
    assert evidence["precision_support"] == profile.precision_support
    assert set(evidence["required_resource_modes"]).issubset(
        set(evidence["measured_resource_modes"])
    )
    assert set(evidence["required_resource_modes"]) == {
        "mfma:fp32->fp32",
        "mfma:fp16->fp32",
        "mfma:bf16->fp32",
        "mfma:fp8->fp32",
        "mfma:int8->int32",
    }
    assert "mfma:int4->int32" in evidence["exempt_resource_modes"]
    assert (
        evidence["calibrator_source_sha256"]
        == hashlib.sha256(
            (_ROOT / "solar/benchmark/calibration.py").read_bytes()
        ).hexdigest()
    )
    assert (
        evidence["probe_source_sha256"]
        == hashlib.sha256(
            (_ROOT / "solar/cli/calibrate_rocm.py").read_bytes()
        ).hexdigest()
    )
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
        manifest.architecture_profile_path
        == (_ROOT / "configs/arch/RX_9060_XT.yaml").resolve()
    )
    assert len(manifest.architecture_hash) == 64
    assert (
        report["source"]["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert report["gate"] == {
        "terminal_evidence_complete": True,
        "all_compatible_formally_attested": True,
        "formal_coverage_complete": True,
        "passed": True,
    }
    assert report["source"]["architecture_profile_sha256"] == (
        manifest.architecture_profile_sha256
    )
    assert report["source"]["architecture_hash"] == manifest.architecture_hash
    assert report["coverage"]["formal_attested_count"] == 9
    assert report["coverage"]["formal_attested"]["dtype"]["fp8"] == 1
    assert report["coverage"]["formal_requirements_met"] is True
    assert all(
        not missing
        for missing in report["coverage"]["missing_formal_requirements"].values()
    )
    incompatible = [
        result
        for result in report["results"].values()
        if result["compatibility"]["status"] == "incompatible"
    ]
    assert len(incompatible) == 1
    assert all(
        result["compatibility"]["reason_code"] == "unsupported_quantization_format"
        for result in incompatible
    )
    assert all(
        result["compatibility"]["fallbacks_used"] == [] for result in incompatible
    )


def test_official_corpus_batch_build_is_deterministic_and_profile_bound() -> None:
    manifest = OfficialCorpusManifest.load(
        _ROOT / "configs/corpus/RX_9060_XT_SOL_EXECBENCH.yaml"
    )
    commands = problem_build_commands(
        manifest,
        Path("/materialized"),
        Path("/artifacts"),
        device="cuda:0",
        orojenesis_home="/opt/orojenesis",
        blob_roots=("/blobs",),
        python_executable="python",
    )

    assert len(commands) == 9
    assert (
        len(
            [
                command
                for command in commands
                if "L1/040_conv2d_residual_block" in " ".join(command)
            ]
        )
        == 1
    )
    assert all(
        command[command.index("--arch-config") + 1]
        == str(manifest.architecture_profile_path)
        for command in commands
    )
    assert all("--workload" not in command for command in commands)
    assert all(command[-2:] == ["--blob-root", "/blobs"] for command in commands)


def test_official_corpus_rejects_artifact_from_other_profile(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = OfficialCorpusManifest.load(
        _ROOT / "configs/corpus/RX_9060_XT_SOL_EXECBENCH.yaml"
    )
    entry = manifest.entries[0]
    benchmark_path = tmp_path / entry.config / entry.problem / "benchmark.yaml"
    benchmark_path.parent.mkdir(parents=True)
    benchmark_path.write_text("schema_version: 3\n")
    analysis = SimpleNamespace(
        architecture_hash="0" * 64,
        sha256="1" * 64,
        flops=entry.golden_flops,
        fused_bytes=entry.golden_external_bytes,
        resource_work=entry.golden_resource_work,
    )
    verification = SimpleNamespace(sha256="2" * 64)
    benchmark = SimpleNamespace(
        workloads=[
            SimpleNamespace(
                uuid=entry.workload_uuid,
                analysis=analysis,
                verification=verification,
            )
        ]
    )
    monkeypatch.setattr(
        "solar.benchmark.official_corpus.BenchmarkSpec.load", lambda _path: benchmark
    )

    result = verify_formal_entry(
        entry,
        tmp_path,
        expected_architecture_hash=manifest.architecture_hash,
    )

    assert result["formal_attested"] is False
    assert result["formal_reason"] == "architecture_profile_mismatch"


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
