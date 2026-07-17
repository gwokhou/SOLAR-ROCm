# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from solar.benchmark.sol_execbench import (
    AmdCompatibilityAuditor,
    SolExecBenchProblem,
    standalone_reference_source,
)
from solar.rocm.environment import Capability, RocmEnvironment


def _problem(
    tmp_path: Path, reference: str = "import torch\ndef run(x):\n    return x + 1\n"
) -> SolExecBenchProblem:
    definition = {
        "name": "dynamic_add",
        "axes": {
            "N": {"type": "var"},
            "TWO_N": {"type": "expr", "expression": "N * 2"},
        },
        "inputs": {"x": {"shape": ["TWO_N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["TWO_N"], "dtype": "float32"}},
        "reference": reference,
    }
    workload = {"uuid": "w0", "axes": {"N": 4}, "inputs": {"x": {"type": "random"}}}
    (tmp_path / "definition.json").write_text(json.dumps(definition))
    (tmp_path / "workload.jsonl").write_text(json.dumps(workload) + "\n")
    return SolExecBenchProblem.load(tmp_path)


def _environment(total: int) -> RocmEnvironment:
    return RocmEnvironment(
        rocm_version="7.2",
        torch_version="2.11",
        hip_version="7.2",
        device_name="AMD",
        gfx_target="gfx1200",
        pytorch_compute_units=16,
        normalized_compute_units=32,
        total_memory_bytes=total,
        capabilities={"pytorch_rocm": Capability(True, "test")},
    )


def test_official_schema_resolves_expressions_and_standalone_factory(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    assert problem.resolved_axes(problem.workloads[0])["TWO_N"] == 8
    source = standalone_reference_source(problem)
    assert "import solar" not in source
    assert "import sol_execbench" not in source


def test_cycle_dependency_is_recorded_without_fallback(tmp_path: Path) -> None:
    problem = _problem(tmp_path, "import solar\ndef run(x):\n    return x\n")
    auditor = AmdCompatibilityAuditor(problem)
    auditor.environment = _environment(1024)
    result = auditor.audit(problem.workloads[0], execute=False)
    assert result["status"] == "incompatible"
    assert result["reason_code"] == "cyclic_reference_dependency"
    assert result["fallbacks_used"] == []


def test_static_capacity_rejection_is_evidence_backed(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    auditor = AmdCompatibilityAuditor(problem)
    auditor.environment = _environment(8)
    result = auditor.audit(problem.workloads[0], execute=False)
    assert result["status"] == "incompatible"
    assert result["reason_code"] == "insufficient_device_capacity"
    assert result["evidence"]["minimum_storage_bytes"] == 64


def test_missing_external_input_is_recorded_without_search_or_fallback(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    raw = problem.workloads[0].raw
    raw["inputs"]["x"] = {
        "type": "safetensors",
        "path": "data/not-present.safetensors",
        "tensor_key": "x",
    }
    auditor = AmdCompatibilityAuditor(problem)
    auditor.environment = _environment(1024)
    result = auditor.audit(problem.workloads[0], execute=False)
    assert result["status"] == "not_checked"
    assert result["reason_code"] == "missing_external_input"
    assert result["fallbacks_used"] == []


def test_external_input_resolves_only_from_explicit_blob_root(tmp_path: Path) -> None:
    problem_root = tmp_path / "problem"
    problem_root.mkdir()
    problem = _problem(problem_root)
    blob_root = tmp_path / "dataset"
    blob = blob_root / "data" / "input.safetensors"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"bound-by-hash")
    rebound = SolExecBenchProblem.load(problem_root, blob_roots=[blob_root])
    assert rebound.resolve_blob("data/input.safetensors") == blob.resolve()
