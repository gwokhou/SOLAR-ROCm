# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solar.benchmark.sol_execbench import (
    AmdCompatibilityAuditor,
    SolExecBenchProblem,
    SolExecBenchFormatError,
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


def test_ocp_fp8_reference_is_native_on_gfx1200(tmp_path: Path) -> None:
    problem = _problem(
        tmp_path,
        "import torch\n"
        "def run(x):\n"
        "    return x.to(torch.float8_e4m3fn).float()\n",
    )
    auditor = AmdCompatibilityAuditor(problem)
    auditor.environment = _environment(1024)
    result = auditor.audit(problem.workloads[0], execute=False)

    assert result["status"] == "compatible"
    assert result["reason_code"] == "compatible"
    assert result["stage"] == "static_complete"
    assert result["fallbacks_used"] == []


def test_gfx94x_fnuz_reference_is_not_substituted_on_gfx1200(
    tmp_path: Path,
) -> None:
    problem = _problem(
        tmp_path,
        "import torch\n"
        "def run(x):\n"
        "    return x.to(torch.float8_e4m3fnuz).float()\n",
    )
    auditor = AmdCompatibilityAuditor(problem)
    auditor.environment = _environment(1024)
    result = auditor.audit(problem.workloads[0], execute=False)

    assert result["status"] == "incompatible"
    assert result["reason_code"] == "unsupported_quantization_format"
    assert result["evidence"]["policy"] == (
        "non-native quantization formats are not substituted on AMD"
    )
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


def _custom_problem(
    tmp_path: Path, *, wrong_dtype: bool = False
) -> SolExecBenchProblem:
    index_expression = "index.float()" if wrong_dtype else "index"
    reference = (
        "import torch\n"
        "def make_inputs(parameters, device):\n"
        "    n = parameters['N']\n"
        "    value = torch.arange(n * 2, dtype=torch.float32, device=device)\n"
        "    index = torch.arange(n, dtype=torch.int64, device=device) * 2\n"
        f"    return {{'value': value, 'index': {index_expression}}}\n"
        "def run(value, index):\n"
        "    return torch.gather(value, 0, index)\n"
    )
    definition = {
        "name": "dynamic_custom_gather",
        "axes": {
            "N": {"type": "var"},
            "TWO_N": {"type": "expr", "expression": "N * 2"},
        },
        "inputs": {
            "value": {"shape": ["TWO_N"], "dtype": "float32"},
            "index": {"shape": ["N"], "dtype": "int64"},
        },
        "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
        "custom_inputs_entrypoint": "make_inputs",
        "reference": reference,
    }
    workloads = [
        {
            "uuid": f"w{n}",
            "axes": {"N": n},
            "inputs": {
                "value": {"type": "custom"},
                "index": {"type": "custom"},
            },
        }
        for n in (2, 5)
    ]
    (tmp_path / "definition.json").write_text(json.dumps(definition))
    (tmp_path / "workload.jsonl").write_text(
        "".join(json.dumps(workload) + "\n" for workload in workloads)
    )
    return SolExecBenchProblem.load(tmp_path)


def test_dynamic_custom_workloads_round_trip_through_standalone_adapter(
    tmp_path: Path,
) -> None:
    problem = _custom_problem(tmp_path)
    namespace: dict[str, object] = {
        "__name__": "_standalone_test",
        "__file__": str(tmp_path / "reference.py"),
    }
    exec(standalone_reference_source(problem), namespace)  # pylint: disable=exec-used
    factory = namespace["get_inputs"]

    for workload, size in zip(problem.workloads, (2, 5)):
        direct = problem.generate_inputs(workload, "cpu", seed=17)
        standalone = factory({"uuid": workload.uuid, "seed": 17}, "cpu")
        assert [tuple(value.shape) for value in direct] == [(size * 2,), (size,)]
        assert [value.dtype for value in direct] == [
            standalone[0].dtype,
            standalone[1].dtype,
        ]
        assert all(left.equal(right) for left, right in zip(direct, standalone))


def test_custom_subset_and_implicit_random_inputs_match_official_contract(
    tmp_path: Path,
) -> None:
    definition = {
        "name": "custom_subset",
        "axes": {"N": {"type": "var"}},
        "inputs": {
            "value": {"shape": ["N"], "dtype": "float32"},
            "index": {"shape": ["N"], "dtype": "int64"},
        },
        "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
        "custom_inputs_entrypoint": "make_inputs",
        "reference": (
            "import torch\n"
            "def make_inputs(parameters, device):\n"
            "    return {'index': torch.arange(parameters['N'], device=device)}\n"
            "def run(value, index):\n"
            "    return value[index]\n"
        ),
    }
    workload = {
        "uuid": "custom-subset",
        "axes": {"N": 4},
        # Unlisted inputs are random in the pinned upstream schema; a custom
        # factory returns only the explicitly custom subset.
        "inputs": {"index": {"type": "custom"}},
    }
    (tmp_path / "definition.json").write_text(json.dumps(definition))
    (tmp_path / "workload.jsonl").write_text(json.dumps(workload) + "\n")
    problem = SolExecBenchProblem.load(tmp_path)

    namespace: dict[str, object] = {
        "__name__": "_mixed_standalone_test",
        "__file__": str(tmp_path / "reference.py"),
    }
    exec(standalone_reference_source(problem), namespace)  # pylint: disable=exec-used
    direct = problem.generate_inputs(problem.workloads[0], "cpu", seed=19)
    standalone = namespace["get_inputs"]({"uuid": "custom-subset", "seed": 19}, "cpu")

    assert all(left.equal(right) for left, right in zip(direct, standalone))


def test_unshaped_random_input_is_a_python_scalar(tmp_path: Path) -> None:
    definition = {
        "name": "random_scalar",
        "axes": {},
        "inputs": {"value": {"shape": None, "dtype": "float32"}},
        "outputs": {"output": {"shape": None, "dtype": "float32"}},
        "reference": "def run(value):\n    return value\n",
    }
    workload = {"uuid": "scalar", "axes": {}, "inputs": {}}
    (tmp_path / "definition.json").write_text(json.dumps(definition))
    (tmp_path / "workload.jsonl").write_text(json.dumps(workload) + "\n")
    problem = SolExecBenchProblem.load(tmp_path)
    namespace: dict[str, object] = {
        "__name__": "_scalar_standalone_test",
        "__file__": str(tmp_path / "reference.py"),
    }
    exec(standalone_reference_source(problem), namespace)  # pylint: disable=exec-used

    direct = problem.generate_inputs(problem.workloads[0], "cpu", seed=23)
    standalone = namespace["get_inputs"]({"uuid": "scalar", "seed": 23}, "cpu")
    assert isinstance(direct[0], float)
    assert direct == standalone


def test_custom_input_dtype_drift_is_rejected_by_both_adapters(tmp_path: Path) -> None:
    problem = _custom_problem(tmp_path, wrong_dtype=True)
    with pytest.raises(SolExecBenchFormatError, match="index dtype"):
        problem.generate_inputs(problem.workloads[0], "cpu")

    namespace: dict[str, object] = {
        "__name__": "_standalone_test",
        "__file__": str(tmp_path / "reference.py"),
    }
    exec(standalone_reference_source(problem), namespace)  # pylint: disable=exec-used
    with pytest.raises(ValueError, match="index dtype"):
        namespace["get_inputs"]({"uuid": "w2"}, "cpu")


def test_workload_axes_must_exactly_match_dynamic_definition(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    problem.workloads[0].raw["axes"]["UNDECLARED"] = 1
    with pytest.raises(SolExecBenchFormatError, match="exactly match"):
        problem.resolved_axes(problem.workloads[0])
