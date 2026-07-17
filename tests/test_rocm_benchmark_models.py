# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from solar.benchmark.models import BaselineRegistry, BenchmarkSpec, SolutionSpec
from solar.benchmark import backends
from solar.benchmark.backends import BackendUnavailable, get_backend
from solar.rocm.environment import Capability, RocmEnvironment
from solar.benchmark.staging import (
    RewardHackDetected,
    SourceIntegrityError,
    stage_solution,
)


def _write_specs(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference = tmp_path / "reference.py"
    reference.write_text(
        "def get_inputs(workload, device): return []\ndef run(): return 1\n"
    )
    source = tmp_path / "solution.py"
    source.write_text("def run(): return 1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source_graph = tmp_path / "einsum_graph.yaml"
    source_graph.write_text("layers: {}\n")
    graph_digest = hashlib.sha256(source_graph.read_bytes()).hexdigest()
    analysis = tmp_path / "analysis.yaml"
    analysis.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "layers": {},
                "total": {
                    "flops": 0,
                    "fused_bytes": 0,
                    "macs_by_precision": {},
                },
                "metadata": {
                    "source_graph_sha256": graph_digest,
                    "dtype_accounting": "per_tensor",
                    "precision": "fp16",
                },
            },
            sort_keys=False,
        )
    )
    analysis_digest = hashlib.sha256(analysis.read_bytes()).hexdigest()
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "one",
                "precision": "fp16",
                "reference": {
                    "source": "reference.py",
                    "entry_point": "run",
                    "input_factory": "get_inputs",
                },
                "workloads": [
                    {
                        "name": "tiny",
                        "analysis": {
                            "path": "analysis.yaml",
                            "sha256": analysis_digest,
                            "source_graph": "einsum_graph.yaml",
                            "source_graph_sha256": graph_digest,
                        },
                    }
                ],
                "cache_policy": "cold",
            }
        )
    )
    solution = tmp_path / "solution.yaml"
    solution.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "candidate",
                "backend": "pytorch",
                "gfx_targets": ["gfx1200"],
                "sources": [{"path": "solution.py", "sha256": digest}],
                "entry_point": "solution.py::run",
            }
        )
    )
    return benchmark, solution, source


def test_yaml_contracts_and_hash_verified_staging(tmp_path: Path):
    benchmark_path, solution_path, source = _write_specs(tmp_path)
    benchmark = BenchmarkSpec.load(benchmark_path)
    solution = SolutionSpec.load(solution_path)
    assert benchmark.workloads[0].name == "tiny"
    with stage_solution(solution) as staged:
        assert (staged.root / "solution.py").read_text() == source.read_text()


def test_benchmark_hash_covers_reference_source(tmp_path: Path):
    benchmark_path, _, _ = _write_specs(tmp_path)
    before = BenchmarkSpec.load(benchmark_path).raw_hash
    (tmp_path / "reference.py").write_text(
        "def get_inputs(workload, device): return []\ndef run(): return 2\n"
    )
    assert BenchmarkSpec.load(benchmark_path).raw_hash != before


def test_benchmark_rejects_manual_sol_totals(tmp_path: Path):
    benchmark_path, _, _ = _write_specs(tmp_path)
    data = yaml.safe_load(benchmark_path.read_text())
    data["workloads"][0] = {"name": "tiny", "flops": 2, "fused_bytes": 4}
    benchmark_path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="manual workload"):
        BenchmarkSpec.load(benchmark_path)


def test_benchmark_rejects_tampered_analysis_artifact(tmp_path: Path):
    benchmark_path, _, _ = _write_specs(tmp_path)
    (tmp_path / "analysis.yaml").write_text("schema_version: 2\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        BenchmarkSpec.load(benchmark_path)


def test_benchmark_rejects_analysis_drift_even_with_updated_hash(tmp_path: Path):
    benchmark_path, _, _ = _write_specs(tmp_path)
    analysis_path = tmp_path / "analysis.yaml"
    analysis = yaml.safe_load(analysis_path.read_text())
    analysis["total"]["fused_bytes"] = 1
    analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    benchmark = yaml.safe_load(benchmark_path.read_text())
    benchmark["workloads"][0]["analysis"]["sha256"] = hashlib.sha256(
        analysis_path.read_bytes()
    ).hexdigest()
    benchmark_path.write_text(yaml.safe_dump(benchmark))

    with pytest.raises(ValueError, match="drifted"):
        BenchmarkSpec.load(benchmark_path)


def test_staging_rejects_hash_mismatch(tmp_path: Path):
    _, solution_path, source = _write_specs(tmp_path)
    solution = SolutionSpec.load(solution_path)
    source.write_text("def run(): return 2\n")
    with pytest.raises(SourceIntegrityError, match="SHA-256"):
        stage_solution(solution)


def test_static_scan_rejects_process_access(tmp_path: Path):
    _, solution_path, source = _write_specs(tmp_path)
    source.write_text("import subprocess\ndef run(): return 1\n")
    data = yaml.safe_load(solution_path.read_text())
    data["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    solution_path.write_text(yaml.safe_dump(data))
    with pytest.raises(RewardHackDetected, match="banned import"):
        stage_solution(SolutionSpec.load(solution_path))


def test_solution_paths_cannot_escape(tmp_path: Path):
    _, solution_path, _ = _write_specs(tmp_path)
    data = yaml.safe_load(solution_path.read_text())
    data["sources"][0]["path"] = "../outside.py"
    solution_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="relative path"):
        SolutionSpec.load(solution_path)


@pytest.mark.parametrize(
    "entry_point,match",
    [
        ("../reference.py::run", "relative path"),
        ("/benchmark/reference.py::run", "relative path"),
        ("reference.py::run", "listed in solution sources"),
        ("solution.py::not-valid", "Python identifier"),
    ],
)
def test_solution_entry_point_must_be_hash_verified(
    tmp_path: Path, entry_point: str, match: str
):
    _, solution_path, _ = _write_specs(tmp_path)
    data = yaml.safe_load(solution_path.read_text())
    data["entry_point"] = entry_point
    solution_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=match):
        SolutionSpec.load(solution_path)


def test_baseline_requires_exact_environment(tmp_path: Path):
    benchmark_path, _, _ = _write_specs(tmp_path)
    benchmark = BenchmarkSpec.load(benchmark_path)
    path = tmp_path / "baseline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "v1",
                "benchmark_hash": benchmark.raw_hash,
                "solution_hash": "a" * 64,
                "architecture_hash": "arch",
                "gfx_target": "gfx1200",
                "timing_profile": "official",
                "cache_policy": "cold",
                "environment_hash": "env",
                "clocks_locked": True,
                "workloads": {"tiny": 1.0},
            }
        )
    )
    baseline = BaselineRegistry.load(path)
    baseline.assert_compatible(benchmark, "env", "arch", "official", "gfx1200", True)
    with pytest.raises(ValueError, match="environment_hash"):
        baseline.assert_compatible(
            benchmark, "different", "arch", "official", "gfx1200", True
        )
    with pytest.raises(ValueError, match="architecture_hash"):
        baseline.assert_compatible(
            benchmark, "env", "different", "official", "gfx1200", True
        )


def test_native_backend_returns_structured_capability_failure():
    environment = RocmEnvironment(
        None,
        None,
        None,
        "RX 9060 XT",
        "gfx1200",
        16,
        32,
        16,
        {"rocwmma": Capability(False, "missing header")},
    )
    with pytest.raises(BackendUnavailable, match="missing header"):
        get_backend("rocwmma").assert_available(environment)


def test_native_backend_uses_detected_gfx_target_and_current_python(
    monkeypatch, tmp_path
):
    observed = None

    def run(command, **kwargs):
        nonlocal observed
        observed = command
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(backends.subprocess, "run", run)
    solution = type(
        "Solution",
        (),
        {"compile_command": ("{python}", "build.py", "{gfx_target}", "{staging}")},
    )()
    get_backend("hip_cpp")._compile(solution, tmp_path, "gfx1100")

    assert observed == [
        backends.sys.executable,
        "build.py",
        "gfx1100",
        str(tmp_path),
    ]


@pytest.mark.parametrize(
    "manifest", ["solution.yaml", "triton_solution.yaml", "hip_solution.yaml"]
)
def test_rocm_example_solution_hashes_and_staging(manifest):
    path = Path(__file__).parents[1] / "examples" / "rocm_matmul" / manifest
    solution = SolutionSpec.load(path)
    with stage_solution(solution) as staged:
        assert all((staged.root / source.path).is_file() for source in solution.sources)
