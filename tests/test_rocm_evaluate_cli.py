# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from solar.cli import evaluate_rocm


def test_untrusted_no_lock_runs_without_requesting_clock_lock(monkeypatch, tmp_path):
    source_root = tmp_path / "package"
    source_root.mkdir()
    monkeypatch.setattr(
        evaluate_rocm.BenchmarkSpec,
        "load",
        lambda _path: SimpleNamespace(source_root=source_root),
    )
    monkeypatch.setattr(
        evaluate_rocm.SolutionSpec,
        "load",
        lambda _path: SimpleNamespace(source_root=source_root),
    )

    def unexpected_clock_lock():
        raise AssertionError("explicit --no-lock-clocks must not acquire a clock lock")

    command: list[str] = []

    def run(command_arg, check):
        command.extend(command_arg)
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluate_rocm, "acquire_clock_lock", unexpected_clock_lock)
    monkeypatch.setattr(evaluate_rocm.subprocess, "run", run)
    args = argparse.Namespace(
        benchmark="benchmark.yaml",
        solution="solution.yaml",
        baseline=None,
        output=str(tmp_path / "evaluation.yaml"),
        timing_profile="quick",
        no_lock_clocks=True,
        image="solar-rocm:7.2",
        arch_config="RX_9060_XT",
    )

    assert evaluate_rocm._run_container(args) == 0
    assert command
    assert "--no-lock-clocks" in command
    assert command[command.index("--arch-config") + 1] == "RX_9060_XT"
    assert f"{Path(source_root)}:/benchmark:ro" in command


def test_report_exit_code_rejects_workload_level_failures():
    for status in ("incorrect", "reward_hack", "runtime_error", "unstable_timing"):
        report = SimpleNamespace(
            failure=None, workloads=[SimpleNamespace(status=status)]
        )
        assert evaluate_rocm._report_exit_code(report) == 1


def test_report_exit_code_accepts_successful_diagnostics():
    report = SimpleNamespace(
        failure=None, workloads=[SimpleNamespace(status="diagnostic")]
    )
    assert evaluate_rocm._report_exit_code(report) == 0
