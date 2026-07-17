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

    command = None

    def run(command_arg, check):
        nonlocal command
        command = command_arg
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
    )

    assert evaluate_rocm._run_container(args) == 0
    assert command is not None
    assert "--no-lock-clocks" in command
    assert f"{Path(source_root)}:/benchmark:ro" in command
