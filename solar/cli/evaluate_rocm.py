# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a SOLAR ROCm solution package."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from solar.benchmark import BenchmarkSpec, RocmEvaluator, SolutionSpec
from solar.benchmark.clock_lock import acquire_clock_lock
from solar.rocm import ArchitectureProfile

_CANDIDATE_FAILURE_STATUSES = frozenset(
    {"incorrect", "reward_hack", "runtime_error", "unstable_timing"}
)


def _report_exit_code(report: object) -> int:
    """Map structured evaluator outcomes to automation-safe exit codes."""
    if getattr(report, "failure", None):
        return 2
    statuses = {
        str(getattr(item, "status", "invalid"))
        for item in getattr(report, "workloads", ())
    }
    if statuses & _CANDIDATE_FAILURE_STATUSES:
        return 1
    if any(status not in {"passed", "diagnostic"} for status in statuses):
        return 2
    return 0


def _run_container(args: argparse.Namespace) -> int:
    benchmark = BenchmarkSpec.load(args.benchmark)
    solution = SolutionSpec.load(args.solution)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture_arg = args.arch_config
    architecture_path = Path(args.arch_config)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add",
        "video",
        "--group-add",
        "render",
        "--security-opt",
        "seccomp=unconfined",
        "-v",
        f"{benchmark.source_root}:/benchmark:ro",
        "-v",
        f"{solution.source_root}:/solution:ro",
        "-v",
        f"{output.parent}:/output",
    ]
    if architecture_path.is_file():
        architecture_path = architecture_path.resolve()
        command += ["-v", f"{architecture_path.parent}:/architecture:ro"]
        architecture_arg = f"/architecture/{architecture_path.name}"
    baseline_args: list[str] = []
    if args.baseline:
        baseline = Path(args.baseline).resolve()
        command += ["-v", f"{baseline.parent}:/baseline:ro"]
        baseline_args = ["--baseline", f"/baseline/{baseline.name}"]
    command += [
        args.image,
        "--benchmark",
        f"/benchmark/{Path(args.benchmark).name}",
        "--solution",
        f"/solution/{Path(args.solution).name}",
        "--output",
        f"/output/{output.name}",
        "--timing-profile",
        args.timing_profile,
        "--arch-config",
        architecture_arg,
        "--no-lock-clocks",
        *baseline_args,
    ]
    if args.no_lock_clocks:
        return subprocess.run(command, check=False).returncode

    with acquire_clock_lock() as lease:
        if not lease.locked:
            print("Unable to verify STABLE_PEAK; retry with --no-lock-clocks")
            return 2
        return subprocess.run(command, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a ROCm kernel against benchmark.yaml"
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", default="evaluation.yaml")
    parser.add_argument(
        "--arch-config",
        default="RX_9060_XT",
        help="AMD architecture profile name or YAML path",
    )
    parser.add_argument(
        "--timing-profile",
        choices=("standard", "official", "quick"),
        default="standard",
    )
    parser.add_argument("--no-lock-clocks", action="store_true")
    parser.add_argument(
        "--untrusted", action="store_true", help="evaluate in the pinned ROCm container"
    )
    parser.add_argument("--image", default="solar-rocm:7.2")
    args = parser.parse_args()
    if args.untrusted:
        raise SystemExit(_run_container(args))
    architecture = ArchitectureProfile.load(args.arch_config)
    report = RocmEvaluator(architecture=architecture).evaluate(
        args.benchmark,
        args.solution,
        baseline=args.baseline,
        timing_profile=args.timing_profile,
        lock_clocks=not args.no_lock_clocks,
        trusted_local=True,
    )
    report.write(args.output)
    print(f"Evaluation written to {args.output}")
    exit_code = _report_exit_code(report)
    if report.failure:
        print(report.failure)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
