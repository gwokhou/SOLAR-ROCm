# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Audit official SOL-ExecBench workloads against the local AMD GPU."""

# The two official-problem CLIs intentionally expose the same input/device
# arguments while performing different phases of the pipeline.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solar.benchmark.sol_execbench import AmdCompatibilityAuditor, SolExecBenchProblem


def main() -> None:
    """Record AMD compatibility for the selected official workloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_dir")
    parser.add_argument(
        "--blob-root",
        action="append",
        default=[],
        help="Explicit root for workload safetensors paths (repeatable).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch HIP device (ROCm uses the cuda:N spelling; must resolve to AMD gfx).",
    )
    parser.add_argument("--output", default="compatibility.jsonl")
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    problem = SolExecBenchProblem.load(args.problem_dir, blob_roots=args.blob_root)
    selected = set(args.workload)
    auditor = AmdCompatibilityAuditor(problem, device=args.device)
    results = [
        auditor.audit(workload, execute=not args.static_only)
        for workload in problem.workloads
        if not selected or workload.uuid in selected
    ]
    if selected - {result["workload_uuid"] for result in results}:
        parser.error("unknown workload UUID(s)")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results),
        encoding="utf-8",
    )
    for result in results:
        print(
            f"{result['workload_uuid']}: {result['status']} ({result['reason_code']})"
        )
    if any(
        result["status"] in {"execution_failed", "not_checked"} for result in results
    ):
        raise SystemExit(3)
    if any(result["status"] == "incompatible" for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
