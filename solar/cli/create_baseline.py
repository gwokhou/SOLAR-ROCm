# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Create an explicit versioned baseline from a stable evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create baseline.yaml from evaluation.yaml"
    )
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", default="baseline.yaml")
    parser.add_argument("--name")
    args = parser.parse_args()
    evaluation = yaml.safe_load(Path(args.evaluation).read_text()) or {}
    if evaluation.get("timing_profile") == "quick":
        raise SystemExit("quick timing cannot create a baseline")
    if not evaluation.get("clocks_locked"):
        raise SystemExit("a verified STABLE_PEAK evaluation is required")
    workloads = {}
    for item in evaluation.get("workloads") or []:
        if not item.get("correct") or not item.get("candidate_latency_ms"):
            raise SystemExit(f"workload is not baseline-ready: {item.get('name')}")
        timing = item.get("timing") or {}
        if not timing.get("stable"):
            raise SystemExit(f"workload timing is unstable: {item.get('name')}")
        workloads[str(item["name"])] = float(item["candidate_latency_ms"])
    if not workloads:
        raise SystemExit("evaluation contains no baseline-ready workloads")
    payload = {
        "schema_version": 1,
        "name": args.name
        or f"{evaluation['solution_name']}@{evaluation['solution_hash'][:12]}",
        "benchmark_hash": evaluation["benchmark_hash"],
        "solution_hash": evaluation["solution_hash"],
        "architecture_hash": evaluation["architecture_hash"],
        "gfx_target": evaluation["environment"]["gfx_target"],
        "timing_profile": evaluation["timing_profile"],
        "cache_policy": evaluation["cache_policy"],
        "environment_hash": evaluation["environment_hash"],
        "clocks_locked": True,
        "workloads": workloads,
    }
    Path(args.output).write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Baseline written to {args.output}")


if __name__ == "__main__":
    main()
