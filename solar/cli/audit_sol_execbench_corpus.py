# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Materialize and audit the pinned NVIDIA official representative corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from solar.benchmark.official_corpus import (
    OFFICIAL_DATASET_ID,
    OFFICIAL_DATASET_REVISION,
    OfficialCorpusManifest,
    verify_formal_entry,
)
from solar.benchmark.sol_execbench import AmdCompatibilityAuditor, SolExecBenchProblem


def _worker(problem_dir: str, workload_uuid: str, device: str, blobs: list[str]) -> int:
    problem = SolExecBenchProblem.load(problem_dir, blob_roots=blobs)
    matches = [
        workload for workload in problem.workloads if workload.uuid == workload_uuid
    ]
    if len(matches) != 1:
        raise ValueError(f"official workload not found: {workload_uuid}")
    result = AmdCompatibilityAuditor(problem, device=device).audit(
        matches[0], execute=True
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _isolated_audit(
    problem_dir: Path,
    workload_uuid: str,
    *,
    device: str,
    blob_roots: list[str],
    timeout: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "solar.cli.audit_sol_execbench_corpus",
        "--worker",
        "--problem-dir",
        str(problem_dir),
        "--workload-uuid",
        workload_uuid,
        "--device",
        device,
    ]
    for root in blob_roots:
        command.extend(("--blob-root", root))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": 2,
            "status": "execution_failed",
            "reason_code": "audit_timeout",
            "stage": "isolated_process",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "fallbacks_used": [],
        }
    if completed.returncode:
        return {
            "schema_version": 2,
            "status": "execution_failed",
            "reason_code": "audit_worker_failed",
            "stage": "isolated_process",
            "error": {
                "type": "WorkerProcessError",
                "message": completed.stderr[-4000:],
            },
            "fallbacks_used": [],
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 2,
            "status": "execution_failed",
            "reason_code": "invalid_worker_evidence",
            "stage": "isolated_process",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "fallbacks_used": [],
        }
    if result.get("fallbacks_used") != []:
        raise RuntimeError("official corpus worker attempted a fallback")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--dataset-root")
    parser.add_argument("--materialized-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--output", default="official-corpus-audit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blob-root", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--problem-dir")
    parser.add_argument("--workload-uuid")
    args = parser.parse_args()

    if args.worker:
        if not args.problem_dir or not args.workload_uuid:
            parser.error("worker requires --problem-dir and --workload-uuid")
        raise SystemExit(
            _worker(args.problem_dir, args.workload_uuid, args.device, args.blob_root)
        )
    if not args.manifest or not args.dataset_root:
        parser.error("manifest and --dataset-root are required")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    manifest = OfficialCorpusManifest.load(args.manifest)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.materialized_root:
        materialized = Path(args.materialized_root).resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="solar-official-corpus-")
        materialized = Path(temporary.name)
    try:
        manifest.materialize(args.dataset_root, materialized)
        results: dict[str, dict[str, Any]] = {}
        for entry in manifest.entries:
            problem_dir = materialized / entry.config / entry.problem
            compatibility = _isolated_audit(
                problem_dir,
                entry.workload_uuid,
                device=args.device,
                blob_roots=args.blob_root,
                timeout=args.timeout,
            )
            if compatibility.get("reason_code") == "runtime_oom":
                confirmation = _isolated_audit(
                    problem_dir,
                    entry.workload_uuid,
                    device=args.device,
                    blob_roots=args.blob_root,
                    timeout=args.timeout,
                )
                repeatable = (
                    confirmation.get("status") == "incompatible"
                    and confirmation.get("reason_code") == "runtime_oom"
                    and confirmation.get("stage") == compatibility.get("stage")
                )
                compatibility["oom_confirmation"] = {
                    "repeatable_in_fresh_process": repeatable,
                    "second_status": confirmation.get("status"),
                    "second_reason_code": confirmation.get("reason_code"),
                    "second_stage": confirmation.get("stage"),
                }
                if not repeatable:
                    compatibility["status"] = "execution_failed"
                    compatibility["reason_code"] = "non_repeatable_runtime_oom"
            formal = verify_formal_entry(
                entry,
                args.artifact_root,
                expected_architecture_hash=manifest.architecture_hash,
            )
            results[entry.slot] = {
                "identity": {
                    "config": entry.config,
                    "problem": entry.problem,
                    "workload_uuid": entry.workload_uuid,
                },
                "compatibility": compatibility,
                **formal,
            }
            print(
                f"{entry.slot}: {compatibility.get('status')} "
                f"({compatibility.get('reason_code')}); "
                f"formal_attested={formal['formal_attested']}"
            )

        terminal = all(
            result["compatibility"].get("status") in {"compatible", "incompatible"}
            for result in results.values()
        )
        compatible_formal = all(
            result["compatibility"].get("status") != "compatible"
            or bool(result.get("formal_attested"))
            for result in results.values()
        )
        coverage = manifest.coverage(results)
        formal_coverage_complete = bool(coverage["formal_requirements_met"])
        report: dict[str, Any] = {
            "schema_version": 2,
            "source": {
                "dataset_id": OFFICIAL_DATASET_ID,
                "revision": OFFICIAL_DATASET_REVISION,
                "manifest": str(Path(args.manifest)),
                "manifest_sha256": hashlib.sha256(
                    Path(args.manifest).read_bytes()
                ).hexdigest(),
                "architecture_profile": manifest.architecture_profile_reference,
                "architecture_profile_sha256": (manifest.architecture_profile_sha256),
                "architecture_hash": manifest.architecture_hash,
                "manifest_schema_version": manifest.schema_version,
            },
            "results": results,
            "coverage": coverage,
            "gate": {
                "terminal_evidence_complete": terminal,
                "all_compatible_formally_attested": compatible_formal,
                "formal_coverage_complete": formal_coverage_complete,
                "passed": (terminal and compatible_formal and formal_coverage_complete),
            },
        }
        Path(args.output).write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
        )
        if not report["gate"]["passed"]:
            raise SystemExit(3)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
