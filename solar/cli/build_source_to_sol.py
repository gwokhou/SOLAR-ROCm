# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build verified SOL artifacts from an official SOL-ExecBench problem."""

# The two official-problem CLIs intentionally expose the same input/device
# arguments while performing different phases of the pipeline.
# pylint: disable=duplicate-code,too-many-locals,too-many-statements,consider-using-from-import,import-outside-toplevel,no-name-in-module,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.analysis.orojenesis import OrojenesisError, OrojenesisRunner
from solar.benchmark.sol_execbench import (
    AmdCompatibilityAuditor,
    SolExecBenchProblem,
    standalone_reference_source,
    write_compatibility_artifact,
)
from solar.einsum import PyTorchToEinsum
from solar.graph.torchview_processor import TorchviewProcessor
from solar.verification import create_verification_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path) -> Any:
    name = f"_solar_problem_{_sha256(path)[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generated reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _copy_blobs(problem: SolExecBenchProblem, output: Path) -> None:
    for workload in problem.workloads:
        for input_spec in workload.raw["inputs"].values():
            if str(input_spec.get("type")) != "safetensors":
                continue
            relative = Path(str(input_spec["path"]))
            try:
                source = problem.resolve_blob(str(input_spec["path"]))
            except (FileNotFoundError, ValueError):
                # The auditor records missing/ambiguous external input per
                # workload. Do not synthesize or substitute a blob here.
                continue
            destination = output / "data" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _extract_graph(
    reference: Any,
    inputs: tuple[Any, ...],
    *,
    device: str,
    output: Path,
    name: str,
) -> Path:
    import torch.nn as nn
    from solar._vendor import torchview

    class ReferenceModule(nn.Module):
        def forward(self, *args: Any) -> Any:
            return reference(*args)

    module = ReferenceModule().eval()
    graph = torchview.draw_graph(
        module,
        input_data=list(inputs),
        device=device,
        save_graph=False,
        expand_nested=True,
        depth=float("inf"),
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=False,
        strict=True,
        collect_attributes=True,
    )
    TorchviewProcessor().process_graph(graph, str(output), name, module)
    PyTorchToEinsum(strict=True).convert(
        output / "pytorch_graph.yaml", output, copy_graph=False, enable_rename=False
    )
    return output / "einsum_graph.yaml"


def _failed_result(
    base: dict[str, Any], reason: str, stage: str, exc: Exception
) -> dict[str, Any]:
    result = dict(base)
    result.update(
        {
            "status": "execution_failed",
            "reason_code": reason,
            "stage": stage,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
            "fallbacks_used": [],
        }
    )
    return result


def main() -> None:
    """Build only formally verified artifacts for compatible AMD workloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_dir")
    parser.add_argument(
        "--blob-root",
        action="append",
        default=[],
        help="Explicit root for workload safetensors paths (repeatable).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch HIP device (ROCm uses the cuda:N spelling; must resolve to AMD gfx).",
    )
    parser.add_argument("--arch-config", required=True)
    parser.add_argument("--orojenesis-home")
    parser.add_argument("--workload", action="append", default=[])
    args = parser.parse_args()

    problem = SolExecBenchProblem.load(args.problem_dir, blob_roots=args.blob_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "reference.py"
    _copy_blobs(problem, output)
    reference_path.write_text(standalone_reference_source(problem))
    reference_module: Any | None = None
    auditor = AmdCompatibilityAuditor(problem, device=args.device)
    selected = set(args.workload)
    manifest_workloads: list[dict[str, Any]] = []
    final_statuses: list[str] = []

    runner: OrojenesisRunner | None = None
    runner_error: Exception | None = None
    try:
        runner = OrojenesisRunner(args.orojenesis_home)
    except OrojenesisError as exc:
        runner_error = exc

    for workload in problem.workloads:
        if selected and workload.uuid not in selected:
            continue
        workdir = output / "workloads" / workload.uuid
        workdir.mkdir(parents=True, exist_ok=True)
        compatibility = auditor.audit(workload, execute=True)
        if compatibility["status"] == "compatible" and runner_error is not None:
            compatibility = _failed_result(
                compatibility, "toolchain_unavailable", "orojenesis_init", runner_error
            )

        tolerance = workload.raw.get("tolerance") or {}
        normalized_tolerance: dict[str, Any] = {
            "max_atol": float(tolerance.get("max_atol", 1e-2)),
            "max_rtol": float(tolerance.get("max_rtol", 1e-2)),
            "required_matched_ratio": float(
                tolerance.get("required_matched_ratio", 0.99)
            ),
            "max_error_cap": (
                float(tolerance["max_error_cap"])
                if tolerance.get("max_error_cap") is not None
                else None
            ),
            "allow_negative_inf": bool(tolerance.get("allow_negative_inf", False)),
        }
        numeric_tolerances = [
            normalized_tolerance["max_atol"],
            normalized_tolerance["max_rtol"],
            normalized_tolerance["required_matched_ratio"],
        ]
        if normalized_tolerance["max_error_cap"] is not None:
            numeric_tolerances.append(normalized_tolerance["max_error_cap"])
        if not all(math.isfinite(value) and value >= 0 for value in numeric_tolerances):
            parser.error(f"workload {workload.uuid} has invalid tolerance values")
        if normalized_tolerance["required_matched_ratio"] > 1:
            parser.error(f"workload {workload.uuid} required_matched_ratio exceeds one")
        entry: dict[str, Any] = {
            "name": workload.uuid,
            "uuid": workload.uuid,
            "status": compatibility["status"],
            "parameters": {"uuid": workload.uuid},
            "tolerance": normalized_tolerance,
        }
        if compatibility["status"] == "compatible":
            stage = "reference_import"
            try:
                # Do not import or execute the externally supplied reference
                # until its AMD/cycle preflight has recorded it as compatible.
                if reference_module is None:
                    reference_module = _load_module(reference_path)
                stage = "input_generation"
                inputs = tuple(
                    reference_module.get_inputs(
                        {"uuid": workload.uuid, "seed": 200}, args.device
                    )
                )
                stage = "graph_extraction"
                graph_path = _extract_graph(
                    reference_module.run,
                    inputs,
                    device=args.device,
                    output=workdir,
                    name=str(problem.definition["name"]),
                )
                stage = "analysis"
                analysis = EinsumGraphAnalyzer().analyze_graph(
                    graph_path,
                    workdir,
                    precision="fp16",
                    copy_graph=False,
                    strict=True,
                    architecture=args.arch_config,
                    orojenesis_runner=runner,
                    require_orojenesis=True,
                )
                if analysis is None:
                    raise RuntimeError("analysis did not produce an artifact")
                analysis_path = workdir / "analysis.yaml"
                verification_path = workdir / "verification.yaml"
                stage = "verification"
                create_verification_artifact(
                    reference_path=reference_path,
                    reference_entry_point="run",
                    input_factory_name="get_inputs",
                    graph_path=graph_path,
                    workload_name=workload.uuid,
                    workload_parameters={"uuid": workload.uuid},
                    output_path=verification_path,
                    atol=normalized_tolerance["max_atol"],
                    rtol=normalized_tolerance["max_rtol"],
                    required_matched_ratio=normalized_tolerance[
                        "required_matched_ratio"
                    ],
                    max_error_cap=normalized_tolerance["max_error_cap"],
                    allow_negative_inf=normalized_tolerance["allow_negative_inf"],
                    device=args.device,
                )
                entry["analysis"] = {
                    "path": str(analysis_path.relative_to(output)),
                    "sha256": _sha256(analysis_path),
                    "source_graph": str(graph_path.relative_to(output)),
                    "source_graph_sha256": _sha256(graph_path),
                }
                entry["verification"] = {
                    "path": str(verification_path.relative_to(output)),
                    "sha256": _sha256(verification_path),
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                compatibility = _failed_result(
                    compatibility, f"{stage}_failed", stage, exc
                )
                entry["status"] = compatibility["status"]

        compatibility_path = workdir / "compatibility.yaml"
        compatibility_sha = write_compatibility_artifact(
            compatibility, compatibility_path
        )
        entry["compatibility"] = {
            "path": str(compatibility_path.relative_to(output)),
            "sha256": compatibility_sha,
        }
        if entry["status"] != "compatible":
            entry.pop("analysis", None)
            entry.pop("verification", None)
        manifest_workloads.append(entry)
        final_statuses.append(entry["status"])
        print(f"{workload.uuid}: {entry['status']} ({compatibility['reason_code']})")

    if selected - {entry["uuid"] for entry in manifest_workloads}:
        parser.error("unknown workload UUID(s)")
    tolerances = [workload.raw.get("tolerance") or {} for workload in problem.workloads]
    manifest = {
        "schema_version": 3,
        "name": str(problem.definition["name"]),
        "source": {
            "format": "sol_execbench",
            "schema_commit": "a9fa0804c793d438e70850c33fe34426e66d53dd",
            "definition_sha256": _sha256(problem.definition_path),
            "workload_sha256": _sha256(problem.workload_path),
        },
        "reference": {
            "source": "reference.py",
            "entry_point": "run",
            "input_factory": "get_inputs",
        },
        "tolerance": {
            "atol": max(
                (float(item.get("max_atol", 1e-2)) for item in tolerances), default=1e-2
            ),
            "rtol": max(
                (float(item.get("max_rtol", 1e-2)) for item in tolerances), default=1e-2
            ),
        },
        "cache_policy": "cold",
        "precision": "fp16",
        "workloads": manifest_workloads,
    }
    (output / "benchmark.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    if any(status in {"execution_failed", "not_checked"} for status in final_statuses):
        raise SystemExit(3)
    if any(status == "incompatible" for status in final_statuses):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
