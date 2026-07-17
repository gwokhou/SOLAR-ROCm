# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Create replayable reference-to-einsum verification attestations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

from solar.verification import create_verification_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Numerically verify every benchmark reference against its bound "
            "einsum graph and write in-toto verification.yaml statements."
        )
    )
    parser.add_argument("--benchmark", required=True, help="Path to benchmark.yaml")
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="Only verify this workload name (repeatable).",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Bind generated attestations into benchmark.yaml and bump schema to v2.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark).resolve()
    root = benchmark_path.parent
    data = yaml.safe_load(benchmark_path.read_text()) or {}
    reference = data.get("reference") or {}
    tolerance = data.get("tolerance") or {}
    selected = set(args.workload)
    found: set[str] = set()

    for workload in data.get("workloads") or []:
        name = str(workload.get("name", ""))
        if selected and name not in selected:
            continue
        found.add(name)
        analysis = workload.get("analysis") or {}
        graph_path = root / str(analysis.get("source_graph", ""))
        verification = workload.get("verification") or {}
        relative_output = str(
            verification.get("path", f"verification/{name}/verification.yaml")
        )
        output_path = root / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_verification_artifact(
            reference_path=root / str(reference.get("source", "")),
            reference_entry_point=str(reference.get("entry_point", "run")),
            input_factory_name=str(reference.get("input_factory", "get_inputs")),
            graph_path=graph_path,
            workload_name=name,
            workload_parameters=dict(workload.get("parameters") or {}),
            output_path=output_path,
            atol=float(tolerance.get("atol", 1e-5)),
            rtol=float(tolerance.get("rtol", 1e-5)),
            device=args.device,
        )
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        workload["verification"] = {"path": relative_output, "sha256": digest}
        print(f"verified {name}: {relative_output} ({digest})")

    missing = selected - found
    if missing:
        parser.error("unknown workload(s): " + ", ".join(sorted(missing)))
    if args.update_manifest:
        if selected and len(found) != len(data.get("workloads") or []):
            incomplete = [
                str(item.get("name", ""))
                for item in data.get("workloads") or []
                if not item.get("verification")
            ]
            if incomplete:
                parser.error(
                    "cannot enable schema v2 until every workload is verified: "
                    + ", ".join(incomplete)
                )
        data["schema_version"] = 2
        benchmark_path.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"updated trusted benchmark manifest: {benchmark_path}")


if __name__ == "__main__":
    main()
