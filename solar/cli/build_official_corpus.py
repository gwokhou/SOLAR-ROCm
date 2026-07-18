# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Rebuild every pinned official corpus artifact without adapting workloads."""

# The batch and audit CLIs intentionally share temporary materialization
# cleanup and module-entry boilerplate.
# pylint: disable=consider-using-with,duplicate-code,too-many-arguments

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from solar.benchmark.official_corpus import OfficialCorpusManifest


def problem_build_commands(
    manifest: OfficialCorpusManifest,
    materialized_root: Path,
    artifact_root: Path,
    *,
    device: str,
    orojenesis_home: str | None,
    blob_roots: tuple[str, ...],
    python_executable: str = sys.executable,
) -> list[list[str]]:
    """Return one deterministic strict-build command per selected problem."""
    problems = sorted({(entry.config, entry.problem) for entry in manifest.entries})
    commands: list[list[str]] = []
    for config, problem in problems:
        command = [
            python_executable,
            "-m",
            "solar.cli.build_source_to_sol",
            str(materialized_root / config / problem),
            "--output",
            str(artifact_root / config / problem),
            "--device",
            device,
            "--arch-config",
            str(manifest.architecture_profile_path),
        ]
        if orojenesis_home:
            command.extend(("--orojenesis-home", orojenesis_home))
        for root in blob_roots:
            command.extend(("--blob-root", root))
        commands.append(command)
    return commands


def main() -> None:
    """Materialize the manifest and rebuild each selected problem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--materialized-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--orojenesis-home")
    parser.add_argument("--blob-root", action="append", default=[])
    args = parser.parse_args()

    manifest = OfficialCorpusManifest.load(args.manifest)
    artifact_root = Path(args.output).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.materialized_root:
        materialized_root = Path(args.materialized_root).resolve()
        materialized_root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="solar-official-build-")
        materialized_root = Path(temporary.name)

    records: list[dict[str, object]] = []
    try:
        manifest.materialize(args.dataset_root, materialized_root)
        commands = problem_build_commands(
            manifest,
            materialized_root,
            artifact_root,
            device=args.device,
            orojenesis_home=args.orojenesis_home,
            blob_roots=tuple(args.blob_root),
        )
        for command in commands:
            completed = subprocess.run(command, check=False)
            # build_source_to_sol uses 2 when at least one attempted workload
            # has terminal incompatibility evidence and none failed.  It uses
            # 3 for failed or unchecked work, which must stop this batch.
            if completed.returncode not in {0, 2}:
                raise subprocess.CalledProcessError(completed.returncode, command)
            benchmark_path = (
                Path(command[command.index("--output") + 1]) / "benchmark.yaml"
            )
            if not benchmark_path.is_file():
                raise RuntimeError(f"corpus build did not produce {benchmark_path}")
            records.append(
                {
                    "benchmark": str(benchmark_path.relative_to(artifact_root)),
                    "benchmark_sha256": hashlib.sha256(
                        benchmark_path.read_bytes()
                    ).hexdigest(),
                    "terminal_incompatible": completed.returncode == 2,
                }
            )
        index = {
            "schema_version": 2,
            "manifest": str(manifest.path),
            "manifest_schema_version": manifest.schema_version,
            "manifest_sha256": hashlib.sha256(manifest.path.read_bytes()).hexdigest(),
            "architecture_profile_sha256": manifest.architecture_profile_sha256,
            "architecture_hash": manifest.architecture_hash,
            "problems": records,
        }
        (artifact_root / "build-index.yaml").write_text(
            yaml.safe_dump(index, sort_keys=False), encoding="utf-8"
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
