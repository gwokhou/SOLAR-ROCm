# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Write the hash-bound identity shipped with the static Orojenesis mapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

OROJENESIS_REPOSITORY = "https://github.com/NVlabs/timeloop.git"
PROVENANCE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def build_provenance(
    home: Path,
    *,
    expected_commit: str,
    builder_image: str,
    compiler_wrapper: Path,
) -> dict:
    """Build a deterministic manifest from the checked-out source and binary."""
    home = home.resolve()
    mapper = home / "bin" / "timeloop-mapper"
    if not mapper.is_file():
        raise ValueError(f"missing mapper binary: {mapper}")
    commit = str(_run(["git", "-C", str(home), "rev-parse", "HEAD"]))
    if commit != expected_commit:
        raise ValueError(
            f"Orojenesis revision mismatch: expected {expected_commit}, got {commit}"
        )
    tree_oid = str(_run(["git", "-C", str(home), "rev-parse", "HEAD^{tree}"]))
    archive = _run(
        ["git", "-C", str(home), "archive", "--format=tar", "HEAD"],
        binary=True,
    )
    if not isinstance(archive, bytes):
        raise TypeError("git archive must produce bytes")
    compiler = str(_run([str(compiler_wrapper), "--version"])).splitlines()[0]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source": {
            "repository": OROJENESIS_REPOSITORY,
            "commit": commit,
            "tree_git_oid": tree_oid,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
        },
        "build": {
            "builder_image": builder_image,
            "compiler": compiler,
            "compiler_wrapper_sha256": _sha256(compiler_wrapper),
        },
        "artifact": {
            "path": "bin/timeloop-mapper",
            "sha256": _sha256(mapper),
        },
    }


def main() -> None:
    """Write provenance for the requested pinned mapper installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--builder-image", required=True)
    parser.add_argument("--compiler-wrapper", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provenance = build_provenance(
        args.home,
        expected_commit=args.expected_commit,
        builder_image=args.builder_image,
        compiler_wrapper=args.compiler_wrapper,
    )
    args.output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
