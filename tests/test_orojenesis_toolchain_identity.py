# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed unit coverage for installed Orojenesis provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from solar.analysis.orojenesis import (
    OROJENESIS_BUILDER_IMAGE,
    OROJENESIS_COMMIT,
    OROJENESIS_COMPILER_WRAPPER_SHA256,
    OROJENESIS_PROVENANCE_FILENAME,
    OROJENESIS_REPOSITORY,
    OROJENESIS_TREE_OID,
    OrojenesisError,
    OrojenesisRunner,
)


def _installed_toolchain(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "orojenesis"
    mapper = home / "bin" / "timeloop-mapper"
    mapper.parent.mkdir(parents=True)
    mapper.write_bytes(b"static-mapper-test-binary")
    mapper.chmod(0o755)
    provenance = {
        "schema_version": 1,
        "source": {
            "repository": OROJENESIS_REPOSITORY,
            "commit": OROJENESIS_COMMIT,
            "tree_git_oid": OROJENESIS_TREE_OID,
            "archive_sha256": "2" * 64,
        },
        "build": {
            "builder_image": OROJENESIS_BUILDER_IMAGE,
            "compiler": "g++ test",
            "compiler_wrapper_sha256": OROJENESIS_COMPILER_WRAPPER_SHA256,
        },
        "artifact": {
            "path": "bin/timeloop-mapper",
            "sha256": hashlib.sha256(mapper.read_bytes()).hexdigest(),
        },
    }
    provenance_path = home / OROJENESIS_PROVENANCE_FILENAME
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return home, provenance_path


def test_installed_toolchain_identity_is_hash_bound(tmp_path: Path) -> None:
    home, provenance_path = _installed_toolchain(tmp_path)

    runner = OrojenesisRunner(home)

    assert runner.toolchain_identity["verification_mode"] == "provenance_manifest"
    assert (
        runner.toolchain_identity["provenance_sha256"]
        == hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    )


def test_installed_toolchain_rejects_binary_and_revision_tampering(
    tmp_path: Path,
) -> None:
    home, provenance_path = _installed_toolchain(tmp_path)
    mapper = home / "bin" / "timeloop-mapper"
    mapper.write_bytes(mapper.read_bytes() + b"tampered")
    with pytest.raises(OrojenesisError, match="binary hash mismatch"):
        OrojenesisRunner(home)

    home, provenance_path = _installed_toolchain(tmp_path / "revision")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source"]["commit"] = "0" * 40
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(OrojenesisError, match="revision mismatch"):
        OrojenesisRunner(home)

    home, provenance_path = _installed_toolchain(tmp_path / "builder")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["build"]["builder_image"] = "untrusted-builder"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(OrojenesisError, match="builder image mismatch"):
        OrojenesisRunner(home)
