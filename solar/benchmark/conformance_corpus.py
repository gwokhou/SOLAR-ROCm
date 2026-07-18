# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hash-bound local contract cases kept separate from the official corpus."""

# Manifest loading intentionally validates identity, coverage, and bound pytest
# node IDs in one fail-closed pass.
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from solar.benchmark.models import canonical_hash
from solar.rocm import ArchitectureProfile

CONFORMANCE_SUITE_ID = "solar/rdna4-source-to-sol-conformance"
_VERDICTS = {"accept", "reject"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ConformanceCase:
    """One locally authored policy contract, never an official workload."""

    case_id: str
    feature: str
    verdict: str
    expected_contract: str
    regression_test: str
    regression_path: Path
    regression_function: str


@dataclass(frozen=True)
class ConformanceCorpusManifest:
    """Validated local conformance suite and its architecture identity."""

    path: Path
    architecture_profile_reference: str
    architecture_profile_path: Path
    architecture_profile_sha256: str
    architecture_hash: str
    feature_minimums: dict[str, int]
    verdict_minimums: dict[str, int]
    cases: tuple[ConformanceCase, ...]

    @classmethod
    def load(cls, path: str | Path) -> "ConformanceCorpusManifest":
        """Load and validate a repository-local conformance manifest."""
        manifest_path = Path(path).resolve()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if data.get("schema_version") != 1:
            raise ValueError("conformance corpus must use schema_version 1")
        source = data.get("source") or {}
        if source != {"kind": "repository_local", "suite_id": CONFORMANCE_SUITE_ID}:
            raise ValueError("conformance corpus source identity mismatch")

        target = data.get("target") or {}
        profile_reference = str(target.get("architecture_profile", ""))
        relative_profile = Path(profile_reference)
        if not profile_reference or relative_profile.is_absolute():
            raise ValueError("target architecture_profile must be a relative path")
        profile_path = (manifest_path.parent / relative_profile).resolve()
        profile_sha256 = str(target.get("architecture_profile_sha256", ""))
        if not profile_path.is_file() or _file_sha256(profile_path) != profile_sha256:
            raise ValueError("target architecture profile identity mismatch")
        architecture = ArchitectureProfile.load(profile_path)
        architecture_identity = architecture.to_dict()
        architecture_identity.pop("source", None)

        requirements = data.get("requirements") or {}
        feature_minimums = {
            str(key): int(value)
            for key, value in (requirements.get("features") or {}).items()
        }
        verdict_minimums = {
            str(key): int(value)
            for key, value in (requirements.get("verdicts") or {}).items()
        }
        if (
            not feature_minimums
            or any(value <= 0 for value in feature_minimums.values())
            or set(verdict_minimums) != _VERDICTS
            or any(value <= 0 for value in verdict_minimums.values())
        ):
            raise ValueError(
                "conformance requirements need positive feature/verdict counts"
            )

        cases: list[ConformanceCase] = []
        for raw in data.get("cases") or []:
            regression_test = str(raw.get("regression_test", ""))
            parts = regression_test.split("::")
            if len(parts) != 2 or not parts[1].startswith("test_"):
                raise ValueError("conformance regression_test must be a pytest node id")
            regression_path = (manifest_path.parent / parts[0]).resolve()
            try:
                regression_path.relative_to(manifest_path.parents[2])
            except ValueError as exc:
                raise ValueError(
                    "conformance regression test escapes the repository"
                ) from exc
            if not regression_path.is_file():
                raise ValueError(f"conformance regression test is missing: {parts[0]}")
            tree = ast.parse(regression_path.read_text(encoding="utf-8"))
            functions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if parts[1] not in functions:
                raise ValueError(
                    f"conformance regression function is missing: {parts[1]}"
                )
            verdict = str(raw.get("verdict", ""))
            if verdict not in _VERDICTS:
                raise ValueError("conformance verdict must be accept or reject")
            contract = str(raw.get("expected_contract", ""))
            if not contract:
                raise ValueError("conformance cases require an expected contract")
            cases.append(
                ConformanceCase(
                    case_id=str(raw.get("id", "")),
                    feature=str(raw.get("feature", "")),
                    verdict=verdict,
                    expected_contract=contract,
                    regression_test=regression_test,
                    regression_path=regression_path,
                    regression_function=parts[1],
                )
            )
        case_ids = [case.case_id for case in cases]
        if not cases or any(not case_id for case_id in case_ids):
            raise ValueError("conformance cases require non-empty ids")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("conformance case ids must be unique")

        def _counts(field: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for case in cases:
                value = str(getattr(case, field))
                counts[value] = counts.get(value, 0) + 1
            return counts

        for label, minimums, counts in (
            ("feature", feature_minimums, _counts("feature")),
            ("verdict", verdict_minimums, _counts("verdict")),
        ):
            deficits = {
                key: minimum - counts.get(key, 0)
                for key, minimum in minimums.items()
                if counts.get(key, 0) < minimum
            }
            if deficits:
                raise ValueError(
                    f"conformance {label} requirements are unmet: {deficits}"
                )

        return cls(
            manifest_path,
            profile_reference,
            profile_path,
            profile_sha256,
            canonical_hash(architecture_identity),
            feature_minimums,
            verdict_minimums,
            tuple(cases),
        )

    def coverage(self, results: Mapping[str, bool]) -> dict[str, Any]:
        """Return a fail-closed gate over explicitly reported local case results."""
        expected = {case.case_id for case in self.cases}
        unknown = sorted(set(results) - expected)
        missing = sorted(expected - set(results))
        failed = sorted(
            case_id for case_id in expected if not results.get(case_id, False)
        )
        return {
            "denominator": len(expected),
            "passed_count": len(expected) - len(set(missing) | set(failed)),
            "missing": missing,
            "failed": failed,
            "unknown": unknown,
            "passed": not missing and not failed and not unknown,
        }


__all__ = [
    "CONFORMANCE_SUITE_ID",
    "ConformanceCase",
    "ConformanceCorpusManifest",
]
