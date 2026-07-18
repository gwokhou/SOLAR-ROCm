# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pinned, non-adapting reader and gate for the official SOL-ExecBench corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from solar.analysis.resources import validate_resource_work
from solar.benchmark.models import BenchmarkSpec, canonical_hash
from solar.rocm import ArchitectureProfile

OFFICIAL_DATASET_ID = "nvidia/SOL-ExecBench"
OFFICIAL_DATASET_REVISION = "63699402f003496acc3af4eb534a5304a8ac1ea9"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CorpusEntry:
    slot: str
    config: str
    problem: str
    workload_uuid: str
    official_row_sha256: str
    official_workload_sha256: str
    operation: str
    dtype: str
    pass_kind: str
    dynamic_path: str
    input_kind: str
    golden_external_bytes: float
    golden_flops: float
    golden_resource_work: dict[str, dict[str, float]]
    golden_derivation: str


@dataclass(frozen=True)
class OfficialCorpusManifest:
    schema_version: int
    path: Path
    parquet_sha256: dict[str, str]
    architecture_profile_reference: str
    architecture_profile_path: Path
    architecture_profile_sha256: str
    architecture_hash: str
    formal_coverage_requirements: dict[str, tuple[str, ...]]
    formal_coverage_minimums: dict[str, dict[str, int]]
    required_combinations: tuple[dict[str, str | int], ...]
    footprint_minimums: dict[str, int]
    shape_pairs: dict[str, tuple[str, ...]]
    l2_bytes: int
    last_level_cache_bytes: int
    entries: tuple[CorpusEntry, ...]

    @classmethod
    def load(cls, path: str | Path) -> "OfficialCorpusManifest":
        manifest_path = Path(path).resolve()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        schema_version = int(data.get("schema_version", 0))
        if schema_version not in {1, 2}:
            raise ValueError("official corpus manifest must use schema_version 1 or 2")
        source = data.get("source") or {}
        if source.get("dataset_id") != OFFICIAL_DATASET_ID:
            raise ValueError("corpus must come from the NVIDIA official dataset")
        if source.get("revision") != OFFICIAL_DATASET_REVISION:
            raise ValueError("corpus must pin the reviewed official dataset revision")
        parquet_sha256 = {
            str(key): str(value)
            for key, value in (source.get("parquet_sha256") or {}).items()
        }
        target = data.get("target") or {}
        relative_profile = Path(str(target.get("architecture_profile", "")))
        if not str(relative_profile) or relative_profile.is_absolute():
            raise ValueError("target architecture_profile must be a relative path")
        architecture_profile_path = (manifest_path.parent / relative_profile).resolve()
        architecture_profile_sha256 = str(target.get("architecture_profile_sha256", ""))
        if (
            not architecture_profile_path.is_file()
            or _file_sha256(architecture_profile_path) != architecture_profile_sha256
        ):
            raise ValueError("target architecture profile identity mismatch")
        architecture = ArchitectureProfile.load(architecture_profile_path)
        architecture_identity = architecture.to_dict()
        architecture_identity.pop("source", None)
        architecture_hash = canonical_hash(architecture_identity)
        axes = ("operation", "dtype", "pass_kind", "dynamic_path", "input_kind")
        raw_requirements = data.get("formal_coverage_requirements") or {}
        if schema_version == 1:
            raw_axes = raw_requirements
            raw_combinations: list[dict[str, Any]] = []
            raw_footprints: dict[str, Any] = {}
            raw_shape_pairs: list[dict[str, Any]] = []
        else:
            raw_axes = raw_requirements.get("axes") or {}
            raw_combinations = raw_requirements.get("combinations") or []
            raw_footprints = raw_requirements.get("footprint_classes") or {}
            raw_shape_pairs = raw_requirements.get("shape_pairs") or []
        if set(raw_axes) != set(axes):
            raise ValueError(
                "official corpus must declare every formal coverage requirement axis"
            )
        formal_coverage_minimums: dict[str, dict[str, int]] = {}
        for axis in axes:
            raw_axis = raw_axes.get(axis) or {}
            if isinstance(raw_axis, list):
                minimums = {str(item): 1 for item in raw_axis}
            elif isinstance(raw_axis, dict):
                minimums = {str(key): int(value) for key, value in raw_axis.items()}
            else:
                raise ValueError("formal coverage axes must be lists or mappings")
            if not minimums or any(value <= 0 for value in minimums.values()):
                raise ValueError("formal coverage minimums must be positive")
            formal_coverage_minimums[axis] = minimums
        formal_coverage_requirements = {
            axis: tuple(minimums) for axis, minimums in formal_coverage_minimums.items()
        }
        required_combinations: list[dict[str, str | int]] = []
        manifest_to_attribute = {"pass": "pass_kind", "pass_kind": "pass_kind"}
        for raw_combination in raw_combinations:
            if not isinstance(raw_combination, dict):
                raise ValueError("formal coverage combinations must be mappings")
            combination: dict[str, str | int] = {}
            for key, value in raw_combination.items():
                normalized = manifest_to_attribute.get(str(key), str(key))
                if normalized == "min_count":
                    combination[normalized] = int(value)
                elif normalized in axes:
                    combination[normalized] = str(value)
                else:
                    raise ValueError(f"unknown formal combination field: {key}")
            combination.setdefault("min_count", 1)
            if int(combination["min_count"]) <= 0 or len(combination) < 2:
                raise ValueError("formal coverage combinations require positive counts")
            required_combinations.append(combination)
        allowed_footprints = {"fits_l2", "l2_to_llc", "exceeds_llc"}
        footprint_minimums = {
            str(key): int(value) for key, value in raw_footprints.items()
        }
        if set(footprint_minimums) - allowed_footprints or any(
            value <= 0 for value in footprint_minimums.values()
        ):
            raise ValueError("invalid external-footprint coverage requirements")
        shape_pairs: dict[str, tuple[str, ...]] = {}
        for raw_pair in raw_shape_pairs:
            name = str(raw_pair.get("name", ""))
            slots = tuple(str(item) for item in raw_pair.get("slots") or [])
            if not name or len(slots) < 2 or len(slots) != len(set(slots)):
                raise ValueError("shape-pair requirements need a name and unique slots")
            shape_pairs[name] = slots
        raw_entries = data.get("entries") or []
        maximum_entries = 15 if schema_version == 1 else 20
        if not 10 <= len(raw_entries) <= maximum_entries:
            raise ValueError(
                f"official representative corpus must contain 10-{maximum_entries} workloads"
            )
        entries: list[CorpusEntry] = []
        for raw in raw_entries:
            golden = raw.get("golden") or {}
            resource_work = validate_resource_work(golden.get("resource_work") or {})
            values = [
                float(golden.get("external_bytes", -1)),
                float(golden.get("flops", -1)),
            ]
            if any(value < 0 for value in values) or not str(
                golden.get("derivation", "")
            ):
                raise ValueError(
                    "every corpus workload requires independent FLOP/byte goldens"
                )
            entries.append(
                CorpusEntry(
                    slot=str(raw["slot"]),
                    config=str(raw["config"]),
                    problem=str(raw["problem"]),
                    workload_uuid=str(raw["workload_uuid"]),
                    official_row_sha256=str(raw["official_row_sha256"]),
                    official_workload_sha256=str(raw["official_workload_sha256"]),
                    operation=str(raw["operation"]),
                    dtype=str(raw["dtype"]),
                    pass_kind=str(raw["pass"]),
                    dynamic_path=str(raw["dynamic_path"]),
                    input_kind=str(raw["input_kind"]),
                    golden_external_bytes=values[0],
                    golden_flops=values[1],
                    golden_resource_work=resource_work,
                    golden_derivation=str(golden["derivation"]),
                )
            )
        keys = [(entry.config, entry.problem, entry.workload_uuid) for entry in entries]
        entry_slots = [entry.slot for entry in entries]
        if len(keys) != len(set(keys)) or len(entry_slots) != len(set(entry_slots)):
            raise ValueError("corpus slots and workload identities must be unique")
        known_slots = set(entry_slots)
        if any(
            slot not in known_slots
            for pair_slots in shape_pairs.values()
            for slot in pair_slots
        ):
            raise ValueError("shape-pair coverage references an unknown corpus slot")
        required_operations = {"attention", "norm", "moe", "ssm", "conv"}
        if not required_operations.issubset({entry.operation for entry in entries}):
            raise ValueError("corpus omits a required operation family")
        if {"forward", "backward"} - {entry.pass_kind for entry in entries}:
            raise ValueError("corpus must cover forward and backward")
        if not {"fp8", "nvfp4"}.issubset({entry.dtype for entry in entries}):
            raise ValueError("corpus must retain official FP8 and NVFP4 cases")
        selected = {
            axis: {str(getattr(entry, axis)) for entry in entries} for axis in axes
        }
        for axis, required in formal_coverage_requirements.items():
            missing = set(required) - selected[axis]
            if missing:
                raise ValueError(
                    f"formal coverage requirements select no {axis}: {sorted(missing)}"
                )
        return cls(
            schema_version,
            manifest_path,
            parquet_sha256,
            str(relative_profile),
            architecture_profile_path,
            architecture_profile_sha256,
            architecture_hash,
            formal_coverage_requirements,
            formal_coverage_minimums,
            tuple(required_combinations),
            footprint_minimums,
            shape_pairs,
            int(architecture.l2_bytes),
            int(architecture.last_level_cache_bytes),
            tuple(entries),
        )

    def materialize(self, dataset_root: str | Path, output_root: str | Path) -> Path:
        """Materialize selected official rows without semantic adaptation."""
        import pandas as pd

        dataset = Path(dataset_root).resolve()
        output = Path(output_root).resolve()
        selected_workloads: dict[Path, list[dict[str, Any]]] = {}
        for config, expected in self.parquet_sha256.items():
            parquet = dataset / "data" / f"{config}.parquet"
            if not parquet.is_file() or _file_sha256(parquet) != expected:
                raise ValueError(f"official parquet identity mismatch: {config}")
        frames: dict[str, Any] = {}
        for entry in self.entries:
            if entry.config not in frames:
                frames[entry.config] = pd.read_parquet(
                    dataset / "data" / f"{entry.config}.parquet"
                )
            matches = frames[entry.config]
            matches = matches[matches["name"] == entry.problem]
            if len(matches) != 1:
                raise ValueError(
                    f"official problem missing or duplicated: {entry.problem}"
                )
            row = matches.iloc[0].to_dict()
            normalized_row = {
                key: (None if value is None else value) for key, value in row.items()
            }
            if _canonical_sha256(normalized_row) != entry.official_row_sha256:
                raise ValueError(f"official row drifted: {entry.problem}")
            workloads = json.loads(str(row["workloads"]))
            selected = [
                workload
                for workload in workloads
                if str(workload.get("uuid")) == entry.workload_uuid
            ]
            if (
                len(selected) != 1
                or _canonical_sha256(selected[0]) != entry.official_workload_sha256
            ):
                raise ValueError(f"official workload drifted: {entry.workload_uuid}")
            definition = {
                "name": str(row["name"]),
                "description": str(row.get("description") or ""),
                "hf_id": str(row.get("hf_id") or ""),
                "axes": json.loads(str(row["axes"])),
                "inputs": json.loads(str(row["inputs"])),
                "outputs": json.loads(str(row["outputs"])),
                "reference": str(row["reference"]),
            }
            custom_entrypoint = row.get("custom_inputs_entrypoint")
            if isinstance(custom_entrypoint, str) and custom_entrypoint:
                definition["custom_inputs_entrypoint"] = custom_entrypoint
            problem_root = output / entry.config / entry.problem
            problem_root.mkdir(parents=True, exist_ok=True)
            (problem_root / "definition.json").write_text(
                json.dumps(definition, indent=2) + "\n",
                encoding="utf-8",
            )
            selected_workloads.setdefault(problem_root, []).append(selected[0])
        for problem_root, workloads in selected_workloads.items():
            (problem_root / "workload.jsonl").write_text(
                "".join(
                    json.dumps(workload, sort_keys=True) + "\n"
                    for workload in workloads
                ),
                encoding="utf-8",
            )
        return output

    def coverage(self, results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Report selection and formal compatible coverage without hiding failures."""
        axes = ("operation", "dtype", "pass_kind", "dynamic_path", "input_kind")
        selection: dict[str, dict[str, int]] = {axis: {} for axis in axes}
        formal: dict[str, dict[str, int]] = {axis: {} for axis in axes}
        for entry in self.entries:
            result = results.get(entry.slot) or {}
            for axis in axes:
                value = str(getattr(entry, axis))
                selection[axis][value] = selection[axis].get(value, 0) + 1
                if bool(result.get("formal_attested")):
                    formal[axis][value] = formal[axis].get(value, 0) + 1
                else:
                    formal[axis].setdefault(value, 0)
        missing_requirements = {
            axis: [
                value
                for value, minimum in self.formal_coverage_minimums[axis].items()
                if formal[axis].get(value, 0) < minimum
            ]
            for axis in axes
        }
        deficits = {
            axis: {
                value: minimum - formal[axis].get(value, 0)
                for value, minimum in self.formal_coverage_minimums[axis].items()
                if formal[axis].get(value, 0) < minimum
            }
            for axis in axes
        }
        combination_coverage: list[dict[str, Any]] = []
        missing_combinations: list[dict[str, Any]] = []
        for requirement in self.required_combinations:
            minimum = int(requirement.get("min_count", 1))
            fields = {
                str(key): str(value)
                for key, value in requirement.items()
                if key != "min_count"
            }
            selected_count = 0
            formal_count = 0
            for entry in self.entries:
                if all(
                    str(getattr(entry, key)) == value for key, value in fields.items()
                ):
                    selected_count += 1
                    if bool((results.get(entry.slot) or {}).get("formal_attested")):
                        formal_count += 1
            item = {
                **fields,
                "min_count": minimum,
                "selection_count": selected_count,
                "formal_attested_count": formal_count,
            }
            combination_coverage.append(item)
            if formal_count < minimum:
                missing_combinations.append(item)

        footprint_selection = {
            name: 0 for name in ("fits_l2", "l2_to_llc", "exceeds_llc")
        }
        footprint_formal = dict(footprint_selection)
        for entry in self.entries:
            if entry.golden_external_bytes <= self.l2_bytes:
                footprint = "fits_l2"
            elif entry.golden_external_bytes <= self.last_level_cache_bytes:
                footprint = "l2_to_llc"
            else:
                footprint = "exceeds_llc"
            footprint_selection[footprint] += 1
            if bool((results.get(entry.slot) or {}).get("formal_attested")):
                footprint_formal[footprint] += 1
        missing_footprints = {
            name: minimum - footprint_formal.get(name, 0)
            for name, minimum in self.footprint_minimums.items()
            if footprint_formal.get(name, 0) < minimum
        }
        shape_pair_coverage = {
            name: {
                "slots": list(slots),
                "formal_attested": all(
                    bool((results.get(slot) or {}).get("formal_attested"))
                    for slot in slots
                ),
            }
            for name, slots in self.shape_pairs.items()
        }
        missing_shape_pairs = [
            name
            for name, item in shape_pair_coverage.items()
            if not item["formal_attested"]
        ]
        requirements_met = not any(
            (
                any(missing_requirements.values()),
                missing_combinations,
                missing_footprints,
                missing_shape_pairs,
            )
        )
        return {
            "denominator": len(self.entries),
            "selection": selection,
            "formal_attested": formal,
            "formal_requirements": {
                axis: dict(values)
                for axis, values in self.formal_coverage_minimums.items()
            },
            "missing_formal_requirements": missing_requirements,
            "formal_requirement_deficits": deficits,
            "combination_coverage": combination_coverage,
            "missing_combinations": missing_combinations,
            "external_footprint": {
                "thresholds": {
                    "l2_bytes": self.l2_bytes,
                    "last_level_cache_bytes": self.last_level_cache_bytes,
                },
                "selection": footprint_selection,
                "formal_attested": footprint_formal,
                "minimums": dict(self.footprint_minimums),
                "missing": missing_footprints,
            },
            "shape_pairs": shape_pair_coverage,
            "missing_shape_pairs": missing_shape_pairs,
            "formal_requirements_met": requirements_met,
            "formal_attested_count": sum(
                bool((results.get(entry.slot) or {}).get("formal_attested"))
                for entry in self.entries
            ),
        }


def verify_formal_entry(
    entry: CorpusEntry,
    artifact_root: str | Path | None,
    expected_architecture_hash: str | None = None,
) -> dict[str, Any]:
    """Load formal artifacts and compare them with the independent golden."""
    if artifact_root is None:
        return {"formal_attested": False, "formal_reason": "artifact_root_missing"}
    benchmark_path = (
        Path(artifact_root) / entry.config / entry.problem / "benchmark.yaml"
    )
    if not benchmark_path.is_file():
        return {"formal_attested": False, "formal_reason": "benchmark_missing"}
    benchmark = BenchmarkSpec.load(benchmark_path)
    matches = [
        workload
        for workload in benchmark.workloads
        if workload.uuid == entry.workload_uuid
    ]
    if (
        len(matches) != 1
        or matches[0].analysis is None
        or matches[0].verification is None
    ):
        return {"formal_attested": False, "formal_reason": "formal_artifacts_missing"}
    analysis = matches[0].analysis
    if (
        expected_architecture_hash is not None
        and analysis.architecture_hash != expected_architecture_hash
    ):
        return {
            "formal_attested": False,
            "formal_reason": "architecture_profile_mismatch",
            "analysis_sha256": analysis.sha256,
            "verification_sha256": matches[0].verification.sha256,
        }
    golden_ok = (
        analysis.flops == entry.golden_flops
        and analysis.fused_bytes == entry.golden_external_bytes
        and analysis.resource_work == entry.golden_resource_work
    )
    return {
        "formal_attested": golden_ok,
        "formal_reason": "verified" if golden_ok else "independent_golden_mismatch",
        "analysis_sha256": analysis.sha256,
        "verification_sha256": matches[0].verification.sha256,
    }


__all__ = [
    "OFFICIAL_DATASET_ID",
    "OFFICIAL_DATASET_REVISION",
    "CorpusEntry",
    "OfficialCorpusManifest",
    "verify_formal_entry",
]
