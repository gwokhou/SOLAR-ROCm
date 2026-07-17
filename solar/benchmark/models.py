# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated YAML contracts for ROCm benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

SUPPORTED_BACKENDS = frozenset(
    {"pytorch", "triton", "hip_cpp", "hipblas", "miopen", "ck", "rocwmma"}
)
SUPPORTED_CACHE_POLICIES = frozenset({"cold", "application"})


def _load_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    data = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML document must be a mapping: {resolved}")
    return resolved, data


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"{field_name} must be a non-empty relative path")
    return value


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    flops: float = 0.0
    fused_bytes: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int) -> "WorkloadSpec":
        result = cls(
            name=str(data.get("name", f"workload_{index}")),
            parameters=dict(data.get("parameters") or {}),
            flops=float(data.get("flops", 0)),
            fused_bytes=float(data.get("fused_bytes", 0)),
        )
        if result.flops < 0 or result.fused_bytes < 0:
            raise ValueError("workload flops and fused_bytes must be non-negative")
        return result


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    source_root: Path
    reference_source: str
    reference_entry_point: str
    input_factory: str
    workloads: tuple[WorkloadSpec, ...]
    atol: float = 1e-5
    rtol: float = 1e-5
    cache_policy: str = "cold"
    precision: str = "fp16"
    schema_version: int = 1
    raw_hash: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSpec":
        resolved, data = _load_yaml(path)
        reference = data.get("reference") or {}
        reference_source = _relative_path(
            str(reference.get("source", "")), "reference.source"
        )
        reference_path = (resolved.parent / reference_source).resolve()
        if (
            resolved.parent not in reference_path.parents
            or not reference_path.is_file()
        ):
            raise ValueError(
                f"reference source is outside the benchmark or missing: {reference_path}"
            )
        identity = dict(data)
        identity["reference_sha256"] = hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest()
        tolerance = data.get("tolerance") or {}
        workloads = tuple(
            WorkloadSpec.from_dict(item, i)
            for i, item in enumerate(data.get("workloads") or [])
        )
        result = cls(
            name=str(data.get("name", "")),
            source_root=resolved.parent,
            reference_source=reference_source,
            reference_entry_point=str(reference.get("entry_point", "run")),
            input_factory=str(reference.get("input_factory", "get_inputs")),
            workloads=workloads,
            atol=float(tolerance.get("atol", 1e-5)),
            rtol=float(tolerance.get("rtol", 1e-5)),
            cache_policy=str(data.get("cache_policy", "cold")),
            precision=str(data.get("precision", "fp16")),
            schema_version=int(data.get("schema_version", 1)),
            raw_hash=canonical_hash(identity),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.name or not self.workloads:
            raise ValueError("benchmark name and at least one workload are required")
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported benchmark schema version: {self.schema_version}"
            )
        if self.cache_policy not in SUPPORTED_CACHE_POLICIES:
            raise ValueError(f"unsupported cache policy: {self.cache_policy}")
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("tolerances must be non-negative")
        source = (self.source_root / self.reference_source).resolve()
        if self.source_root not in source.parents or not source.is_file():
            raise ValueError(
                f"reference source is outside the benchmark or missing: {source}"
            )


@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceFile":
        path = _relative_path(str(data.get("path", "")), "sources.path")
        digest = str(data.get("sha256", "")).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA-256 for {path}")
        return cls(path, digest)


@dataclass(frozen=True)
class SolutionSpec:
    name: str
    source_root: Path
    backend: str
    gfx_targets: tuple[str, ...]
    sources: tuple[SourceFile, ...]
    entry_point: str
    compile_command: tuple[str, ...] | None = None
    schema_version: int = 1
    raw_hash: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "SolutionSpec":
        resolved, data = _load_yaml(path)
        compile_data = data.get("compile") or {}
        command = compile_data.get("command")
        result = cls(
            name=str(data.get("name", "")),
            source_root=resolved.parent,
            backend=str(data.get("backend", "")),
            gfx_targets=tuple(str(item) for item in data.get("gfx_targets") or []),
            sources=tuple(
                SourceFile.from_dict(item) for item in data.get("sources") or []
            ),
            entry_point=str(data.get("entry_point", "run")),
            compile_command=tuple(str(item) for item in command) if command else None,
            schema_version=int(data.get("schema_version", 1)),
            raw_hash=canonical_hash(data),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.name or self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"solution name and supported backend are required: {self.backend}"
            )
        if self.schema_version != 1 or not self.sources:
            raise ValueError(
                "solution schema_version=1 and at least one source are required"
            )
        if "gfx1200" not in self.gfx_targets:
            raise ValueError("the ROCm v1 evaluator only supports gfx1200 solutions")
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("solution source paths must be unique")

        if self.backend in {"pytorch", "triton"}:
            if "::" not in self.entry_point:
                raise ValueError(
                    "Python solution entry_point must use file.py::function"
                )
            entry_source, function_name = self.entry_point.rsplit("::", 1)
            entry_source = _relative_path(entry_source, "entry_point source")
            if entry_source not in paths:
                raise ValueError(
                    "entry_point source must be listed in solution sources"
                )
            if not function_name.isidentifier():
                raise ValueError("entry_point function must be a Python identifier")
        else:
            _, function_name = (
                self.entry_point.rsplit("::", 1)
                if "::" in self.entry_point
                else ("", self.entry_point)
            )
            if not function_name.isidentifier():
                raise ValueError("entry_point function must be a Python identifier")

        for source in self.sources:
            actual = (self.source_root / source.path).resolve()
            if self.source_root not in actual.parents or not actual.is_file():
                raise ValueError(
                    f"solution source is outside the package or missing: {actual}"
                )


@dataclass(frozen=True)
class BaselineEntry:
    workload: str
    latency_ms: float


@dataclass(frozen=True)
class BaselineRegistry:
    name: str
    benchmark_hash: str
    solution_hash: str
    gfx_target: str
    timing_profile: str
    cache_policy: str
    environment_hash: str
    clocks_locked: bool
    workloads: dict[str, float]
    schema_version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "BaselineRegistry":
        _, data = _load_yaml(path)
        result = cls(
            name=str(data.get("name", "")),
            benchmark_hash=str(data.get("benchmark_hash", "")),
            solution_hash=str(data.get("solution_hash", "")),
            gfx_target=str(data.get("gfx_target", "")),
            timing_profile=str(data.get("timing_profile", "")),
            cache_policy=str(data.get("cache_policy", "")),
            environment_hash=str(data.get("environment_hash", "")),
            clocks_locked=bool(data.get("clocks_locked", False)),
            workloads={
                str(k): float(v) for k, v in (data.get("workloads") or {}).items()
            },
            schema_version=int(data.get("schema_version", 1)),
        )
        if result.schema_version != 1 or not result.workloads:
            raise ValueError("baseline schema_version=1 and workloads are required")
        return result

    def assert_compatible(
        self,
        benchmark: BenchmarkSpec,
        environment_hash: str,
        timing_profile: str,
        gfx_target: str,
        clocks_locked: bool,
    ) -> None:
        expected = {
            "benchmark_hash": (self.benchmark_hash, benchmark.raw_hash),
            "gfx_target": (self.gfx_target, gfx_target),
            "timing_profile": (self.timing_profile, timing_profile),
            "cache_policy": (self.cache_policy, benchmark.cache_policy),
            "environment_hash": (self.environment_hash, environment_hash),
            "clocks_locked": (self.clocks_locked, clocks_locked),
        }
        mismatch = [
            name for name, (actual, wanted) in expected.items() if actual != wanted
        ]
        if mismatch:
            raise ValueError("baseline mismatch: " + ", ".join(mismatch))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "BaselineRegistry",
    "BenchmarkSpec",
    "SolutionSpec",
    "SourceFile",
    "WorkloadSpec",
    "canonical_hash",
]
