# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated YAML contracts for ROCm benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
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
class AnalysisArtifact:
    """Hash-bound, hardware-independent SOLAR analysis for one workload."""

    path: str
    sha256: str
    source_graph: str
    source_graph_sha256: str
    flops: float
    fused_bytes: float
    macs_by_precision: dict[str, float]
    lower_bound_seconds: float

    @classmethod
    def load(cls, data: Mapping[str, Any], source_root: Path) -> "AnalysisArtifact":
        path = _relative_path(str(data.get("path", "")), "analysis.path")
        source_graph = _relative_path(
            str(data.get("source_graph", "")), "analysis.source_graph"
        )
        expected_analysis_sha = cls._parse_sha256(data.get("sha256"), "analysis.sha256")
        expected_graph_sha = cls._parse_sha256(
            data.get("source_graph_sha256"), "analysis.source_graph_sha256"
        )
        analysis_path = (source_root / path).resolve()
        graph_path = (source_root / source_graph).resolve()
        for candidate, field_name in (
            (analysis_path, "analysis.path"),
            (graph_path, "analysis.source_graph"),
        ):
            if source_root not in candidate.parents or not candidate.is_file():
                raise ValueError(f"{field_name} is outside the benchmark or missing")
        actual_analysis_sha = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
        actual_graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        if actual_analysis_sha != expected_analysis_sha:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if actual_graph_sha != expected_graph_sha:
            raise ValueError(f"SHA-256 mismatch: {source_graph}")

        artifact = yaml.safe_load(analysis_path.read_text()) or {}
        if (
            not isinstance(artifact, dict)
            or int(artifact.get("schema_version", 0)) != 3
        ):
            raise ValueError(
                "benchmark analysis must use latest SOLAR analysis schema_version=3"
            )
        total = artifact.get("total") or {}
        metadata = artifact.get("metadata") or {}
        if metadata.get("source_graph_sha256") != actual_graph_sha:
            raise ValueError("analysis provenance does not match its source graph")
        if metadata.get("dtype_accounting") != "per_tensor":
            raise ValueError(
                "benchmark analysis requires explicit dtype for every tensor"
            )
        required = {"flops", "fused_bytes", "macs_by_precision", "lower_bound_seconds"}
        if not required.issubset(total):
            raise ValueError(
                "analysis total must contain flops, fused_bytes, and macs_by_precision"
            )
        macs_by_precision = {
            str(key).lower(): float(value)
            for key, value in (total.get("macs_by_precision") or {}).items()
        }
        flops = float(total["flops"])
        fused_bytes = float(total["fused_bytes"])
        lower_bound_seconds = float(total["lower_bound_seconds"])
        if (
            flops < 0
            or fused_bytes < 0
            or lower_bound_seconds < 0
            or any(value < 0 for value in macs_by_precision.values())
        ):
            raise ValueError("analysis compute and traffic totals must be non-negative")
        if abs(2.0 * sum(macs_by_precision.values()) - flops) > max(
            1e-9, flops * 1e-12
        ):
            raise ValueError("analysis flops must equal twice macs_by_precision")

        # Hashes prove identity, while deterministic re-analysis proves that
        # the checked-in totals were actually derived from the bound graph.
        from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

        with tempfile.TemporaryDirectory(prefix="solar-analysis-verify-") as output:
            derived = EinsumGraphAnalyzer().analyze_graph(
                graph_path,
                output,
                precision=str(metadata.get("precision", "fp16")),
                copy_graph=False,
                architecture=metadata.get("architecture"),
            )
        if derived is None:
            raise ValueError("failed to rederive bound SOLAR analysis")
        derived_total = derived.get("total") or {}
        derived_metadata = derived.get("metadata") or {}
        if derived_metadata.get("source_graph_sha256") != actual_graph_sha:
            raise ValueError("analysis rederivation did not use the bound source graph")
        derived_identity = {
            "flops": float(derived_total.get("flops", -1)),
            "fused_bytes": float(derived_total.get("fused_bytes", -1)),
            "macs_by_precision": {
                str(key).lower(): float(value)
                for key, value in (derived_total.get("macs_by_precision") or {}).items()
            },
            "lower_bound_seconds": float(derived_total.get("lower_bound_seconds", -1)),
        }
        artifact_identity = {
            "flops": flops,
            "fused_bytes": fused_bytes,
            "macs_by_precision": macs_by_precision,
            "lower_bound_seconds": lower_bound_seconds,
        }
        if derived_identity != artifact_identity:
            raise ValueError("analysis totals drifted from the bound source graph")
        return cls(
            path=path,
            sha256=expected_analysis_sha,
            source_graph=source_graph,
            source_graph_sha256=expected_graph_sha,
            flops=flops,
            fused_bytes=fused_bytes,
            macs_by_precision=macs_by_precision,
            lower_bound_seconds=lower_bound_seconds,
        )

    @staticmethod
    def _parse_sha256(value: Any, field_name: str) -> str:
        digest = str(value or "").lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        return digest


@dataclass(frozen=True)
class VerificationArtifact:
    """Replay-verified in-toto statement binding reference to einsum graph."""

    path: str
    sha256: str
    reference_sha256: str
    source_graph_sha256: str

    @classmethod
    def load(
        cls,
        data: Mapping[str, Any],
        source_root: Path,
        *,
        reference_path: Path,
        graph_path: Path,
        workload_name: str,
        workload_parameters: Mapping[str, Any],
        atol: float,
        rtol: float,
    ) -> "VerificationArtifact":
        path = _relative_path(str(data.get("path", "")), "verification.path")
        digest = AnalysisArtifact._parse_sha256(
            data.get("sha256"), "verification.sha256"
        )
        artifact_path = (source_root / path).resolve()
        if source_root not in artifact_path.parents or not artifact_path.is_file():
            raise ValueError("verification.path is outside the benchmark or missing")
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch: {path}")
        artifact = yaml.safe_load(artifact_path.read_text()) or {}
        if not isinstance(artifact, Mapping):
            raise ValueError("verification artifact must be a mapping")

        from solar.verification import replay_verification_artifact

        replay_verification_artifact(
            artifact,
            reference_path=reference_path,
            graph_path=graph_path,
            workload_name=workload_name,
            workload_parameters=workload_parameters,
            atol=atol,
            rtol=rtol,
        )
        return cls(
            path=path,
            sha256=digest,
            reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            source_graph_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        )


@dataclass(frozen=True)
class CompatibilityArtifact:
    """Hash-bound evidence explaining why a workload was not built."""

    path: str
    sha256: str
    status: str
    reason_code: str

    @classmethod
    def load(
        cls, data: Mapping[str, Any], source_root: Path
    ) -> "CompatibilityArtifact":
        path = _relative_path(str(data.get("path", "")), "compatibility.path")
        digest = AnalysisArtifact._parse_sha256(
            data.get("sha256"), "compatibility.sha256"
        )
        artifact_path = (source_root / path).resolve()
        if source_root not in artifact_path.parents or not artifact_path.is_file():
            raise ValueError("compatibility.path is outside the benchmark or missing")
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch: {path}")
        artifact = yaml.safe_load(artifact_path.read_text()) or {}
        status = str(artifact.get("status", ""))
        reason_code = str(artifact.get("reason_code", ""))
        if status not in {
            "compatible",
            "incompatible",
            "execution_failed",
            "not_checked",
        }:
            raise ValueError("compatibility artifact has an invalid status")
        if not reason_code:
            raise ValueError("compatibility artifact requires a reason_code")
        return cls(path=path, sha256=digest, status=status, reason_code=reason_code)


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    status: str = "compatible"
    uuid: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    analysis: AnalysisArtifact | None = None
    verification: VerificationArtifact | None = None
    compatibility: CompatibilityArtifact | None = None
    atol: float = 1e-5
    rtol: float = 1e-5

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        index: int,
        source_root: Path,
        *,
        reference_path: Path,
        atol: float,
        rtol: float,
    ) -> "WorkloadSpec":
        if "flops" in data or "fused_bytes" in data:
            raise ValueError(
                "manual workload flops/fused_bytes are unsupported; reference a "
                "hash-bound SOLAR analysis artifact"
            )
        name = str(data.get("name", f"workload_{index}"))
        status = str(data.get("status", ""))
        parameters = dict(data.get("parameters") or {})
        local_tolerance = data.get("tolerance") or {}
        workload_atol = float(
            local_tolerance.get("max_atol", local_tolerance.get("atol", atol))
        )
        workload_rtol = float(
            local_tolerance.get("max_rtol", local_tolerance.get("rtol", rtol))
        )
        if status != "compatible":
            if status not in {"incompatible", "execution_failed", "not_checked"}:
                raise ValueError(f"invalid workload status: {status}")
            if data.get("analysis") is not None or data.get("verification") is not None:
                raise ValueError(
                    "non-compatible workloads cannot contain SOL artifacts"
                )
            compatibility_data = data.get("compatibility")
            if not isinstance(compatibility_data, Mapping):
                raise ValueError(
                    "non-compatible workloads require compatibility evidence"
                )
            compatibility = CompatibilityArtifact.load(compatibility_data, source_root)
            if compatibility.status != status:
                raise ValueError(
                    "workload status disagrees with compatibility artifact"
                )
            return cls(
                name=name,
                status=status,
                uuid=str(data.get("uuid", "")) or None,
                parameters=parameters,
                compatibility=compatibility,
                atol=workload_atol,
                rtol=workload_rtol,
            )

        analysis_data = data.get("analysis")
        if not isinstance(analysis_data, Mapping):
            raise ValueError("compatible workloads require an analysis artifact")
        analysis = AnalysisArtifact.load(analysis_data, source_root)
        verification_data = data.get("verification")
        verification = None
        if verification_data is not None:
            if not isinstance(verification_data, Mapping):
                raise ValueError("workload verification must be a mapping")
            verification = VerificationArtifact.load(
                verification_data,
                source_root,
                reference_path=reference_path,
                graph_path=(source_root / analysis.source_graph).resolve(),
                workload_name=name,
                workload_parameters=parameters,
                atol=workload_atol,
                rtol=workload_rtol,
            )
        else:
            raise ValueError(
                "compatible workloads require a replayable verification artifact"
            )
        compatibility = None
        compatibility_data = data.get("compatibility")
        if compatibility_data is not None:
            if not isinstance(compatibility_data, Mapping):
                raise ValueError("workload compatibility must be a mapping")
            compatibility = CompatibilityArtifact.load(compatibility_data, source_root)
            if compatibility.status != "compatible":
                raise ValueError("compatible workload has non-compatible evidence")
        return cls(
            name=name,
            status=status,
            uuid=str(data.get("uuid", "")) or None,
            parameters=parameters,
            analysis=analysis,
            verification=verification,
            compatibility=compatibility,
            atol=workload_atol,
            rtol=workload_rtol,
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    source_root: Path
    reference_source: str
    reference_sha256: str
    reference_entry_point: str
    input_factory: str
    workloads: tuple[WorkloadSpec, ...]
    atol: float = 1e-5
    rtol: float = 1e-5
    cache_policy: str = "cold"
    precision: str = "fp16"
    schema_version: int = 3
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
        reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        identity = dict(data)
        identity["reference_sha256"] = reference_sha256
        tolerance = data.get("tolerance") or {}
        schema_version = int(data.get("schema_version", 0))
        atol = float(tolerance.get("atol", 1e-5))
        rtol = float(tolerance.get("rtol", 1e-5))
        workloads = tuple(
            WorkloadSpec.from_dict(
                item,
                i,
                resolved.parent,
                reference_path=reference_path,
                atol=atol,
                rtol=rtol,
            )
            for i, item in enumerate(data.get("workloads") or [])
        )
        result = cls(
            name=str(data.get("name", "")),
            source_root=resolved.parent,
            reference_source=reference_source,
            reference_sha256=reference_sha256,
            reference_entry_point=str(reference.get("entry_point", "run")),
            input_factory=str(reference.get("input_factory", "get_inputs")),
            workloads=workloads,
            atol=atol,
            rtol=rtol,
            cache_policy=str(data.get("cache_policy", "cold")),
            precision=str(data.get("precision", "fp16")),
            schema_version=schema_version,
            raw_hash=canonical_hash(identity),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.name or not self.workloads:
            raise ValueError("benchmark name and at least one workload are required")
        if self.schema_version != 3:
            raise ValueError(
                "benchmark must use latest schema_version=3 with a verified "
                "source-to-SOL chain"
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
        if not self.gfx_targets or any(
            not target.startswith("gfx") for target in self.gfx_targets
        ):
            raise ValueError("ROCm solutions require one or more gfx targets")
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
    architecture_hash: str
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
            architecture_hash=str(data.get("architecture_hash", "")),
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
        architecture_hash: str,
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
            "architecture_hash": (self.architecture_hash, architecture_hash),
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
    "AnalysisArtifact",
    "BaselineRegistry",
    "BenchmarkSpec",
    "CompatibilityArtifact",
    "SolutionSpec",
    "SourceFile",
    "VerificationArtifact",
    "WorkloadSpec",
    "canonical_hash",
]
