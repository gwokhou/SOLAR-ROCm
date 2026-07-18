# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Measured ROCm resource audits; never a theoretical SOL replacement."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from solar.analysis.resources import RESOURCE_MODEL_VERSION
from solar.benchmark.timing import AdaptiveTimer, TimingPolicy
from solar.rocm import ArchitectureProfile, RocmEnvironment


@dataclass(frozen=True)
class ResourceProbe:
    """One exact operation and its audited amount of architectural work."""

    operation: Callable[[], Any]
    work_amount: float
    resource: str
    mode: str


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: int
    architecture: str
    profile_revision: str
    resource_model_version: str
    gfx_target: str | None
    timing_profile: str
    clocks_locked: bool
    clock_levels: tuple[str, ...]
    environment: dict[str, Any]
    measured_throughput_per_second: dict[str, float]
    upper_bound_per_second: dict[str, float]
    upper_bound_ratio: dict[str, float]
    required_resource_modes: tuple[str, ...]
    measured_resource_modes: tuple[str, ...]
    exempt_resource_modes: dict[str, str]
    precision_support: dict[str, dict[str, Any]]
    timing_ms: dict[str, dict[str, Any]]
    probe_source_sha256: str | None
    calibrator_source_sha256: str
    audit_status: str
    tolerance_ratio: float
    diagnostic_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RocmCalibrator:
    """Measure every modeled AMD resource under one immutable profile."""

    def __init__(
        self,
        architecture: ArchitectureProfile | None = None,
        environment: RocmEnvironment | None = None,
    ):
        self.architecture = architecture or ArchitectureProfile.load("RX_9060_XT")
        if self.architecture.vendor.upper() != "AMD":
            raise ValueError("ROCm calibration requires an AMD profile")
        self.environment = environment

    def calibrate(
        self,
        operations: Mapping[str, ResourceProbe],
        *,
        timing_profile: str = "official",
        clocks_locked: bool = False,
        clock_levels: tuple[str, ...] = (),
        probe_source_sha256: str | None = None,
        tolerance_ratio: float = 0.05,
    ) -> CalibrationArtifact:
        """Run resource probes and reject measurements contradicting ceilings."""
        environment = self.environment or RocmEnvironment.detect()
        if (
            not environment.supported_target
            or environment.gfx_target != self.architecture.gfx_target
        ):
            raise RuntimeError(
                f"calibration requires {self.architecture.gfx_target}, "
                f"got {environment.gfx_target}"
            )
        policy = TimingPolicy.for_name(timing_profile)
        if policy.publishable and not clocks_locked:
            raise RuntimeError(
                "publishable calibration requires a STABLE_PEAK clock lock"
            )
        if tolerance_ratio < 0 or not math.isfinite(tolerance_ratio):
            raise ValueError(
                "calibration tolerance_ratio must be finite and non-negative"
            )
        required_resources = set(self.architecture.resource_limits) | {"memory"}
        present_resources = {probe.resource for probe in operations.values()}
        exempt_modes = {
            f"{resource}:{mode}": reason
            for resource, modes in self.architecture.calibration_exempt_modes.items()
            for mode, reason in modes.items()
        }
        required_modes = {
            (resource, mode)
            for resource, modes in self.architecture.resource_limits.items()
            for mode in modes
            if mode != "generic" and f"{resource}:{mode}" not in exempt_modes
        }
        present_modes = {(probe.resource, probe.mode) for probe in operations.values()}
        missing_resources = required_resources - present_resources
        missing_modes = required_modes - present_modes
        if policy.publishable and (missing_resources or missing_modes):
            raise ValueError(
                "official calibration must cover every non-exempt resource mode; "
                f"missing_resources={sorted(missing_resources)}, "
                f"missing_modes={sorted(f'{resource}:{mode}' for resource, mode in missing_modes)}"
            )

        throughput: dict[str, float] = {}
        upper_bounds: dict[str, float] = {}
        ratios: dict[str, float] = {}
        timings: dict[str, dict[str, Any]] = {}
        for name, probe in operations.items():
            if probe.work_amount <= 0 or not math.isfinite(probe.work_amount):
                raise ValueError(
                    f"probe {name} work amount must be positive and finite"
                )
            stats = AdaptiveTimer(policy).measure(probe.operation)
            timings[name] = stats.to_dict()
            measured = float(probe.work_amount) / (stats.p50_ms / 1000.0)
            upper = (
                self.architecture.memory_bandwidth_bytes_per_second
                if probe.resource == "memory"
                else self.architecture.resource_rate_for(probe.resource, probe.mode)
            )
            ratio = measured / upper
            if not math.isfinite(measured) or measured <= 0:
                raise RuntimeError(f"probe {name} produced invalid throughput")
            if ratio > 1.0 + tolerance_ratio:
                raise RuntimeError(
                    f"probe {name} measured {measured:g}, exceeding formal upper "
                    f"bound {upper:g} by more than {tolerance_ratio:.1%}"
                )
            throughput[name] = measured
            upper_bounds[name] = upper
            ratios[name] = ratio

        audit_status = (
            "verified" if policy.publishable and clocks_locked else "diagnostic"
        )
        return CalibrationArtifact(
            schema_version=3,
            architecture=self.architecture.name,
            profile_revision=self.architecture.profile_revision,
            resource_model_version=RESOURCE_MODEL_VERSION,
            gfx_target=environment.gfx_target,
            timing_profile=policy.name,
            clocks_locked=clocks_locked,
            clock_levels=clock_levels,
            environment=environment.to_dict(),
            measured_throughput_per_second=throughput,
            upper_bound_per_second=upper_bounds,
            upper_bound_ratio=ratios,
            required_resource_modes=tuple(
                sorted(f"{resource}:{mode}" for resource, mode in required_modes)
            ),
            measured_resource_modes=tuple(
                sorted(f"{resource}:{mode}" for resource, mode in present_modes)
            ),
            exempt_resource_modes=dict(sorted(exempt_modes.items())),
            precision_support={
                precision: dict(support)
                for precision, support in self.architecture.precision_support.items()
            },
            timing_ms=timings,
            probe_source_sha256=probe_source_sha256,
            calibrator_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            audit_status=audit_status,
            tolerance_ratio=tolerance_ratio,
            diagnostic_only=not (policy.publishable and clocks_locked),
        )


__all__ = ["CalibrationArtifact", "ResourceProbe", "RocmCalibrator"]
