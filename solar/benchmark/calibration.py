# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Measured ROCm calibration diagnostics; never a theoretical SOL replacement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from solar.benchmark.timing import AdaptiveTimer, TimingPolicy
from solar.rocm import ArchitectureProfile, RocmEnvironment


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: int
    architecture: str
    gfx_target: str | None
    timing_profile: str
    measured_throughput_per_second: dict[str, float]
    timing_ms: dict[str, dict[str, Any]]
    diagnostic_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RocmCalibrator:
    def __init__(
        self,
        architecture: ArchitectureProfile | None = None,
        environment: RocmEnvironment | None = None,
    ):
        self.architecture = architecture or ArchitectureProfile.load("RX_9060_XT")
        self.environment = environment

    def calibrate(
        self,
        operations: Mapping[str, tuple[Callable[[], Any], float]],
        *,
        timing_profile: str = "standard",
    ) -> CalibrationArtifact:
        environment = self.environment or RocmEnvironment.detect()
        if not environment.supported_target:
            raise RuntimeError(
                f"calibration requires gfx1200, got {environment.gfx_target}"
            )
        policy = TimingPolicy.for_name(timing_profile)
        throughput: dict[str, float] = {}
        timings: dict[str, dict[str, Any]] = {}
        for name, (operation, work_amount) in operations.items():
            stats = AdaptiveTimer(policy).measure(operation)
            timings[name] = stats.to_dict()
            throughput[name] = float(work_amount) / (stats.p50_ms / 1000.0)
        return CalibrationArtifact(
            schema_version=1,
            architecture=self.architecture.name,
            gfx_target=environment.gfx_target,
            timing_profile=policy.name,
            measured_throughput_per_second=throughput,
            timing_ms=timings,
        )


__all__ = ["CalibrationArtifact", "RocmCalibrator"]
