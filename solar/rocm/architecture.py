# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Normalized AMD ROCm architecture profiles used by SOL roofline calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

_PRECISION_ALIASES = {
    "float32": "fp32",
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float8": "fp8",
}


@dataclass(frozen=True)
class ArchitectureProfile:
    """Normalized AMD hardware limits used by SOL roofline calculations."""

    name: str
    vendor: str
    gfx_target: str
    compute_units: int
    memory_capacity_bytes: int
    memory_bandwidth_bytes_per_second: float
    l2_bytes: int
    last_level_cache_bytes: int
    peak_ops_per_second: dict[str, float] = field(default_factory=dict)
    precision_aliases: dict[str, str] = field(default_factory=dict)
    clock_hz: float | None = None
    source: str | None = None

    @classmethod
    def load(cls, value: str | Path | Mapping[str, Any]) -> "ArchitectureProfile":
        """Load a normalized AMD ROCm architecture description."""
        if isinstance(value, Mapping):
            data = dict(value)
            source = None
        else:
            path = Path(value)
            if not path.exists():
                root = Path(__file__).resolve().parents[2]
                path = root / "configs" / "arch" / f"{value}.yaml"
            if path.exists():
                data = yaml.safe_load(path.read_text()) or {}
                source = str(path)
            else:
                resource = resources.files("solar.configs.arch").joinpath(
                    f"{value}.yaml"
                )
                if not resource.is_file():
                    raise FileNotFoundError(f"Architecture profile not found: {value}")
                data = yaml.safe_load(resource.read_text()) or {}
                source = str(resource)
        if "peak_ops_per_second" not in data:
            raise ValueError(
                "ROCm architecture profiles must define normalized "
                "peak_ops_per_second fields"
            )
        profile = cls(
            name=str(data["name"]),
            vendor=str(data.get("vendor", "")),
            gfx_target=str(data.get("gfx_target", "")),
            compute_units=int(data.get("compute_units", 0)),
            memory_capacity_bytes=int(data.get("memory_capacity_bytes", 0)),
            memory_bandwidth_bytes_per_second=float(
                data["memory_bandwidth_bytes_per_second"]
            ),
            l2_bytes=int(data.get("l2_bytes", 0)),
            last_level_cache_bytes=int(data.get("last_level_cache_bytes", 0)),
            peak_ops_per_second={
                str(k).lower(): float(v) for k, v in data["peak_ops_per_second"].items()
            },
            precision_aliases={
                str(k).lower(): str(v).lower()
                for k, v in (data.get("precision_aliases") or {}).items()
            },
            clock_hz=(float(data["clock_hz"]) if data.get("clock_hz") else None),
            source=str(data.get("source") or source or "") or None,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.name:
            raise ValueError("architecture name is required")
        if self.vendor.upper() != "AMD":
            raise ValueError("SOLAR-ROCm accepts AMD architecture profiles only")
        if self.memory_bandwidth_bytes_per_second <= 0:
            raise ValueError("memory bandwidth must be positive")
        if not self.peak_ops_per_second or any(
            value <= 0 for value in self.peak_ops_per_second.values()
        ):
            raise ValueError("at least one positive peak throughput is required")

    def peak_for(self, precision: str) -> float:
        key = self.normalize_precision(precision)
        try:
            return self.peak_ops_per_second[key]
        except KeyError as exc:
            raise ValueError(
                f"Precision {precision!r} is not supported by {self.name}"
            ) from exc

    def normalize_precision(self, precision: str) -> str:
        """Resolve spelling and vendor-specific format aliases."""
        key = _PRECISION_ALIASES.get(precision.lower(), precision.lower())
        return self.precision_aliases.get(key, key)

    def theoretical_seconds(
        self, flops: float, fused_bytes: float, precision: str
    ) -> float:
        """Return max(compute time, memory time), the published SOL lower bound."""
        return max(
            float(flops) / self.peak_for(precision),
            float(fused_bytes) / self.memory_bandwidth_bytes_per_second,
        )

    def theoretical_seconds_by_precision(
        self, macs_by_precision: Mapping[str, float], fused_bytes: float
    ) -> float:
        """Return SOL using the artifact's per-operation compute precisions."""
        compute_seconds = sum(
            2.0 * float(macs) / self.peak_for(precision)
            for precision, macs in macs_by_precision.items()
        )
        memory_seconds = float(fused_bytes) / self.memory_bandwidth_bytes_per_second
        return max(compute_seconds, memory_seconds)

    @property
    def cache_flush_bytes(self) -> int:
        """Return the largest declared AMD cache that cold-cache timing must evict."""
        return max(self.l2_bytes, self.last_level_cache_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "gfx_target": self.gfx_target,
            "compute_units": self.compute_units,
            "memory_capacity_bytes": self.memory_capacity_bytes,
            "memory_bandwidth_bytes_per_second": self.memory_bandwidth_bytes_per_second,
            "l2_bytes": self.l2_bytes,
            "last_level_cache_bytes": self.last_level_cache_bytes,
            "peak_ops_per_second": dict(self.peak_ops_per_second),
            "precision_aliases": dict(self.precision_aliases),
            "clock_hz": self.clock_hz,
            "source": self.source,
        }
