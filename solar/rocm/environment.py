# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Read-only ROCm environment discovery."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Capability:
    available: bool
    detail: str = ""


@dataclass(frozen=True)
class RocmEnvironment:
    """A serializable snapshot of the local ROCm runtime."""

    rocm_version: str | None
    torch_version: str | None
    hip_version: str | None
    device_name: str | None
    gfx_target: str | None
    pytorch_compute_units: int | None
    normalized_compute_units: int | None
    total_memory_bytes: int | None
    capabilities: dict[str, Capability] = field(default_factory=dict)
    os: str = field(default_factory=platform.platform)

    @classmethod
    def detect(cls) -> "RocmEnvironment":
        hipcc = shutil.which("hipcc")
        rocm_roots = [Path("/opt/rocm"), Path("/usr"), Path("/usr/local")]
        if hipcc:
            rocm_roots.insert(0, Path(hipcc).resolve().parent.parent)

        def find_tool(name: str) -> str | None:
            located = shutil.which(name)
            if located:
                return located
            for root in rocm_roots:
                candidate = root / "bin" / name
                if candidate.is_file():
                    return str(candidate)
            return None

        tools = {
            "hipcc": hipcc,
            "rocprofv3": find_tool("rocprofv3"),
            "amd_smi": find_tool("amd-smi"),
        }
        caps = {
            name: Capability(path is not None, path or "not found")
            for name, path in tools.items()
        }
        for name, header in {
            "hipblas": "hipblas/hipblas.h",
            "miopen": "miopen/miopen.h",
            "ck": "ck/ck.hpp",
            "rocwmma": "rocwmma/rocwmma.hpp",
        }.items():
            found_path = next(
                (
                    root / "include" / header
                    for root in rocm_roots
                    if (root / "include" / header).is_file()
                ),
                None,
            )
            caps[name] = Capability(
                found_path is not None,
                str(found_path) if found_path else f"missing {header} under ROCm roots",
            )

        rocm_version = cls._rocm_version()
        values: dict[str, Any] = {
            "torch_version": None,
            "hip_version": None,
            "device_name": None,
            "gfx_target": None,
            "pytorch_compute_units": None,
            "normalized_compute_units": None,
            "total_memory_bytes": None,
        }
        try:
            import torch

            values["torch_version"] = torch.__version__
            values["hip_version"] = getattr(torch.version, "hip", None)
            available = bool(torch.cuda.is_available() and values["hip_version"])
            caps["pytorch_rocm"] = Capability(
                available,
                "HIP-backed torch.cuda" if available else "ROCm GPU unavailable",
            )
            if available:
                props = torch.cuda.get_device_properties(0)
                values["device_name"] = props.name
                values["gfx_target"] = (
                    getattr(props, "gcnArchName", "").split(":", 1)[0] or None
                )
                values["pytorch_compute_units"] = int(props.multi_processor_count)
                values["total_memory_bytes"] = int(props.total_memory)
                # RDNA exposes WGPs through PyTorch, while published specs use CUs.
                values["normalized_compute_units"] = (
                    values["pytorch_compute_units"] * 2
                    if str(values["gfx_target"]).startswith("gfx12")
                    else values["pytorch_compute_units"]
                )
                try:
                    import triton  # noqa: F401

                    caps["triton_rocm"] = Capability(True, "importable")
                except Exception as exc:  # pragma: no cover - environment dependent
                    caps["triton_rocm"] = Capability(False, str(exc))
        except Exception as exc:  # pragma: no cover - environment dependent
            caps["pytorch_rocm"] = Capability(False, str(exc))
        return cls(rocm_version=rocm_version, capabilities=caps, **values)

    @staticmethod
    def _rocm_version() -> str | None:
        try:
            result = subprocess.run(
                ["hipcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in (result.stdout + result.stderr).splitlines():
                if "HIP version" in line:
                    return line.split(":", 1)[-1].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    @property
    def supported_target(self) -> bool:
        """Whether a usable AMD GCN target was discovered."""
        capability = self.capabilities.get("pytorch_rocm")
        return bool(
            self.gfx_target
            and self.gfx_target.startswith("gfx")
            and (capability is None or capability.available)
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))
