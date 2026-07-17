# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Backend capability checks and staged entry-point loading."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from solar.benchmark.models import SolutionSpec
from solar.rocm import RocmEnvironment


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendAdapter:
    name: str
    capability: str
    native: bool = False

    def assert_available(self, environment: RocmEnvironment) -> None:
        capability = environment.capabilities.get(self.capability)
        if capability is None or not capability.available:
            detail = capability.detail if capability else "not probed"
            raise BackendUnavailable(f"{self.name} unavailable: {detail}")

    def load(self, solution: SolutionSpec, staging_root: Path) -> Callable[..., Any]:
        if self.native:
            self._compile(solution, staging_root)
            modules = sorted(staging_root.glob("*.so"))
            if not modules:
                raise RuntimeError(
                    "native compile command did not produce a top-level .so"
                )
            entry_file = modules[0]
        else:
            source_name, _ = _parse_entry_point(solution.entry_point)
            entry_file = (staging_root / source_name).resolve(strict=True)
            if staging_root not in entry_file.parents or not entry_file.is_file():
                raise RuntimeError(
                    f"solution entry point is outside staging: {source_name}"
                )
        _, function_name = _parse_entry_point(solution.entry_point)
        module_name = f"_solar_solution_{solution.raw_hash[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, entry_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load solution module: {entry_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        if not callable(function):
            raise TypeError(f"solution entry point is not callable: {function_name}")
        return function

    def _compile(self, solution: SolutionSpec, staging_root: Path) -> None:
        if not solution.compile_command:
            raise BackendUnavailable(f"{self.name} requires compile.command")
        command = [
            item.replace("{staging}", str(staging_root)).replace(
                "{gfx_target}", "gfx1200"
            )
            for item in solution.compile_command
        ]
        result = subprocess.run(
            command,
            cwd=staging_root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"native compilation failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
            )


def _parse_entry_point(entry_point: str) -> tuple[str, str]:
    if "::" in entry_point:
        return tuple(entry_point.rsplit("::", 1))  # type: ignore[return-value]
    return "", entry_point


BACKENDS = {
    "pytorch": BackendAdapter("pytorch", "pytorch_rocm"),
    "triton": BackendAdapter("triton", "triton_rocm"),
    "hip_cpp": BackendAdapter("hip_cpp", "hipcc", native=True),
    "hipblas": BackendAdapter("hipblas", "hipblas", native=True),
    "miopen": BackendAdapter("miopen", "miopen", native=True),
    "ck": BackendAdapter("ck", "ck", native=True),
    "rocwmma": BackendAdapter("rocwmma", "rocwmma", native=True),
}


def get_backend(name: str) -> BackendAdapter:
    return BACKENDS[name]


__all__ = ["BACKENDS", "BackendAdapter", "BackendUnavailable", "get_backend"]
