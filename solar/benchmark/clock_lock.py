# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Verified AMD-SMI STABLE_PEAK lease."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any


def _performance_levels(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            # AMD-SMI has used both ``performance_level`` and ``perf_level``
            # across releases.  Accept both spellings while keeping the
            # recursive parser tolerant of the nested per-device JSON shape.
            normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized_key in {"perf_level", "performance_level"} or (
                "level" in normalized_key
                and ("performance" in normalized_key or "perf" in normalized_key)
            ):
                found.append(str(item))
            found.extend(_performance_levels(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_performance_levels(item))
    return found


def query_clock_level() -> tuple[str, ...]:
    executable = _amd_smi_executable()
    if not executable:
        return ()
    try:
        result = subprocess.run(
            [executable, "metric", "-l", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return tuple(_performance_levels(json.loads(result.stdout)))
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return ()


def _amd_smi_executable() -> str | None:
    located = shutil.which("amd-smi")
    if located:
        return located
    from pathlib import Path

    for candidate in (
        "/opt/rocm/bin/amd-smi",
        "/usr/bin/amd-smi",
        "/usr/local/bin/amd-smi",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _is_level(levels: tuple[str, ...], expected: str) -> bool:
    return bool(levels) and all(expected.upper() in item.upper() for item in levels)


@dataclass
class ClockLockLease:
    locked: bool
    acquired: bool

    def release(self) -> None:
        if not self.acquired:
            return
        executable = _amd_smi_executable() or "amd-smi"
        subprocess.run(
            ["sudo", "-n", executable, "set", "-l", "AUTO"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        if not _is_level(query_clock_level(), "AUTO"):
            raise RuntimeError("failed to verify AMD GPU clock restoration to AUTO")
        self.acquired = False
        self.locked = False

    def __enter__(self) -> "ClockLockLease":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


def acquire_clock_lock() -> ClockLockLease:
    levels = query_clock_level()
    if _is_level(levels, "STABLE_PEAK"):
        return ClockLockLease(True, False)
    if not _is_level(levels, "AUTO"):
        return ClockLockLease(False, False)
    executable = _amd_smi_executable() or "amd-smi"
    try:
        result = subprocess.run(
            ["sudo", "-n", executable, "set", "-l", "STABLE_PEAK"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ClockLockLease(False, False)
    if result.returncode:
        return ClockLockLease(False, False)
    time.sleep(3)
    if not _is_level(query_clock_level(), "STABLE_PEAK"):
        subprocess.run(
            ["sudo", "-n", executable, "set", "-l", "AUTO"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return ClockLockLease(False, False)
    return ClockLockLease(True, True)


__all__ = ["ClockLockLease", "acquire_clock_lock", "query_clock_level"]
