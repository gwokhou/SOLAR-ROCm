# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hash-verified source staging and conservative reward-hack scanning."""

from __future__ import annotations

import ast
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from solar.benchmark.models import SolutionSpec


class SourceIntegrityError(ValueError):
    pass


class RewardHackDetected(ValueError):
    pass


_BANNED_IMPORTS = {
    "builtins",
    "ctypes",
    "concurrent",
    "http",
    "importlib",
    "inspect",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "resource",
    "socket",
    "solar",
    "subprocess",
    "threading",
    "_thread",
    "time",
    "urllib",
}
_BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "delattr",
    "open",
    "setattr",
    "os.popen",
    "os.system",
    "torch.cuda.Event",
    "torch.load",
}


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def scan_python_source(path: Path) -> None:
    """Reject network/process/file access in untrusted Python solutions."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        raise RewardHackDetected(f"invalid Python source: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _call_name(node).startswith(
            ("sys.modules", "sys._getframe", "sys.settrace", "sys.setprofile")
        ):
            raise RewardHackDetected(f"banned interpreter access: {_call_name(node)}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in _BANNED_IMPORTS:
                    raise RewardHackDetected(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in _BANNED_IMPORTS:
                raise RewardHackDetected(f"banned import: {node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _BANNED_CALLS:
                raise RewardHackDetected(f"banned call: {name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and _call_name(target).startswith(
                    ("torch.", "builtins.")
                ):
                    raise RewardHackDetected(
                        f"banned monkey patch: {_call_name(target)}"
                    )


@dataclass
class StagedSolution:
    spec: SolutionSpec
    root: Path
    _temporary: tempfile.TemporaryDirectory[str]

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> "StagedSolution":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def stage_solution(solution: SolutionSpec, *, scan: bool = True) -> StagedSolution:
    temporary = tempfile.TemporaryDirectory(prefix="solar-rocm-eval-")
    root = Path(temporary.name).resolve()
    try:
        for source in solution.sources:
            original = (solution.source_root / source.path).resolve(strict=True)
            if solution.source_root not in original.parents or original.is_symlink():
                raise SourceIntegrityError(f"unsafe source path: {source.path}")
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            if digest != source.sha256:
                raise SourceIntegrityError(f"SHA-256 mismatch: {source.path}")
            destination = root / source.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination, follow_symlinks=False)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != source.sha256:
                raise SourceIntegrityError(f"staged SHA-256 mismatch: {source.path}")
            if scan and destination.suffix == ".py":
                scan_python_source(destination)
        return StagedSolution(solution, root, temporary)
    except Exception:
        temporary.cleanup()
        raise


__all__ = [
    "RewardHackDetected",
    "SourceIntegrityError",
    "StagedSolution",
    "scan_python_source",
    "stage_solution",
]
