# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# The generated Timeloop input is intentionally kept adjacent to the runner.
# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

OROJENESIS_COMMIT = "97d52178bf9a9c209bf79be96b87c164bcd35625"
_TOKEN = re.compile(r"[A-Za-z][0-9]*")


class OrojenesisError(RuntimeError):
    """The official external solver could not produce an auditable bound."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OrojenesisRunner:
    """Run and parse Timeloop's OAVES/Orojenesis mode at a pinned commit."""

    def __init__(self, home: str | Path | None = None, *, timeout_seconds: int = 7200):
        configured = home or os.environ.get("SOLAR_OROJENESIS_HOME")
        if not configured:
            raise OrojenesisError(
                "Orojenesis is required; set --orojenesis-home or SOLAR_OROJENESIS_HOME"
            )
        self.home = Path(configured).resolve()
        self.timeout_seconds = int(timeout_seconds)
        self.mapper = self.home / "bin" / "timeloop-mapper"
        self._validate_toolchain()

    def _validate_toolchain(self) -> None:
        if not self.mapper.is_file() or not os.access(self.mapper, os.X_OK):
            raise OrojenesisError(f"missing executable: {self.mapper}")
        try:
            head = subprocess.run(
                ["git", "-C", str(self.home), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError("cannot verify Orojenesis git revision") from exc
        if head != OROJENESIS_COMMIT:
            raise OrojenesisError(
                f"Orojenesis revision mismatch: expected {OROJENESIS_COMMIT}, got {head}"
            )

    @staticmethod
    def problem_for_layer(layer: Mapping[str, Any]) -> dict[str, Any]:
        semantic = layer.get("semantic_op") or {}
        if semantic.get("kind") != "einsum":
            raise OrojenesisError("Orojenesis accepts exact einsum layers only")
        equation = str(semantic.get("equation", ""))
        lhs, rhs = equation.split("->", 1)
        operands = lhs.split(",")
        shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
        output_shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
        if len(operands) != len(shapes) or len(output_shapes) != 1:
            raise OrojenesisError("einsum operand arity does not match tensor metadata")
        dimension_sizes: dict[str, int] = {}
        data_spaces: list[dict[str, Any]] = []
        for index, (operand, shape) in enumerate(zip(operands, shapes)):
            tokens = _TOKEN.findall(operand)
            if len(tokens) != len(shape):
                raise OrojenesisError("einsum rank does not match input shape")
            for token, size in zip(tokens, shape):
                if token in dimension_sizes and dimension_sizes[token] != int(size):
                    raise OrojenesisError(f"inconsistent dimension {token}")
                dimension_sizes[token] = int(size)
            data_spaces.append(
                {"name": f"Input{index}", "projection": [[[token]] for token in tokens]}
            )
        output_tokens = _TOKEN.findall(rhs)
        if len(output_tokens) != len(output_shapes[0]):
            raise OrojenesisError("einsum rank does not match output shape")
        for token, size in zip(output_tokens, output_shapes[0]):
            if token in dimension_sizes and dimension_sizes[token] != int(size):
                raise OrojenesisError(f"inconsistent dimension {token}")
            dimension_sizes[token] = int(size)
        data_spaces.append(
            {
                "name": "Output",
                "projection": [[[token]] for token in output_tokens],
                "read-write": True,
            }
        )
        dimensions = list(dimension_sizes)
        return {
            "problem": {
                "instance": dimension_sizes,
                "shape": {"data-spaces": data_spaces, "dimensions": dimensions},
            }
        }

    @staticmethod
    def architecture(word_bits: int) -> dict[str, Any]:
        return {
            "architecture": {
                "version": 0.2,
                "subtree": [
                    {
                        "name": "System",
                        "local": [
                            {
                                "name": "MainMemory",
                                "class": "DRAM",
                                "attributes": {"width": 64, "word-bits": word_bits},
                            }
                        ],
                        "subtree": [
                            {
                                "name": "PE",
                                "local": [
                                    {
                                        "name": "Buffer",
                                        "class": "regfile",
                                        "attributes": {
                                            "sizeKB": 2147483648,
                                            "instances": 1,
                                            "word-bits": word_bits,
                                        },
                                    },
                                    {
                                        "name": "MACC",
                                        "class": "intmac",
                                        "attributes": {"datawidth": word_bits},
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        }

    @staticmethod
    def mapper_config(dimensions: list[str], spaces: list[str]) -> dict[str, Any]:
        return {
            "mapper": {
                "optimization-metrics": ["last-level-accesses"],
                "algorithm": "linear-pruned",
                "victory-condition": 0,
                "timeout": 0,
                "log-oaves": True,
                "num-threads": 8,
                "log-oaves-mappings": False,
            },
            "mapspace_constraints": [
                {
                    "target": "Buffer",
                    "type": "temporal",
                    "permutation": "".join(dimensions),
                },
                {"target": "MainMemory", "type": "temporal"},
                {
                    "target": "MainMemory",
                    "type": "datatype",
                    "keep": spaces,
                    "bypass": [],
                },
            ],
        }

    @staticmethod
    def parse_curve(path: str | Path, *, word_bytes: int) -> list[dict[str, Any]]:
        source = Path(path)
        if not source.is_file():
            raise OrojenesisError(f"missing OAVES output: {source}")
        best: dict[int, dict[str, Any]] = {}
        with source.open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                try:
                    buffer_bytes = int(float(row[0]))
                    intensity = float(row[1])
                    accesses = float(row[2])
                except ValueError:
                    continue
                point = {
                    "buffer_bytes": buffer_bytes,
                    "operational_intensity": intensity,
                    "dram_accesses_words": accesses,
                    "dram_bytes": accesses * word_bytes,
                }
                previous = best.get(buffer_bytes)
                if previous is None or point["dram_bytes"] < previous["dram_bytes"]:
                    best[buffer_bytes] = point
        if not best:
            raise OrojenesisError("OAVES output contains no valid curve points")
        pareto: list[dict[str, Any]] = []
        best_traffic = float("inf")
        for point in sorted(best.values(), key=lambda item: item["buffer_bytes"]):
            if point["dram_bytes"] < best_traffic:
                pareto.append(point)
                best_traffic = float(point["dram_bytes"])
        return pareto

    def run_layer(
        self, layer: Mapping[str, Any], output_dir: str | Path, *, word_bits: int
    ) -> dict[str, Any]:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        problem = self.problem_for_layer(layer)
        dimensions = list(problem["problem"]["shape"]["dimensions"])
        spaces = [item["name"] for item in problem["problem"]["shape"]["data-spaces"]]
        inputs = {
            "problem.yaml": problem,
            "architecture.yaml": self.architecture(word_bits),
            "mapper.yaml": self.mapper_config(dimensions, spaces),
        }
        paths: dict[str, Path] = {}
        for name, data in inputs.items():
            path = output / name
            path.write_text(yaml.safe_dump(data, sort_keys=False))
            paths[name] = path
        try:
            completed = subprocess.run(
                [
                    str(self.mapper),
                    str(paths["architecture.yaml"]),
                    str(paths["problem.yaml"]),
                    str(paths["mapper.yaml"]),
                    "-o",
                    str(output),
                ],
                cwd=output,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError("Orojenesis execution failed") from exc
        (output / "stdout.log").write_text(completed.stdout)
        (output / "stderr.log").write_text(completed.stderr)
        if completed.returncode != 0:
            raise OrojenesisError(
                f"Orojenesis exited with status {completed.returncode}"
            )
        raw = output / "timeloop-mapper.oaves.csv"
        curve = self.parse_curve(raw, word_bytes=max(1, word_bits // 8))
        return {
            "solver": "NVlabs/timeloop oaves_keep_max",
            "commit": OROJENESIS_COMMIT,
            "word_bits": int(word_bits),
            "curve": curve,
            "evidence_files": {
                **{
                    name: {"path": name, "sha256": _sha256(path)}
                    for name, path in paths.items()
                },
                "curve": {
                    "path": raw.name,
                    "sha256": _sha256(raw),
                },
            },
        }


def select_capacity_point(
    curve: Sequence[Mapping[str, Any]], capacity_bytes: int
) -> dict[str, Any] | None:
    candidates = [
        point for point in curve if int(point["buffer_bytes"]) <= capacity_bytes
    ]
    if not candidates:
        return None
    return dict(min(candidates, key=lambda point: float(point["dram_bytes"])))


__all__ = [
    "OROJENESIS_COMMIT",
    "OrojenesisError",
    "OrojenesisRunner",
    "select_capacity_point",
]
