# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# The generated Timeloop input is intentionally kept adjacent to the runner.
# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals,too-many-statements,too-many-branches,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import string
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

OROJENESIS_COMMIT = "97d52178bf9a9c209bf79be96b87c164bcd35625"
OROJENESIS_REPOSITORY = "https://github.com/NVlabs/timeloop.git"
OROJENESIS_TREE_OID = "05b05ec5a2a2979b1fe92046b937556d9ad99847"
OROJENESIS_BUILDER_IMAGE = (
    "ubuntu:24.04@sha256:"
    "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"
)
OROJENESIS_COMPILER_WRAPPER_SHA256 = (
    "a68dd5baf6ca67674b7d94c2413d1fe34c06bddafb84d41a9fb18e9699abc75e"
)
OROJENESIS_PROVENANCE_FILENAME = "orojenesis-provenance.json"
OROJENESIS_IDENTITY_SCHEMA_VERSION = 1
MULTI_EINSUM_SOLVER = "NVlabs/Orojenesis tiled-fusion"
MULTI_EINSUM_COMPOSITION = "linear_matmul_compatible_tiles_sum_capacity_v1"
MULTI_EINSUM_LAYOUT_COMPOSITION = "linear_matmul_axis_map_tile_shape_v2"
MULTI_EINSUM_BATCH_COMPOSITION = "broadcast_batch_linear_tile_shape_v1"
MULTI_EINSUM_FANOUT_COMPOSITION = "matmul_fanout_tree_tile_shape_v1"
_TOKEN = re.compile(r"[A-Za-z][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


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
        self.toolchain_identity = self._validate_toolchain()

    def _validate_toolchain(self) -> dict[str, Any]:
        if not self.mapper.is_file() or not os.access(self.mapper, os.X_OK):
            raise OrojenesisError(f"missing executable: {self.mapper}")
        binary_sha256 = _sha256(self.mapper)
        provenance_path = self.home / OROJENESIS_PROVENANCE_FILENAME
        if provenance_path.is_file():
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise OrojenesisError("cannot parse Orojenesis provenance") from exc
            if not isinstance(provenance, dict):
                raise OrojenesisError("Orojenesis provenance must be an object")
            if (
                int(provenance.get("schema_version", 0))
                != OROJENESIS_IDENTITY_SCHEMA_VERSION
            ):
                raise OrojenesisError("unsupported Orojenesis provenance schema")
            source = provenance.get("source") or {}
            artifact = provenance.get("artifact") or {}
            build = provenance.get("build") or {}
            if source.get("repository") != OROJENESIS_REPOSITORY:
                raise OrojenesisError("Orojenesis provenance repository mismatch")
            if source.get("commit") != OROJENESIS_COMMIT:
                raise OrojenesisError(
                    "Orojenesis provenance revision mismatch: expected "
                    f"{OROJENESIS_COMMIT}, got {source.get('commit')}"
                )
            tree_oid = str(source.get("tree_git_oid", ""))
            if tree_oid != OROJENESIS_TREE_OID:
                raise OrojenesisError("Orojenesis provenance source tree mismatch")
            if not _SHA256.fullmatch(str(source.get("archive_sha256", ""))):
                raise OrojenesisError(
                    "Orojenesis provenance lacks a source archive SHA-256"
                )
            artifact_path = Path(str(artifact.get("path", "")))
            if (
                artifact_path.is_absolute()
                or ".." in artifact_path.parts
                or (self.home / artifact_path).resolve() != self.mapper
            ):
                raise OrojenesisError("Orojenesis provenance artifact path mismatch")
            recorded_binary = str(artifact.get("sha256", ""))
            if not _SHA256.fullmatch(recorded_binary):
                raise OrojenesisError("Orojenesis provenance lacks a binary SHA-256")
            if recorded_binary != binary_sha256:
                raise OrojenesisError("Orojenesis mapper binary hash mismatch")
            wrapper_sha256 = str(build.get("compiler_wrapper_sha256", ""))
            if wrapper_sha256 != OROJENESIS_COMPILER_WRAPPER_SHA256:
                raise OrojenesisError("Orojenesis provenance compiler-wrapper mismatch")
            if build.get("builder_image") != OROJENESIS_BUILDER_IMAGE:
                raise OrojenesisError("Orojenesis provenance builder image mismatch")
            if not str(build.get("compiler", "")):
                raise OrojenesisError("Orojenesis provenance lacks build identity")
            return {
                **provenance,
                "verification_mode": "provenance_manifest",
                "provenance_sha256": _sha256(provenance_path),
            }
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
        try:
            tree_oid = subprocess.run(
                ["git", "-C", str(self.home), "rev-parse", "HEAD^{tree}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError("cannot verify Orojenesis source tree") from exc
        if tree_oid != OROJENESIS_TREE_OID:
            raise OrojenesisError("Orojenesis source tree identity mismatch")
        try:
            archive = subprocess.run(
                ["git", "-C", str(self.home), "archive", "--format=tar", "HEAD"],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError("cannot hash Orojenesis source archive") from exc
        return {
            "schema_version": OROJENESIS_IDENTITY_SCHEMA_VERSION,
            "verification_mode": "git_checkout",
            "source": {
                "repository": OROJENESIS_REPOSITORY,
                "commit": head,
                "tree_git_oid": tree_oid,
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
            },
            "artifact": {
                "path": "bin/timeloop-mapper",
                "sha256": binary_sha256,
            },
        }

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
        dimension_symbols: dict[str, str] = {}
        data_spaces: list[dict[str, Any]] = []

        def symbol(token: str) -> str:
            if token not in dimension_symbols:
                index = len(dimension_symbols)
                symbols = string.ascii_uppercase + string.ascii_lowercase
                if index >= len(symbols):
                    raise OrojenesisError(
                        "Orojenesis supports at most 52 distinct dimensions"
                    )
                dimension_symbols[token] = symbols[index]
            return dimension_symbols[token]

        for index, (operand, shape) in enumerate(zip(operands, shapes)):
            tokens = _TOKEN.findall(operand)
            if len(tokens) != len(shape):
                raise OrojenesisError("einsum rank does not match input shape")
            for token, size in zip(tokens, shape):
                if token in dimension_sizes and dimension_sizes[token] != int(size):
                    raise OrojenesisError(f"inconsistent dimension {token}")
                dimension_sizes[token] = int(size)
            data_spaces.append(
                {
                    "name": f"Input{index}",
                    "projection": [[[symbol(token)]] for token in tokens],
                }
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
                "projection": [[[symbol(token)]] for token in output_tokens],
                "read-write": True,
            }
        )
        remapped_sizes = {
            dimension_symbols[token]: size for token, size in dimension_sizes.items()
        }
        dimensions = list(remapped_sizes)
        return {
            "problem": {
                "instance": remapped_sizes,
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
    def multi_architecture(word_bits: int) -> dict[str, Any]:
        """Return the official two-buffer abstraction used for tiled fusion."""
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
                                "attributes": {
                                    "width": 64,
                                    "word-bits": int(word_bits),
                                },
                            }
                        ],
                        "subtree": [
                            {
                                "name": "PE",
                                "local": [
                                    {
                                        "name": "InputOutputBuffer",
                                        "class": "regfile",
                                        "attributes": {
                                            "sizeKB": 2147483648,
                                            "instances": 1,
                                            "word-bits": int(word_bits),
                                        },
                                    },
                                    {
                                        "name": "WeightBuffer",
                                        "class": "regfile",
                                        "attributes": {
                                            "sizeKB": 2147483648,
                                            "instances": 1,
                                            "word-bits": int(word_bits),
                                        },
                                    },
                                    {
                                        "name": "MACC",
                                        "class": "intmac",
                                        "attributes": {"datawidth": int(word_bits)},
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        }

    @staticmethod
    def multi_mapper_config(row_tile: int, *, role: str) -> dict[str, Any]:
        """Build a fusion-friendly mapping sweep for a linear matmul chain.

        This is the M/K/N equivalent of the FFMT constraints used by the
        pinned ``orojenesis_multi.ipynb`` workflow.  A fixed inner M factor
        makes equal producer-output and consumer-input utilizations an exact
        two-dimensional tile-shape match rather than a byte-count heuristic.
        """
        if int(row_tile) <= 0:
            raise OrojenesisError("multi-einsum row tile must be positive")
        main_memory_constraints = {
            "first": (None, "KNM"),
            "second": ("N=1", "KNM"),
            "middle": ("K=1 N=1", "KNM"),
            "last": ("K=1", "KNM"),
            "second_last": (None, "NKM"),
        }
        if role not in main_memory_constraints:
            raise OrojenesisError(f"invalid multi-einsum mapper role: {role}")
        main_factors, main_permutation = main_memory_constraints[role]
        main_temporal: dict[str, Any] = {
            "target": "MainMemory",
            "type": "temporal",
            "permutation": main_permutation,
        }
        if main_factors is not None:
            main_temporal["factors"] = main_factors
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
                    "target": "MainMemory",
                    "type": "datatype",
                    "keep": ["Weights", "Inputs", "Outputs"],
                    "bypass": [],
                },
                {
                    "target": "InputOutputBuffer",
                    "type": "datatype",
                    "keep": ["Inputs", "Outputs"],
                    "bypass": ["Weights"],
                },
                {
                    "target": "WeightBuffer",
                    "type": "datatype",
                    "keep": ["Weights"],
                    "bypass": ["Inputs", "Outputs"],
                },
                main_temporal,
                {
                    "target": "InputOutputBuffer",
                    "type": "temporal",
                    "factors": "M=1",
                    "permutation": "MNK",
                },
                {
                    "target": "WeightBuffer",
                    "type": "temporal",
                    "factors": f"M={int(row_tile)}",
                    "permutation": "MKN",
                },
            ],
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
            "toolchain": self.toolchain_identity,
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

    def run_multi_chain(
        self,
        chain: Sequence[tuple[str, Mapping[str, Any]]],
        output_dir: str | Path,
        *,
        word_bits: int,
    ) -> dict[str, Any]:
        """Run and compose official fusion-friendly mappings for a matmul chain."""
        descriptor = multi_einsum_problem(chain)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        word_bytes = max(1, int(word_bits) // 8)
        if int(word_bits) <= 0 or int(word_bits) % 8:
            raise OrojenesisError("multi-einsum word width must be byte aligned")

        environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
        documents = {
            "chain.yaml": descriptor,
            "architecture.yaml": self.multi_architecture(word_bits),
            "environment.yaml": environment,
        }
        paths: dict[str, Path] = {}
        for name, document in documents.items():
            path = output / name
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            paths[name] = path

        row_tiles = _divisors(int(descriptor["chain"]["layers"][0]["m"]))
        sweeps: list[dict[str, Any]] = []
        raw_paths: list[list[Path]] = []
        for layer_index, layer_descriptor in enumerate(descriptor["chain"]["layers"]):
            role = multi_einsum_mapper_role(
                layer_index, len(descriptor["chain"]["layers"])
            )
            problem = multi_einsum_layer_problem(layer_descriptor)
            problem_name = f"problem-layer-{layer_index}.yaml"
            problem_path = output / problem_name
            problem_path.write_text(yaml.safe_dump(problem, sort_keys=False))
            paths[problem_name] = problem_path
            layer_raw_paths: list[Path] = []
            for row_tile in row_tiles:
                sweep_dir = output / f"layer-{layer_index}-m-{row_tile}"
                sweep_dir.mkdir(parents=True, exist_ok=True)
                architecture_path = sweep_dir / "architecture.yaml"
                mapper_path = sweep_dir / "mapper.yaml"
                local_problem_path = sweep_dir / "problem.yaml"
                architecture_path.write_text(
                    yaml.safe_dump(self.multi_architecture(word_bits), sort_keys=False)
                )
                mapper_path.write_text(
                    yaml.safe_dump(
                        self.multi_mapper_config(row_tile, role=role),
                        sort_keys=False,
                    )
                )
                local_problem_path.write_text(yaml.safe_dump(problem, sort_keys=False))
                try:
                    completed = subprocess.run(
                        [
                            str(self.mapper),
                            str(architecture_path),
                            str(local_problem_path),
                            str(mapper_path),
                            "-o",
                            str(sweep_dir),
                        ],
                        cwd=sweep_dir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        check=False,
                        env={**os.environ, **environment},
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise OrojenesisError(
                        "multi-einsum Orojenesis execution failed"
                    ) from exc
                (sweep_dir / "stdout.log").write_text(completed.stdout)
                (sweep_dir / "stderr.log").write_text(completed.stderr)
                if completed.returncode != 0:
                    raise OrojenesisError(
                        "multi-einsum Orojenesis exited with status "
                        f"{completed.returncode} for layer {layer_index}, M={row_tile}"
                    )
                raw_path = sweep_dir / "timeloop-mapper.oaves.csv"
                # Parse now so missing mapping-level fields fail before an
                # artifact can be emitted.
                parse_multi_mapping_records(raw_path, word_bytes=word_bytes)
                layer_raw_paths.append(raw_path)
                for leaf in ("architecture.yaml", "mapper.yaml", "problem.yaml"):
                    evidence_name = f"layer-{layer_index}-m-{row_tile}-{leaf}"
                    paths[evidence_name] = sweep_dir / leaf
                paths[f"layer-{layer_index}-m-{row_tile}-raw"] = raw_path
            raw_paths.append(layer_raw_paths)
            sweeps.append(
                {
                    "layer_id": str(layer_descriptor["id"]),
                    "row_tiles": row_tiles,
                    "role": role,
                }
            )

        curve = compose_multi_einsum_curve(
            raw_paths, row_tiles=row_tiles, word_bytes=word_bytes
        )
        curve_path = output / "multi-einsum-curve.csv"
        with curve_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for point in curve:
                writer.writerow(
                    [
                        point["buffer_bytes"],
                        point["operational_intensity"],
                        point["dram_accesses_words"],
                        json.dumps(point.get("mappings") or [], separators=(",", ":")),
                        point["row_tile"],
                    ]
                )
        paths["curve"] = curve_path
        return {
            "solver": MULTI_EINSUM_SOLVER,
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "composition": MULTI_EINSUM_COMPOSITION,
            "word_bits": int(word_bits),
            "environment": environment,
            "problem": descriptor,
            "sweeps": sweeps,
            "curve": curve,
            "evidence_files": {
                name: {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
        }

    def run_multi_region(
        self,
        region: Mapping[str, Any],
        output_dir: str | Path,
        *,
        word_bits: int,
    ) -> dict[str, Any]:
        """Run independent mapper sweeps and compose an extended MatMul region."""
        descriptor = multi_einsum_region_problem(region)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        word_bytes = max(1, int(word_bits) // 8)
        if int(word_bits) <= 0 or int(word_bits) % 8:
            raise OrojenesisError("multi-einsum word width must be byte aligned")
        environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
        documents = {
            "region.yaml": descriptor,
            "architecture.yaml": self.multi_architecture(word_bits),
            "environment.yaml": environment,
        }
        paths: dict[str, Path] = {}
        for name, document in documents.items():
            path = output / name
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            paths[name] = path

        raw_paths: dict[str, list[Path]] = {}
        row_tiles_by_node: dict[str, list[int]] = {}
        sweeps: list[dict[str, Any]] = []
        for node_index, node in enumerate(descriptor["nodes"]):
            node_id = str(node["id"])
            row_tiles = _divisors(int(node["m"]))
            row_tiles_by_node[node_id] = row_tiles
            role = multi_einsum_region_mapper_role(descriptor, node_id)
            problem = multi_einsum_layer_problem(node)
            problem_name = f"problem-node-{node_index}.yaml"
            problem_path = output / problem_name
            problem_path.write_text(yaml.safe_dump(problem, sort_keys=False))
            paths[problem_name] = problem_path
            node_raw_paths: list[Path] = []
            for row_tile in row_tiles:
                prefix = f"node-{node_index}-m-{row_tile}"
                sweep_dir = output / prefix
                sweep_dir.mkdir(parents=True, exist_ok=True)
                architecture_path = sweep_dir / "architecture.yaml"
                mapper_path = sweep_dir / "mapper.yaml"
                local_problem_path = sweep_dir / "problem.yaml"
                architecture_path.write_text(
                    yaml.safe_dump(self.multi_architecture(word_bits), sort_keys=False)
                )
                mapper_path.write_text(
                    yaml.safe_dump(
                        self.multi_mapper_config(row_tile, role=role), sort_keys=False
                    )
                )
                local_problem_path.write_text(yaml.safe_dump(problem, sort_keys=False))
                try:
                    completed = subprocess.run(
                        [
                            str(self.mapper),
                            str(architecture_path),
                            str(local_problem_path),
                            str(mapper_path),
                            "-o",
                            str(sweep_dir),
                        ],
                        cwd=sweep_dir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        check=False,
                        env={**os.environ, **environment},
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise OrojenesisError(
                        "multi-einsum region Orojenesis execution failed"
                    ) from exc
                (sweep_dir / "stdout.log").write_text(completed.stdout)
                (sweep_dir / "stderr.log").write_text(completed.stderr)
                if completed.returncode != 0:
                    raise OrojenesisError(
                        "multi-einsum region Orojenesis exited with status "
                        f"{completed.returncode} for node {node_id}, M={row_tile}"
                    )
                raw_path = sweep_dir / "timeloop-mapper.oaves.csv"
                parse_multi_mapping_records(raw_path, word_bytes=word_bytes)
                node_raw_paths.append(raw_path)
                for leaf in ("architecture.yaml", "mapper.yaml", "problem.yaml"):
                    paths[f"{prefix}-{leaf}"] = sweep_dir / leaf
                paths[f"{prefix}-raw"] = raw_path
            raw_paths[node_id] = node_raw_paths
            sweeps.append(
                {
                    "node_id": node_id,
                    "row_tiles": row_tiles,
                    "role": role,
                }
            )

        curve = compose_multi_einsum_region_curve(
            descriptor,
            raw_paths,
            row_tiles_by_node=row_tiles_by_node,
            word_bytes=word_bytes,
        )
        curve_path = output / "multi-einsum-region-curve.csv"
        with curve_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for point in curve:
                writer.writerow(
                    [
                        point["buffer_bytes"],
                        point["operational_intensity"],
                        point["dram_accesses_words"],
                        json.dumps(point.get("mappings") or [], separators=(",", ":")),
                    ]
                )
        paths["curve"] = curve_path
        return {
            "solver": MULTI_EINSUM_SOLVER,
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "composition": descriptor["composition"],
            "word_bits": int(word_bits),
            "environment": environment,
            "problem": descriptor,
            "sweeps": sweeps,
            "curve": curve,
            "evidence_files": {
                name: {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
                for name, path in paths.items()
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


def _divisors(value: int) -> list[int]:
    if value <= 0:
        raise OrojenesisError("multi-einsum dimensions must be positive")
    small: list[int] = []
    large: list[int] = []
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            small.append(candidate)
            if candidate * candidate != value:
                large.append(value // candidate)
        candidate += 1
    return small + list(reversed(large))


def multi_einsum_row_tiles(value: int) -> list[int]:
    """Return the complete deterministic FFMT sweep for the shared M axis."""
    return _divisors(value)


def multi_einsum_mapper_role(layer_index: int, layer_count: int) -> str:
    """Map a chain position to the pinned ``_relax_io_kn`` FFMT variant."""
    if layer_count < 2 or layer_index not in range(layer_count):
        raise OrojenesisError("invalid multi-einsum chain position")
    if layer_index == 0:
        return "first"
    if layer_count == 2:
        return "second_last"
    if layer_index == 1:
        return "second"
    if layer_index == layer_count - 1:
        return "last"
    return "middle"


def _matmul_descriptor(layer_id: str, layer: Mapping[str, Any]) -> dict[str, Any]:
    semantic = layer.get("semantic_op") or {}
    if semantic.get("kind") != "einsum":
        raise OrojenesisError("multi-einsum chains accept exact einsum layers only")
    equation = str(semantic.get("equation", ""))
    if "->" not in equation:
        raise OrojenesisError("multi-einsum equation must have an explicit output")
    lhs, rhs = equation.split("->", 1)
    operands = lhs.split(",")
    operand_tokens = [_TOKEN.findall(operand) for operand in operands]
    output_tokens = _TOKEN.findall(rhs)
    if (
        len(operand_tokens) != 2
        or any(len(tokens) != 2 for tokens in operand_tokens)
        or len(output_tokens) != 2
    ):
        raise OrojenesisError("multi-einsum currently requires binary rank-2 matmul")
    m_token, k_token = operand_tokens[0]
    second_k, n_token = operand_tokens[1]
    if (
        k_token != second_k
        or output_tokens != [m_token, n_token]
        or len({m_token, k_token, n_token}) != 3
    ):
        raise OrojenesisError("multi-einsum layer is not a canonical matmul")
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    input_names = [str(name) for name in names.get("inputs") or []]
    output_names = [str(name) for name in names.get("outputs") or []]
    input_shapes = [list(shape) for shape in shapes.get("inputs") or []]
    output_shapes = [list(shape) for shape in shapes.get("outputs") or []]
    input_dtypes = [str(dtype) for dtype in dtypes.get("inputs") or []]
    output_dtypes = [str(dtype) for dtype in dtypes.get("outputs") or []]
    if not (
        len(input_names) == len(input_shapes) == len(input_dtypes) == 2
        and len(output_names) == len(output_shapes) == len(output_dtypes) == 1
    ):
        raise OrojenesisError("multi-einsum tensor metadata arity mismatch")
    m_size, k_size = (int(value) for value in input_shapes[0])
    second_k_size, n_size = (int(value) for value in input_shapes[1])
    if second_k_size != k_size or output_shapes[0] != [m_size, n_size]:
        raise OrojenesisError("multi-einsum matmul shapes are inconsistent")
    effects = semantic.get("effects") or {}
    if any(
        (
            effects.get("mutates"),
            effects.get("aliases"),
            effects.get("atomic"),
            effects.get("opaque_library_call"),
        )
    ):
        raise OrojenesisError("multi-einsum chain contains observable effects")
    if len(set(input_dtypes + output_dtypes)) != 1:
        raise OrojenesisError("multi-einsum chain requires one exact tensor dtype")
    return {
        "id": str(layer_id),
        "equation": equation,
        "input": input_names[0],
        "weight": input_names[1],
        "output": output_names[0],
        "m": m_size,
        "k": k_size,
        "n": n_size,
        "dtype": input_dtypes[0],
    }


def multi_einsum_problem(
    chain: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Create a canonical, hashable linear-matmul-chain problem."""
    descriptors = [
        _matmul_descriptor(str(layer_id), layer) for layer_id, layer in chain
    ]
    if len(descriptors) < 2:
        raise OrojenesisError("multi-einsum proof requires at least two layers")
    first_m = descriptors[0]["m"]
    dtype = descriptors[0]["dtype"]
    for previous, current in zip(descriptors, descriptors[1:]):
        if previous["output"] != current["input"]:
            raise OrojenesisError(
                "multi-einsum layers are not a producer-consumer chain"
            )
        if previous["m"] != current["m"] or previous["n"] != current["k"]:
            raise OrojenesisError("multi-einsum boundary shapes do not match")
        if current["m"] != first_m or current["dtype"] != dtype:
            raise OrojenesisError("multi-einsum chain M dimension or dtype drifted")
    return {
        "schema_version": 1,
        "chain": {"kind": "linear_matmul", "layers": descriptors},
    }


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _region_matmul_descriptor(
    layer_id: str, layer: Mapping[str, Any]
) -> dict[str, Any]:
    """Canonicalize rank-2 or broadcast-weight batched matrix multiplication."""
    semantic = layer.get("semantic_op") or {}
    if semantic.get("kind") != "einsum":
        raise OrojenesisError("multi-einsum regions accept exact einsum layers only")
    equation = str(semantic.get("equation", ""))
    if "->" not in equation:
        raise OrojenesisError("multi-einsum equation must have an explicit output")
    lhs, rhs = equation.split("->", 1)
    operand_tokens = [_TOKEN.findall(operand) for operand in lhs.split(",")]
    output_tokens = _TOKEN.findall(rhs)
    if len(operand_tokens) != 2 or len(output_tokens) < 2:
        raise OrojenesisError("multi-einsum region requires binary matrix contraction")

    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    input_names = [str(name) for name in names.get("inputs") or []]
    output_names = [str(name) for name in names.get("outputs") or []]
    input_shapes = [list(shape) for shape in shapes.get("inputs") or []]
    output_shapes = [list(shape) for shape in shapes.get("outputs") or []]
    input_dtypes = [str(dtype) for dtype in dtypes.get("inputs") or []]
    output_dtypes = [str(dtype) for dtype in dtypes.get("outputs") or []]
    if not (
        len(input_names) == len(input_shapes) == len(input_dtypes) == 2
        and len(output_names) == len(output_shapes) == len(output_dtypes) == 1
        and all(
            len(tokens) == len(shape)
            for tokens, shape in zip(operand_tokens, input_shapes)
        )
        and len(output_tokens) == len(output_shapes[0])
    ):
        raise OrojenesisError("multi-einsum region tensor metadata arity mismatch")

    token_sizes: dict[str, int] = {}
    for tokens, shape in [
        *zip(operand_tokens, input_shapes),
        (output_tokens, output_shapes[0]),
    ]:
        for token, size in zip(tokens, shape):
            size = int(size)
            if size <= 0 or (token in token_sizes and token_sizes[token] != size):
                raise OrojenesisError("multi-einsum region dimensions are inconsistent")
            token_sizes[token] = size

    reductions = (set(operand_tokens[0]) & set(operand_tokens[1])) - set(output_tokens)
    if len(reductions) != 1:
        raise OrojenesisError("multi-einsum region requires one reduction dimension")
    reduction = next(iter(reductions))
    candidates: list[tuple[int, int, str, list[str]]] = []
    for activation_index, weight_index in ((0, 1), (1, 0)):
        activation_tokens = operand_tokens[activation_index]
        weight_tokens = operand_tokens[weight_index]
        weight_free = [token for token in weight_tokens if token != reduction]
        activation_free = [token for token in activation_tokens if token != reduction]
        if (
            len(weight_tokens) == 2
            and len(weight_free) == 1
            and len(activation_free) >= 1
            and set(output_tokens) == set([*activation_free, weight_free[0]])
            and len(output_tokens) == len(set(output_tokens))
        ):
            candidates.append(
                (activation_index, weight_index, weight_free[0], activation_free)
            )
    if len(candidates) > 1:
        # ATen matmul/linear handlers preserve the activation as positional
        # operand zero.  Use that exact call ordering to resolve the otherwise
        # symmetric rank-2 equation without guessing from tensor names.
        candidates = [item for item in candidates if item[0] == 0]
    if len(candidates) != 1:
        raise OrojenesisError(
            "multi-einsum region requires an unambiguous broadcast-weight matmul"
        )
    activation_index, weight_index, n_token, activation_free = candidates[0]
    row_tokens = [token for token in output_tokens if token != n_token]
    if set(row_tokens) != set(activation_free):
        raise OrojenesisError("multi-einsum output does not preserve activation axes")
    if (
        operand_tokens[activation_index][-1] != reduction
        or output_tokens[-1] != n_token
    ):
        raise OrojenesisError(
            "multi-einsum region requires row-major activation/output axes"
        )

    effects = semantic.get("effects") or {}
    if any(
        (
            effects.get("mutates"),
            effects.get("aliases"),
            effects.get("atomic"),
            effects.get("opaque_library_call"),
        )
    ):
        raise OrojenesisError("multi-einsum region contains observable effects")
    ordered_dtypes = [
        input_dtypes[activation_index],
        input_dtypes[weight_index],
        output_dtypes[0],
    ]
    if len(set(ordered_dtypes)) != 1:
        raise OrojenesisError("multi-einsum region requires one exact tensor dtype")

    row_shape = [token_sizes[token] for token in row_tokens]
    descriptor = {
        "id": str(layer_id),
        "equation": equation,
        "kind": "batched_matmul" if len(row_tokens) > 1 else "matmul",
        "input": input_names[activation_index],
        "weight": input_names[weight_index],
        "output": output_names[0],
        "activation_operand": activation_index,
        "weight_operand": weight_index,
        "activation_axes": operand_tokens[activation_index],
        "weight_axes": operand_tokens[weight_index],
        "output_axes": output_tokens,
        "row_axes": row_tokens,
        "row_shape": row_shape,
        "m": _shape_product(row_shape),
        "k": token_sizes[reduction],
        "n": token_sizes[n_token],
        "dtype": ordered_dtypes[0],
    }
    if len(row_tokens) > 1:
        descriptor["batch_axes"] = row_tokens[:-1]
        descriptor["batch_shape"] = row_shape[:-1]
    else:
        descriptor["batch_axes"] = []
        descriptor["batch_shape"] = []
    return descriptor


def _internal_zero_copy_view(layer: Mapping[str, Any]) -> dict[str, Any] | None:
    semantic = layer.get("semantic_op") or {}
    target = str(semantic.get("target", ""))
    if semantic.get("kind") != "aten" or target not in {
        "view",
        "transpose",
        "permute",
        "squeeze",
        "unsqueeze",
    }:
        return None
    effects = semantic.get("effects") or {}
    if (
        effects.get("mutates")
        or effects.get("atomic")
        or effects.get("opaque_library_call")
    ):
        return None
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    input_names = [str(name) for name in names.get("inputs") or []]
    output_names = [str(name) for name in names.get("outputs") or []]
    input_shapes = [list(shape) for shape in shapes.get("inputs") or []]
    output_shapes = [list(shape) for shape in shapes.get("outputs") or []]
    input_dtypes = [str(dtype) for dtype in dtypes.get("inputs") or []]
    output_dtypes = [str(dtype) for dtype in dtypes.get("outputs") or []]
    if not (
        len(input_names) == len(output_names) == 1
        and len(input_shapes) == len(output_shapes) == 1
        and len(input_dtypes) == len(output_dtypes) == 1
        and input_dtypes[0] == output_dtypes[0]
        and _shape_product(input_shapes[0]) == _shape_product(output_shapes[0])
    ):
        return None
    aliases = effects.get("aliases")
    if (
        not isinstance(aliases, list)
        or len(aliases) != 1
        or not isinstance(aliases[0], Mapping)
    ):
        return None
    alias = aliases[0]
    if (
        alias.get("input") != 0
        or alias.get("output") != 0
        or alias.get("conditional") is not False
    ):
        return None
    if target in {"view", "squeeze", "unsqueeze"}:
        input_flat = [_shape_product(input_shapes[0][:-1]), input_shapes[0][-1]]
        output_flat = [_shape_product(output_shapes[0][:-1]), output_shapes[0][-1]]
        if input_flat != output_flat:
            return None
        axis_map = [0, 1]
    else:
        if len(input_shapes[0]) != 2 or len(output_shapes[0]) != 2:
            return None
        literal_arguments = [
            item.get("value")
            for item in semantic.get("arguments") or []
            if isinstance(item, Mapping) and "value" in item
        ]
        if target == "transpose":
            if len(literal_arguments) != 2 or any(
                not isinstance(value, int) for value in literal_arguments
            ):
                return None
            dimensions = [
                value % 2 for value in literal_arguments if isinstance(value, int)
            ]
            if dimensions == [0, 1] or dimensions == [1, 0]:
                axis_map = [1, 0]
            elif dimensions[0] == dimensions[1]:
                axis_map = [0, 1]
            else:
                return None
        else:
            permutation = literal_arguments[-1] if literal_arguments else None
            if (
                not isinstance(permutation, (list, tuple))
                or len(permutation) != 2
                or any(not isinstance(value, int) for value in permutation)
            ):
                return None
            axis_map = [int(value) % 2 for value in permutation]
            if sorted(axis_map) != [0, 1]:
                return None
        expected_output_shape = [input_shapes[0][index] for index in axis_map]
        if output_shapes[0] != expected_output_shape:
            return None
    return {
        "target": target,
        "input": input_names[0],
        "output": output_names[0],
        "input_shape": input_shapes[0],
        "output_shape": output_shapes[0],
        "dtype": input_dtypes[0],
        "axis_map": axis_map,
    }


def _region_axis_map(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    bridges: Sequence[str],
    views: Mapping[str, Mapping[str, Any]],
) -> list[int] | None:
    producer_shape = [int(producer["m"]), int(producer["n"])]
    consumer_shape = [int(consumer["m"]), int(consumer["k"])]
    axis_map = [0, 1]
    for bridge in bridges:
        bridge_map = list(views[bridge]["axis_map"])
        axis_map = [axis_map[index] for index in bridge_map]
    mapped_shape = [producer_shape[index] for index in axis_map]
    return axis_map if mapped_shape == consumer_shape else None


def find_multi_einsum_regions(
    layers: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Find endpoint-proven MatMul regions beyond the legacy direct chain."""
    layer_map = {str(key): value for key, value in layers.items()}
    producers = {
        str(name): str(layer_id)
        for layer_id, layer in layer_map.items()
        for name in (layer.get("tensor_names") or {}).get("outputs") or []
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    for layer_id, layer in layer_map.items():
        for name in (layer.get("tensor_names") or {}).get("inputs") or []:
            consumers[str(name)].append(layer_id)
    descriptors: dict[str, dict[str, Any]] = {}
    views: dict[str, dict[str, Any]] = {}
    for layer_id, layer in layer_map.items():
        try:
            descriptors[layer_id] = _region_matmul_descriptor(layer_id, layer)
        except OrojenesisError:
            pass
        view = _internal_zero_copy_view(layer)
        if view is not None:
            views[layer_id] = view

    def trace_back(tensor: str) -> tuple[str | None, list[str], str]:
        path: list[str] = []
        current = str(tensor)
        seen: set[str] = set()
        while True:
            producer = producers.get(current)
            if producer is None or producer in seen:
                return None, [], current
            seen.add(producer)
            if producer in descriptors:
                return producer, list(reversed(path)), current
            if producer not in views:
                return producer, list(reversed(path)), current
            view = views[producer]
            if len(consumers.get(str(view["output"])) or []) != 1:
                return None, [], current
            path.append(producer)
            current = str(view["input"])

    edges: list[dict[str, Any]] = []
    entry_bridges: dict[str, list[str]] = {}
    valid_nodes: set[str] = set(descriptors)
    for consumer_id, descriptor in descriptors.items():
        producer_id, bridges, _source_tensor = trace_back(str(descriptor["input"]))
        if producer_id in descriptors:
            axis_map = _region_axis_map(
                descriptors[producer_id], descriptor, bridges, views
            )
            if axis_map is None:
                valid_nodes.discard(consumer_id)
                continue
            edges.append(
                {
                    "producer": producer_id,
                    "consumer": consumer_id,
                    "tensor": str(descriptors[producer_id]["output"]),
                    "bridges": bridges,
                    "axis_map": axis_map,
                    "layer_path": [producer_id, *bridges, consumer_id],
                }
            )
        else:
            source = layer_map.get(str(producer_id), {})
            if str(source.get("type", "")).lower() != "start":
                valid_nodes.discard(consumer_id)
                continue
            entry_bridges[consumer_id] = bridges

    edges = [
        edge
        for edge in edges
        if edge["producer"] in valid_nodes and edge["consumer"] in valid_nodes
    ]
    predecessors = {str(edge["consumer"]): str(edge["producer"]) for edge in edges}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))

    # Every weight and every root activation must resolve to an explicit graph input.
    for node_id in list(valid_nodes):
        descriptor = descriptors[node_id]
        weight_producer, weight_bridges, _ = trace_back(str(descriptor["weight"]))
        if (
            str(layer_map.get(str(weight_producer), {}).get("type", "")).lower()
            != "start"
        ):
            valid_nodes.discard(node_id)
            continue
        descriptor["weight_bridges"] = weight_bridges
        if node_id not in predecessors:
            activation_producer, bridges, _ = trace_back(str(descriptor["input"]))
            if (
                str(layer_map.get(str(activation_producer), {}).get("type", "")).lower()
                != "start"
            ):
                valid_nodes.discard(node_id)
            else:
                entry_bridges[node_id] = bridges

    edges = [
        edge
        for edge in edges
        if edge["producer"] in valid_nodes and edge["consumer"] in valid_nodes
    ]
    predecessors = {str(edge["consumer"]): str(edge["producer"]) for edge in edges}
    successors = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))

    # Build deterministic undirected components of the contraction graph.
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        left, right = str(edge["producer"]), str(edge["consumer"])
        neighbors[left].add(right)
        neighbors[right].add(left)
    legacy_sets = {tuple(chain) for chain in find_multi_einsum_chains(layer_map)}
    regions: list[dict[str, Any]] = []
    visited: set[str] = set()
    for seed in sorted(neighbors):
        if seed in visited:
            continue
        stack = [seed]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(neighbors[node], reverse=True))
        visited.update(component)
        component_edges = [
            edge
            for edge in edges
            if edge["producer"] in component and edge["consumer"] in component
        ]
        roots = sorted(node for node in component if node not in predecessors)
        if (
            len(component) < 2
            or len(roots) != 1
            or len(component_edges) != len(component) - 1
        ):
            continue
        schedule: list[str] = []
        ready = list(roots)
        while ready:
            node = ready.pop(0)
            schedule.append(node)
            ready.extend(sorted(successors.get(node) or []))
        if len(schedule) != len(component):
            continue
        leaves = sorted(node for node in component if not successors.get(node))
        if any(consumers.get(str(descriptors[node]["output"])) for node in leaves):
            continue
        # Direct canonical endpoint chains remain on the already validated v1 path.
        if (
            tuple(schedule) in legacy_sets
            and all(not edge["bridges"] for edge in component_edges)
            and all(descriptors[node]["kind"] == "matmul" for node in component)
        ):
            continue
        has_fanout = any(len(successors.get(node) or []) > 1 for node in component)
        has_batch = any(
            descriptors[node]["kind"] == "batched_matmul" for node in component
        )
        if has_fanout:
            composition = MULTI_EINSUM_FANOUT_COMPOSITION
            kind = "matmul_fanout_tree"
        elif has_batch:
            composition = MULTI_EINSUM_BATCH_COMPOSITION
            kind = "broadcast_batch_linear_matmul"
        else:
            composition = MULTI_EINSUM_LAYOUT_COMPOSITION
            kind = "linear_matmul_with_axis_maps"
        physical_paths = [list(edge["layer_path"]) for edge in component_edges]
        for node in schedule:
            entry = entry_bridges.get(node) or []
            if entry:
                physical_paths.append([*entry, node])
            weight = descriptors[node].get("weight_bridges") or []
            if weight:
                physical_paths.append([*weight, node])
        ordered_edges = [
            edge
            for producer in schedule
            for consumer in schedule
            for edge in component_edges
            if str(edge["producer"]) == producer and str(edge["consumer"]) == consumer
        ]
        regions.append(
            {
                "schema_version": 1,
                "kind": kind,
                "composition": composition,
                "nodes": [descriptors[node] for node in schedule],
                "edges": ordered_edges,
                "roots": roots,
                "leaves": leaves,
                "schedule": schedule,
                "physical_paths": physical_paths,
            }
        )
    return regions


def multi_einsum_region_problem(region: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a supported extended MatMul region."""
    try:
        descriptor = json.loads(json.dumps(region, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise OrojenesisError("multi-einsum region is not serializable") from exc
    if int(descriptor.get("schema_version", 0)) != 1:
        raise OrojenesisError("unsupported multi-einsum region schema")
    compositions = {
        "linear_matmul_with_axis_maps": MULTI_EINSUM_LAYOUT_COMPOSITION,
        "broadcast_batch_linear_matmul": MULTI_EINSUM_BATCH_COMPOSITION,
        "matmul_fanout_tree": MULTI_EINSUM_FANOUT_COMPOSITION,
    }
    if compositions.get(str(descriptor.get("kind"))) != descriptor.get("composition"):
        raise OrojenesisError("multi-einsum region composition mismatch")
    nodes = descriptor.get("nodes") or []
    schedule = [str(item) for item in descriptor.get("schedule") or []]
    node_ids = [str(node.get("id")) for node in nodes]
    if len(nodes) < 2 or schedule != node_ids or len(node_ids) != len(set(node_ids)):
        raise OrojenesisError("multi-einsum region schedule is invalid")
    for node in nodes:
        if (
            str(node.get("kind")) not in {"matmul", "batched_matmul"}
            or any(int(node.get(name, 0)) <= 0 for name in ("m", "k", "n"))
            or not str(node.get("dtype", ""))
        ):
            raise OrojenesisError("multi-einsum region node is invalid")
    positions = {node_id: index for index, node_id in enumerate(schedule)}
    predecessors: dict[str, str] = {}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in descriptor.get("edges") or []:
        producer = str(edge.get("producer"))
        consumer = str(edge.get("consumer"))
        axis_map = edge.get("axis_map")
        if (
            producer not in positions
            or consumer not in positions
            or positions[producer] >= positions[consumer]
            or consumer in predecessors
            or axis_map not in ([0, 1], [1, 0])
            or list(edge.get("layer_path") or [])[0:1] != [producer]
            or list(edge.get("layer_path") or [])[-1:] != [consumer]
        ):
            raise OrojenesisError("multi-einsum region edge is invalid")
        predecessors[consumer] = producer
        successors[producer].append(consumer)
    roots = sorted(node_id for node_id in schedule if node_id not in predecessors)
    leaves = sorted(node_id for node_id in schedule if not successors.get(node_id))
    if (
        len(roots) != 1
        or len(predecessors) != len(nodes) - 1
        or descriptor.get("roots") != roots
        or descriptor.get("leaves") != leaves
    ):
        raise OrojenesisError("multi-einsum region is not an arborescence")
    if descriptor["composition"] != MULTI_EINSUM_FANOUT_COMPOSITION and any(
        len(items) > 1 for items in successors.values()
    ):
        raise OrojenesisError("linear multi-einsum region contains fan-out")
    if descriptor["composition"] == MULTI_EINSUM_BATCH_COMPOSITION and not any(
        node.get("kind") == "batched_matmul" for node in nodes
    ):
        raise OrojenesisError("batched multi-einsum region has no batch dimension")
    return descriptor


def multi_einsum_region_mapper_role(region: Mapping[str, Any], node_id: str) -> str:
    """Choose the pinned FFMT constraint variant for a region node."""
    descriptor = multi_einsum_region_problem(region)
    schedule = [str(item) for item in descriptor["schedule"]]
    edges = descriptor["edges"]
    predecessors = {str(edge["consumer"]): str(edge["producer"]) for edge in edges}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))
    if node_id not in schedule:
        raise OrojenesisError("multi-einsum region mapper node is unknown")
    if node_id not in predecessors:
        return "first"
    if not successors.get(node_id):
        return "second_last" if len(schedule) == 2 else "last"
    parent = predecessors[node_id]
    if parent not in predecessors:
        return "second"
    return "middle"


def _mapping_tile(
    record: Mapping[str, Any], *, row_tile: int, word_bytes: int, side: str
) -> tuple[int, int]:
    utilization = int(record[f"{side}_util_bytes"])
    denominator = int(row_tile) * int(word_bytes)
    if denominator <= 0 or utilization % denominator:
        raise OrojenesisError("multi-einsum mapping tile is not rectangular")
    feature_tile = utilization // denominator
    if feature_tile <= 0:
        raise OrojenesisError("multi-einsum mapping tile is empty")
    return int(row_tile), int(feature_tile)


def compose_multi_einsum_region_curve(
    region: Mapping[str, Any],
    raw_paths: Mapping[str, Sequence[str | Path]],
    *,
    row_tiles_by_node: Mapping[str, Sequence[int]],
    word_bytes: int,
) -> list[dict[str, Any]]:
    """Compose replayable mapping assignments for a linear or fan-out region."""
    descriptor = multi_einsum_region_problem(region)
    if int(word_bytes) <= 0:
        raise OrojenesisError("multi-einsum region word width must be positive")
    schedule = [str(item) for item in descriptor["schedule"]]
    edges = descriptor["edges"]
    edge_by_consumer = {str(edge["consumer"]): edge for edge in edges}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))
    leaves = set(str(item) for item in descriptor["leaves"])
    candidates: dict[str, list[dict[str, Any]]] = {}
    for node_id in schedule:
        paths = list(raw_paths.get(node_id) or [])
        row_tiles = [int(item) for item in row_tiles_by_node.get(node_id) or []]
        if not paths or len(paths) != len(row_tiles):
            raise OrojenesisError("multi-einsum region sweep matrix is incomplete")
        node_candidates: list[dict[str, Any]] = []
        for path, row_tile in zip(paths, row_tiles):
            for raw_record in parse_multi_mapping_records(
                path, word_bytes=int(word_bytes)
            ):
                record = dict(raw_record)
                record["row_tile"] = row_tile
                record["input_tile"] = _mapping_tile(
                    record,
                    row_tile=row_tile,
                    word_bytes=word_bytes,
                    side="input",
                )
                record["output_tile"] = _mapping_tile(
                    record,
                    row_tile=row_tile,
                    word_bytes=word_bytes,
                    side="output",
                )
                node_candidates.append(record)
        if not node_candidates:
            raise OrojenesisError("multi-einsum region node has no mapping candidates")
        candidates[node_id] = node_candidates

    states: list[dict[str, Any]] = [
        {
            "assignments": {},
            "buffer_bytes": 0,
            "dram_accesses_words": 0.0,
            "compute_ops": 0.0,
            "mappings": [],
        }
    ]
    processed: set[str] = set()
    for node_id in schedule:
        edge = edge_by_consumer.get(node_id)
        next_states: list[dict[str, Any]] = []
        for state in states:
            for record in candidates[node_id]:
                if edge is not None:
                    producer_record = state["assignments"].get(str(edge["producer"]))
                    if producer_record is None:
                        continue
                    producer_tile = tuple(producer_record["output_tile"])
                    consumer_tile = tuple(record["input_tile"])
                    axis_map = [int(item) for item in edge["axis_map"]]
                    transformed = tuple(producer_tile[index] for index in axis_map)
                    if transformed != consumer_tile:
                        continue
                accesses = float(record["weight_accesses_words"])
                if edge is None:
                    accesses += float(record["input_accesses_words"])
                if node_id in leaves:
                    accesses += float(record["output_accesses_words"])
                assignments = dict(state["assignments"])
                assignments[node_id] = record
                next_states.append(
                    {
                        "assignments": assignments,
                        "buffer_bytes": int(state["buffer_bytes"])
                        + int(record["buffer_bytes"]),
                        "dram_accesses_words": float(state["dram_accesses_words"])
                        + accesses,
                        "compute_ops": float(state["compute_ops"])
                        + float(record["compute_ops"]),
                        "mappings": [
                            *list(state["mappings"]),
                            str(record["mapping"]),
                        ],
                    }
                )
        if not next_states:
            raise OrojenesisError(
                f"multi-einsum region has no compatible mapping for {node_id}"
            )
        processed.add(node_id)
        active = [
            item
            for item in schedule
            if item in processed
            and any(child not in processed for child in successors.get(item) or [])
        ]
        best: dict[tuple[Any, ...], dict[str, Any]] = {}
        for state in next_states:
            key: tuple[Any, ...] = (
                int(state["buffer_bytes"]),
                *(
                    (item, tuple(state["assignments"][item]["output_tile"]))
                    for item in active
                ),
            )
            previous = best.get(key)
            if previous is None or float(state["dram_accesses_words"]) < float(
                previous["dram_accesses_words"]
            ):
                best[key] = state
        states = list(best.values())

    points: list[dict[str, Any]] = []
    for state in states:
        accesses = float(state["dram_accesses_words"])
        points.append(
            {
                "buffer_bytes": int(state["buffer_bytes"]),
                "operational_intensity": (
                    0.0
                    if accesses == 0
                    else float(state["compute_ops"]) / (accesses * int(word_bytes))
                ),
                "dram_accesses_words": accesses,
                "dram_bytes": accesses * int(word_bytes),
                "mappings": list(state["mappings"]),
            }
        )
    best_by_capacity: dict[int, dict[str, Any]] = {}
    for point in points:
        capacity = int(point["buffer_bytes"])
        previous = best_by_capacity.get(capacity)
        if previous is None or float(point["dram_bytes"]) < float(
            previous["dram_bytes"]
        ):
            best_by_capacity[capacity] = point
    pareto: list[dict[str, Any]] = []
    best_traffic = float("inf")
    for point in sorted(
        best_by_capacity.values(), key=lambda item: int(item["buffer_bytes"])
    ):
        if float(point["dram_bytes"]) < best_traffic:
            pareto.append(point)
            best_traffic = float(point["dram_bytes"])
    if not pareto:
        raise OrojenesisError("multi-einsum region has no Pareto mapping")
    return pareto


def parse_multi_einsum_region_curve(
    path: str | Path, *, word_bytes: int
) -> list[dict[str, Any]]:
    """Parse an extended-region joint curve without trusting analysis YAML."""
    source = Path(path)
    if not source.is_file():
        raise OrojenesisError(f"missing multi-einsum region curve: {source}")
    points: list[dict[str, Any]] = []
    with source.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                continue
            try:
                accesses = float(row[2])
                mappings = json.loads(row[3])
                point = {
                    "buffer_bytes": int(float(row[0])),
                    "operational_intensity": float(row[1]),
                    "dram_accesses_words": accesses,
                    "dram_bytes": accesses * int(word_bytes),
                    "mappings": mappings,
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                point["buffer_bytes"] <= 0
                or point["dram_accesses_words"] < 0
                or not isinstance(mappings, list)
                or any(not isinstance(item, str) for item in mappings)
            ):
                raise OrojenesisError("serialized multi-einsum region curve is invalid")
            points.append(point)
    if not points:
        raise OrojenesisError("serialized multi-einsum region curve has no points")
    return points


def multi_einsum_layer_problem(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "problem": {
            "instance": {
                "M": int(descriptor["m"]),
                "K": int(descriptor["k"]),
                "N": int(descriptor["n"]),
            },
            "shape": {
                "data-spaces": [
                    {"name": "Weights", "projection": [[["K"]], [["N"]]]},
                    {"name": "Inputs", "projection": [[["M"]], [["K"]]]},
                    {
                        "name": "Outputs",
                        "projection": [[["M"]], [["N"]]],
                        "read-write": True,
                    },
                ],
                "dimensions": ["M", "K", "N"],
            },
        }
    }


def find_multi_einsum_chains(
    layers: Mapping[str, Mapping[str, Any]],
) -> list[list[str]]:
    """Find complete endpoint-to-endpoint chains supported by tiled fusion."""
    producers = {
        str(name): str(layer_id)
        for layer_id, layer in layers.items()
        for name in (layer.get("tensor_names") or {}).get("outputs") or []
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    for layer_id, layer in layers.items():
        for name in (layer.get("tensor_names") or {}).get("inputs") or []:
            consumers[str(name)].append(str(layer_id))
    einsums: dict[str, dict[str, Any]] = {}
    for layer_id, layer in layers.items():
        try:
            einsums[str(layer_id)] = _matmul_descriptor(str(layer_id), layer)
        except OrojenesisError:
            continue

    successor: dict[str, str] = {}
    predecessor: dict[str, str] = {}
    for layer_id, descriptor in einsums.items():
        output = str(descriptor["output"])
        output_consumers = consumers.get(output) or []
        if len(output_consumers) != 1 or output_consumers[0] not in einsums:
            continue
        consumer_id = output_consumers[0]
        if einsums[consumer_id]["input"] != output:
            continue
        successor[layer_id] = consumer_id
        predecessor[consumer_id] = layer_id

    chains: list[list[str]] = []
    for start in sorted(einsums):
        if start in predecessor or start not in successor:
            continue
        chain = [start]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        if len(chain) < 2:
            continue
        # The official composition drops intermediate traffic.  Restrict it
        # to complete graph endpoints and graph-input weights so that every
        # dropped access has an explicit producer-consumer witness.
        first = einsums[chain[0]]
        last = einsums[chain[-1]]
        external_names = [first["input"], *(einsums[item]["weight"] for item in chain)]
        if any(
            str(layers.get(producers.get(str(name), ""), {}).get("type", "")).lower()
            != "start"
            for name in external_names
        ):
            continue
        if consumers.get(str(last["output"])):
            continue
        try:
            multi_einsum_problem([(item, layers[item]) for item in chain])
        except OrojenesisError:
            continue
        chains.append(chain)
    return chains


def parse_multi_mapping_records(
    path: str | Path, *, word_bytes: int
) -> list[dict[str, Any]]:
    """Parse mapping-level OAVES fields used by the official fusion workflow."""
    source = Path(path)
    if not source.is_file():
        raise OrojenesisError(f"missing multi-einsum OAVES output: {source}")
    records: list[dict[str, Any]] = []
    with source.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 24:
                continue
            try:
                record: dict[str, Any] = {
                    "buffer_bytes": int(float(row[0])),
                    "dram_accesses_words": float(row[2]),
                    "mapping": str(row[3]),
                    "compute_ops": float(row[5]),
                    "weight_util_bytes": int(float(row[6])),
                    "input_util_bytes": int(float(row[10])),
                    "output_util_bytes": int(float(row[11])),
                    "weight_accesses_words": float(row[21]),
                    "input_accesses_words": float(row[22]),
                    "output_accesses_words": float(row[23]),
                }
            except ValueError:
                continue
            component_sum = sum(
                float(record[name])
                for name in (
                    "weight_accesses_words",
                    "input_accesses_words",
                    "output_accesses_words",
                )
            )
            if abs(component_sum - float(record["dram_accesses_words"])) > max(
                1e-6, component_sum * 1e-9
            ):
                raise OrojenesisError("multi-einsum OAVES access fields disagree")
            if (
                int(record["buffer_bytes"]) <= 0
                or float(record["compute_ops"]) <= 0
                or int(record["input_util_bytes"]) <= 0
                or int(record["output_util_bytes"]) <= 0
                or any(
                    float(value) < 0
                    for name, value in record.items()
                    if name.endswith("words")
                )
            ):
                raise OrojenesisError("multi-einsum OAVES record is invalid")
            record["dram_bytes"] = component_sum * int(word_bytes)
            records.append(record)
    if not records:
        raise OrojenesisError("multi-einsum OAVES output has no mapping records")
    return records


def compose_multi_einsum_curve(
    raw_paths: Sequence[Sequence[str | Path]],
    *,
    row_tiles: Sequence[int],
    word_bytes: int,
) -> list[dict[str, Any]]:
    """Compose compatible per-layer mappings into a replayable joint curve."""
    if len(raw_paths) < 2 or any(len(paths) != len(row_tiles) for paths in raw_paths):
        raise OrojenesisError("multi-einsum sweep matrix is incomplete")
    points: list[dict[str, Any]] = []
    for tile_index, row_tile in enumerate(row_tiles):
        per_layer = [
            parse_multi_mapping_records(paths[tile_index], word_bytes=word_bytes)
            for paths in raw_paths
        ]
        states: list[dict[str, Any]] = [
            {
                "buffer_bytes": int(record["buffer_bytes"]),
                "dram_accesses_words": float(record["weight_accesses_words"])
                + float(record["input_accesses_words"]),
                "compute_ops": float(record["compute_ops"]),
                "output_util_bytes": int(record["output_util_bytes"]),
                "mappings": [str(record["mapping"])],
                "row_tile": int(row_tile),
            }
            for record in per_layer[0]
        ]
        for layer_index, records in enumerate(per_layer[1:], start=1):
            next_states: dict[tuple[int, int], dict[str, Any]] = {}
            final_layer = layer_index == len(per_layer) - 1
            by_input: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                by_input[int(record["input_util_bytes"])].append(record)
            for state in states:
                for record in by_input.get(int(state["output_util_bytes"]), []):
                    accesses = float(state["dram_accesses_words"]) + float(
                        record["weight_accesses_words"]
                    )
                    compute_ops = float(state["compute_ops"]) + float(
                        record["compute_ops"]
                    )
                    if final_layer:
                        accesses += float(record["output_accesses_words"])
                    buffer_bytes = int(state["buffer_bytes"]) + int(
                        record["buffer_bytes"]
                    )
                    candidate: dict[str, Any] = {
                        "buffer_bytes": buffer_bytes,
                        "dram_accesses_words": accesses,
                        "compute_ops": compute_ops,
                        "output_util_bytes": int(record["output_util_bytes"]),
                        "mappings": [
                            *list(state["mappings"]),
                            str(record["mapping"]),
                        ],
                        "row_tile": int(row_tile),
                    }
                    key = (buffer_bytes, int(record["output_util_bytes"]))
                    previous = next_states.get(key)
                    if previous is None or accesses < float(
                        previous["dram_accesses_words"]
                    ):
                        next_states[key] = candidate
            states = list(next_states.values())
            if not states:
                break
        for state in states:
            accesses = float(state["dram_accesses_words"])
            points.append(
                {
                    "buffer_bytes": int(state["buffer_bytes"]),
                    "operational_intensity": (
                        0.0
                        if accesses == 0
                        else float(state["compute_ops"]) / (accesses * int(word_bytes))
                    ),
                    "dram_accesses_words": accesses,
                    "dram_bytes": accesses * int(word_bytes),
                    "row_tile": int(state["row_tile"]),
                    "mappings": list(state["mappings"]),
                }
            )
    if not points:
        raise OrojenesisError("multi-einsum sweeps contain no compatible tile path")
    best_by_capacity: dict[int, dict[str, Any]] = {}
    for point in points:
        capacity = int(point["buffer_bytes"])
        previous = best_by_capacity.get(capacity)
        if previous is None or float(point["dram_bytes"]) < float(
            previous["dram_bytes"]
        ):
            best_by_capacity[capacity] = point
    pareto: list[dict[str, Any]] = []
    best_traffic = float("inf")
    for point in sorted(
        best_by_capacity.values(), key=lambda item: item["buffer_bytes"]
    ):
        if float(point["dram_bytes"]) < best_traffic:
            pareto.append(point)
            best_traffic = float(point["dram_bytes"])
    return pareto


def parse_multi_einsum_curve(
    path: str | Path, *, word_bytes: int
) -> list[dict[str, Any]]:
    """Parse a serialized joint curve without trusting analysis.yaml fields."""
    source = Path(path)
    if not source.is_file():
        raise OrojenesisError(f"missing multi-einsum curve: {source}")
    points: list[dict[str, Any]] = []
    with source.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 5:
                continue
            try:
                accesses = float(row[2])
                mappings = json.loads(row[3])
                point = {
                    "buffer_bytes": int(float(row[0])),
                    "operational_intensity": float(row[1]),
                    "dram_accesses_words": accesses,
                    "dram_bytes": accesses * int(word_bytes),
                    "row_tile": int(row[4]),
                    "mappings": mappings,
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                point["buffer_bytes"] <= 0
                or point["dram_accesses_words"] < 0
                or point["row_tile"] <= 0
                or not isinstance(mappings, list)
                or any(not isinstance(item, str) for item in mappings)
            ):
                raise OrojenesisError("serialized multi-einsum curve is invalid")
            points.append(point)
    if not points:
        raise OrojenesisError("serialized multi-einsum curve has no valid points")
    return points


__all__ = [
    "MULTI_EINSUM_COMPOSITION",
    "MULTI_EINSUM_BATCH_COMPOSITION",
    "MULTI_EINSUM_FANOUT_COMPOSITION",
    "MULTI_EINSUM_LAYOUT_COMPOSITION",
    "MULTI_EINSUM_SOLVER",
    "OROJENESIS_COMMIT",
    "OROJENESIS_BUILDER_IMAGE",
    "OROJENESIS_COMPILER_WRAPPER_SHA256",
    "OROJENESIS_IDENTITY_SCHEMA_VERSION",
    "OROJENESIS_PROVENANCE_FILENAME",
    "OROJENESIS_REPOSITORY",
    "OROJENESIS_TREE_OID",
    "OrojenesisError",
    "OrojenesisRunner",
    "compose_multi_einsum_curve",
    "compose_multi_einsum_region_curve",
    "find_multi_einsum_chains",
    "find_multi_einsum_regions",
    "multi_einsum_layer_problem",
    "multi_einsum_mapper_role",
    "multi_einsum_problem",
    "multi_einsum_region_mapper_role",
    "multi_einsum_region_problem",
    "multi_einsum_row_tiles",
    "parse_multi_einsum_curve",
    "parse_multi_einsum_region_curve",
    "parse_multi_mapping_records",
    "select_capacity_point",
]
