# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated YAML contracts for ROCm benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
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
    resource_work: dict[str, dict[str, float]]
    lower_bound_seconds: float
    architecture_hash: str

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
        required = {
            "flops",
            "fused_bytes",
            "io_lower_bound_bytes",
            "macs_by_precision",
            "resource_work",
            "resource_seconds",
            "compute_resource",
            "lower_bound_seconds",
        }
        if not required.issubset(total):
            raise ValueError(
                "analysis total must contain flops, fused_bytes, and macs_by_precision"
            )
        macs_by_precision = {
            str(key).lower(): float(value)
            for key, value in (total.get("macs_by_precision") or {}).items()
        }
        from solar.analysis.resources import (
            RESOURCE_MODEL_VERSION,
            validate_resource_work,
        )

        resource_metadata = metadata.get("resource_model") or {}
        if resource_metadata.get("version") != RESOURCE_MODEL_VERSION:
            raise ValueError(
                f"benchmark analysis requires resource model {RESOURCE_MODEL_VERSION}"
            )
        coverage = resource_metadata.get("coverage") or {}
        if int(coverage.get("unclassified", -1)) != 0 or not bool(
            resource_metadata.get("fail_closed")
        ):
            raise ValueError("benchmark resource classification must be fail-closed")
        resource_work = validate_resource_work(total.get("resource_work"))
        serialized_resource_seconds = {
            str(key): float(value)
            for key, value in (total.get("resource_seconds") or {}).items()
        }
        flops = float(total["flops"])
        fused_bytes = float(total["fused_bytes"])
        io_lower_bound_bytes = float(total["io_lower_bound_bytes"])
        lower_bound_seconds = float(total["lower_bound_seconds"])
        if (
            not all(
                math.isfinite(value)
                for value in (
                    flops,
                    fused_bytes,
                    io_lower_bound_bytes,
                    lower_bound_seconds,
                    *macs_by_precision.values(),
                    *(
                        amount
                        for modes in resource_work.values()
                        for amount in modes.values()
                    ),
                    *serialized_resource_seconds.values(),
                )
            )
            or flops < 0
            or fused_bytes < 0
            or io_lower_bound_bytes < fused_bytes
            or lower_bound_seconds < 0
            or any(value < 0 for value in macs_by_precision.values())
            or any(value < 0 for value in serialized_resource_seconds.values())
        ):
            raise ValueError("analysis compute and traffic totals must be non-negative")
        if abs(2.0 * sum(macs_by_precision.values()) - flops) > max(
            1e-9, flops * 1e-12
        ):
            raise ValueError("analysis flops must equal twice macs_by_precision")

        if metadata.get("bound_kind") != "capacity_constrained_tile_aware_v1":
            raise ValueError(
                "benchmark scoring requires a formal capacity-constrained tile-aware bound"
            )

        # Hashes prove identity, while deterministic re-analysis proves the
        # graph-derived totals.  Orojenesis evidence is verified separately
        # against its pinned inputs and raw curve below.
        from solar.analysis.graph_analyzer import (
            EinsumGraphAnalyzer,
            contraction_operands_are_graph_external,
        )
        from solar.analysis.orojenesis import (
            MULTI_EINSUM_COMPOSITION,
            MULTI_EINSUM_BATCH_COMPOSITION,
            MULTI_EINSUM_FANOUT_COMPOSITION,
            MULTI_EINSUM_LAYOUT_COMPOSITION,
            MULTI_EINSUM_SOLVER,
            OROJENESIS_COMMIT,
            OROJENESIS_REPOSITORY,
            OrojenesisRunner,
            compose_multi_einsum_curve,
            compose_multi_einsum_region_curve,
            find_multi_einsum_chains,
            find_multi_einsum_regions,
            multi_einsum_layer_problem,
            multi_einsum_mapper_role,
            multi_einsum_problem,
            multi_einsum_region_mapper_role,
            multi_einsum_region_problem,
            multi_einsum_row_tiles,
            parse_multi_einsum_curve,
            parse_multi_einsum_region_curve,
            parse_multi_mapping_records,
            select_capacity_point,
        )
        from solar.rocm.architecture import ArchitectureProfile

        with tempfile.TemporaryDirectory(prefix="solar-analysis-verify-") as output:
            derived = EinsumGraphAnalyzer().analyze_graph(
                graph_path,
                output,
                precision=str(metadata.get("precision", "fp16")),
                copy_graph=False,
                strict=True,
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
            "resource_work": validate_resource_work(derived_total.get("resource_work")),
        }
        artifact_identity = {
            "flops": flops,
            "fused_bytes": fused_bytes,
            "macs_by_precision": macs_by_precision,
            "resource_work": resource_work,
        }
        if derived_identity != artifact_identity:
            raise ValueError("analysis totals drifted from the bound source graph")

        graph = yaml.safe_load(graph_path.read_text()) or {}
        layers = graph.get("layers") or {}
        einsum_layers = {
            str(layer_id): layer
            for layer_id, layer in layers.items()
            if (layer.get("semantic_op") or {}).get("kind") == "einsum"
        }
        solver = metadata.get("orojenesis") or {}
        if int(solver.get("schema_version", 0)) != 2:
            raise ValueError("formal analysis requires Orojenesis evidence schema 2")
        if einsum_layers and solver.get("status") != "complete":
            raise ValueError("formal analysis lacks complete Orojenesis evidence")
        if not einsum_layers and (
            solver.get("status") != "not_applicable"
            or solver.get("layers")
            or solver.get("chains")
            or solver.get("regions")
        ):
            raise ValueError(
                "non-einsum formal analysis has inconsistent solver status"
            )
        toolchain = solver.get("toolchain")
        if einsum_layers:
            if not isinstance(toolchain, dict):
                raise ValueError("formal analysis lacks Orojenesis toolchain identity")
            source_identity = toolchain.get("source") or {}
            artifact_identity = toolchain.get("artifact") or {}
            if (
                int(toolchain.get("schema_version", 0)) != 1
                or source_identity.get("repository") != OROJENESIS_REPOSITORY
                or source_identity.get("commit") != OROJENESIS_COMMIT
                or not re.fullmatch(
                    r"[0-9a-f]{40,64}", str(source_identity.get("tree_git_oid", ""))
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(source_identity.get("archive_sha256", "")),
                )
                or artifact_identity.get("path") != "bin/timeloop-mapper"
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(artifact_identity.get("sha256", ""))
                )
            ):
                raise ValueError("invalid Orojenesis toolchain identity")

        fusion = derived_metadata.get("fusion") or {}
        region_by_layer = {
            str(layer_id): region
            for region in fusion.get("regions") or []
            for layer_id in region.get("layers") or []
        }
        solver_excesses: list[float] = []
        applicable_layer_count = 0
        recorded_layers = solver.get("layers") or {}
        recorded_chains = solver.get("chains") or {}
        recorded_regions = solver.get("regions") or {}
        expected_chain_ids = {
            f"chain_{index}": chain
            for index, chain in enumerate(find_multi_einsum_chains(layers))
        }
        if set(recorded_chains) != set(expected_chain_ids):
            raise ValueError("Orojenesis multi-einsum chain set drifted")
        expected_region_ids = {
            f"region_{index}": region
            for index, region in enumerate(find_multi_einsum_regions(layers))
        }
        if set(recorded_regions) != set(expected_region_ids):
            raise ValueError("Orojenesis multi-einsum region set drifted")
        recorded_chain_members = {
            layer_id for chain in expected_chain_ids.values() for layer_id in chain
        }
        recorded_region_members = {
            str(layer_id)
            for region in expected_region_ids.values()
            for layer_id in region.get("schedule") or []
        }
        multi_members = recorded_chain_members | recorded_region_members
        if set(recorded_layers) & multi_members or (
            set(recorded_layers) | multi_members
        ) != set(einsum_layers):
            raise ValueError("Orojenesis evidence does not cover every einsum layer")
        for layer_id, result in recorded_layers.items():
            layer = einsum_layers[layer_id]
            if result.get("solver") != "NVlabs/timeloop oaves_keep_max":
                raise ValueError(f"Orojenesis solver identity mismatch for {layer_id}")
            if result.get("commit") != OROJENESIS_COMMIT:
                raise ValueError(f"Orojenesis revision mismatch for {layer_id}")
            if result.get("toolchain") != toolchain:
                raise ValueError(f"Orojenesis toolchain drifted for {layer_id}")
            word_bits = int(result.get("word_bits", 0))
            if word_bits <= 0 or word_bits % 8:
                raise ValueError(f"invalid Orojenesis word width for {layer_id}")
            evidence_files = result.get("evidence_files") or {}
            required_files = {
                "problem.yaml",
                "architecture.yaml",
                "mapper.yaml",
                "curve",
            }
            if set(evidence_files) != required_files:
                raise ValueError(f"incomplete Orojenesis evidence files for {layer_id}")
            resolved_files: dict[str, Path] = {}
            for name, evidence in evidence_files.items():
                relative = _relative_path(
                    str((evidence or {}).get("path", "")),
                    f"orojenesis.{layer_id}.{name}.path",
                )
                candidate = (analysis_path.parent / relative).resolve()
                if (
                    analysis_path.parent not in candidate.parents
                    or not candidate.is_file()
                ):
                    raise ValueError(f"Orojenesis evidence file is missing: {relative}")
                digest = cls._parse_sha256(
                    (evidence or {}).get("sha256"),
                    f"orojenesis.{layer_id}.{name}.sha256",
                )
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
                    raise ValueError(
                        f"Orojenesis evidence SHA-256 mismatch: {relative}"
                    )
                resolved_files[name] = candidate

            expected_problem = OrojenesisRunner.problem_for_layer(layer)
            if (
                yaml.safe_load(resolved_files["problem.yaml"].read_text())
                != expected_problem
            ):
                raise ValueError(f"Orojenesis problem drifted from layer {layer_id}")
            if yaml.safe_load(
                resolved_files["architecture.yaml"].read_text()
            ) != OrojenesisRunner.architecture(word_bits):
                raise ValueError(f"Orojenesis architecture drifted for {layer_id}")
            dimensions = list(expected_problem["problem"]["shape"]["dimensions"])
            spaces = [
                item["name"]
                for item in expected_problem["problem"]["shape"]["data-spaces"]
            ]
            if yaml.safe_load(
                resolved_files["mapper.yaml"].read_text()
            ) != OrojenesisRunner.mapper_config(dimensions, spaces):
                raise ValueError(f"Orojenesis mapper drifted for {layer_id}")
            curve = OrojenesisRunner.parse_curve(
                resolved_files["curve"], word_bytes=word_bits // 8
            )
            if curve != result.get("curve"):
                raise ValueError(f"Orojenesis curve drifted for {layer_id}")
            selected = result.get("selected_capacity") or {}
            point = select_capacity_point(curve, int(selected.get("capacity_bytes", 0)))
            if point is None or point != selected.get("point"):
                raise ValueError(f"Orojenesis capacity point is invalid for {layer_id}")
            applicability = result.get("formal_applicability") or {}
            expected_applicable = contraction_operands_are_graph_external(layer, layers)
            if bool(applicability.get("applicable")) != expected_applicable:
                raise ValueError(f"Orojenesis applicability drifted for {layer_id}")
            expected_provenance = (
                "graph_input_or_recomputable_preprocess"
                if expected_applicable
                else "internal"
            )
            expected_reason = (
                "graph_input_or_recomputable_preprocess_contraction"
                if expected_applicable
                else "internal_operand_requires_multi_einsum_composition"
            )
            legacy_reason = (
                "graph_input_contraction" if expected_applicable else expected_reason
            )
            if applicability.get("operand_provenance") not in {
                None,
                expected_provenance,
            } or applicability.get("reason") not in {expected_reason, legacy_reason}:
                raise ValueError(
                    f"Orojenesis applicability reason drifted for {layer_id}"
                )
            region = region_by_layer.get(layer_id)
            if region is None or applicability.get("region") != region.get("id"):
                raise ValueError(f"Orojenesis fusion region mismatch for {layer_id}")
            solver_bytes = float(point["dram_bytes"])
            if float(result.get("audited_dram_bytes", -1)) != solver_bytes:
                if applicability.get("applicable") is True:
                    raise ValueError(
                        f"Orojenesis audited traffic drifted for {layer_id}"
                    )
            if applicability.get("applicable") is True:
                word_bytes = word_bits // 8
                names = layer.get("tensor_names") or {}
                shapes = layer.get("tensor_shapes") or {}
                modeled_tensors: dict[str, list[int]] = {}
                for side in ("inputs", "outputs"):
                    for name, shape in zip(
                        names.get(side) or [], shapes.get(side) or []
                    ):
                        modeled_tensors[str(name)] = list(shape)
                compulsory_bytes = float(
                    sum(math.prod(shape) for shape in modeled_tensors.values())
                    * word_bytes
                )
                if (
                    float(result.get("modeled_compulsory_bytes", -1))
                    != compulsory_bytes
                ):
                    raise ValueError(
                        f"Orojenesis compulsory traffic drifted for {layer_id}"
                    )
                solver_excesses.append(max(0.0, solver_bytes - compulsory_bytes))
                applicable_layer_count += 1
            elif applicability.get("reason") != expected_reason:
                raise ValueError(f"invalid Orojenesis applicability for {layer_id}")

        for chain_id, layer_ids in expected_chain_ids.items():
            result = recorded_chains[chain_id]
            if result.get("solver") != MULTI_EINSUM_SOLVER:
                raise ValueError(
                    f"multi-einsum solver identity mismatch for {chain_id}"
                )
            if result.get("commit") != OROJENESIS_COMMIT:
                raise ValueError(f"Orojenesis revision mismatch for {chain_id}")
            if result.get("toolchain") != toolchain:
                raise ValueError(f"Orojenesis toolchain drifted for {chain_id}")
            if result.get("composition") != MULTI_EINSUM_COMPOSITION:
                raise ValueError(
                    f"multi-einsum composition identity mismatch for {chain_id}"
                )
            word_bits = int(result.get("word_bits", 0))
            if word_bits <= 0 or word_bits % 8:
                raise ValueError(f"invalid Orojenesis word width for {chain_id}")
            word_bytes = word_bits // 8
            chain_layers = [
                (layer_id, einsum_layers[layer_id]) for layer_id in layer_ids
            ]
            expected_problem = multi_einsum_problem(chain_layers)
            if result.get("problem") != expected_problem:
                raise ValueError(f"multi-einsum problem drifted for {chain_id}")
            descriptors = expected_problem["chain"]["layers"]
            row_tiles = multi_einsum_row_tiles(int(descriptors[0]["m"]))
            expected_sweeps = [
                {
                    "layer_id": layer_id,
                    "row_tiles": row_tiles,
                    "role": multi_einsum_mapper_role(index, len(layer_ids)),
                }
                for index, layer_id in enumerate(layer_ids)
            ]
            if result.get("sweeps") != expected_sweeps:
                raise ValueError(f"multi-einsum sweep set drifted for {chain_id}")

            expected_environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
            if result.get("environment") != expected_environment:
                raise ValueError(
                    f"multi-einsum solver environment drifted for {chain_id}"
                )
            required_files = {
                "chain.yaml",
                "architecture.yaml",
                "environment.yaml",
                "curve",
            }
            for layer_index in range(len(layer_ids)):
                required_files.add(f"problem-layer-{layer_index}.yaml")
                for row_tile in row_tiles:
                    required_files.update(
                        {
                            f"layer-{layer_index}-m-{row_tile}-architecture.yaml",
                            f"layer-{layer_index}-m-{row_tile}-mapper.yaml",
                            f"layer-{layer_index}-m-{row_tile}-problem.yaml",
                            f"layer-{layer_index}-m-{row_tile}-raw",
                        }
                    )
            evidence_files = result.get("evidence_files") or {}
            if set(evidence_files) != required_files:
                raise ValueError(
                    f"incomplete multi-einsum evidence files for {chain_id}"
                )
            chain_files: dict[str, Path] = {}
            for name, evidence in evidence_files.items():
                relative = _relative_path(
                    str((evidence or {}).get("path", "")),
                    f"orojenesis.{chain_id}.{name}.path",
                )
                candidate = (analysis_path.parent / relative).resolve()
                if (
                    analysis_path.parent not in candidate.parents
                    or not candidate.is_file()
                ):
                    raise ValueError(
                        f"multi-einsum evidence file is missing: {relative}"
                    )
                digest = cls._parse_sha256(
                    (evidence or {}).get("sha256"),
                    f"orojenesis.{chain_id}.{name}.sha256",
                )
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
                    raise ValueError(
                        f"multi-einsum evidence SHA-256 mismatch: {relative}"
                    )
                chain_files[name] = candidate

            if (
                yaml.safe_load(chain_files["chain.yaml"].read_text())
                != expected_problem
            ):
                raise ValueError(f"multi-einsum chain file drifted for {chain_id}")
            expected_architecture = OrojenesisRunner.multi_architecture(word_bits)
            if (
                yaml.safe_load(chain_files["architecture.yaml"].read_text())
                != expected_architecture
            ):
                raise ValueError(f"multi-einsum architecture drifted for {chain_id}")
            if (
                yaml.safe_load(chain_files["environment.yaml"].read_text())
                != expected_environment
            ):
                raise ValueError(
                    f"multi-einsum environment evidence drifted for {chain_id}"
                )

            chain_raw_paths: list[list[Path]] = []
            for layer_index, descriptor in enumerate(descriptors):
                expected_layer_problem = multi_einsum_layer_problem(descriptor)
                problem_name = f"problem-layer-{layer_index}.yaml"
                if (
                    yaml.safe_load(chain_files[problem_name].read_text())
                    != expected_layer_problem
                ):
                    raise ValueError(
                        f"multi-einsum layer problem drifted for {chain_id}"
                    )
                layer_raw_paths: list[Path] = []
                for row_tile in row_tiles:
                    prefix = f"layer-{layer_index}-m-{row_tile}"
                    if (
                        yaml.safe_load(
                            chain_files[f"{prefix}-architecture.yaml"].read_text()
                        )
                        != expected_architecture
                    ):
                        raise ValueError(
                            f"multi-einsum sweep architecture drifted for {chain_id}"
                        )
                    expected_mapper = OrojenesisRunner.multi_mapper_config(
                        row_tile,
                        role=multi_einsum_mapper_role(layer_index, len(layer_ids)),
                    )
                    if (
                        yaml.safe_load(chain_files[f"{prefix}-mapper.yaml"].read_text())
                        != expected_mapper
                    ):
                        raise ValueError(f"multi-einsum mapper drifted for {chain_id}")
                    if (
                        yaml.safe_load(
                            chain_files[f"{prefix}-problem.yaml"].read_text()
                        )
                        != expected_layer_problem
                    ):
                        raise ValueError(
                            f"multi-einsum sweep problem drifted for {chain_id}"
                        )
                    raw_path = chain_files[f"{prefix}-raw"]
                    parse_multi_mapping_records(raw_path, word_bytes=word_bytes)
                    layer_raw_paths.append(raw_path)
                chain_raw_paths.append(layer_raw_paths)

            recomposed_curve = compose_multi_einsum_curve(
                chain_raw_paths, row_tiles=row_tiles, word_bytes=word_bytes
            )
            if recomposed_curve != result.get("curve"):
                raise ValueError(f"multi-einsum curve drifted for {chain_id}")
            serialized_curve = parse_multi_einsum_curve(
                chain_files["curve"], word_bytes=word_bytes
            )
            if serialized_curve != recomposed_curve:
                raise ValueError(
                    f"serialized multi-einsum curve drifted for {chain_id}"
                )
            selected = result.get("selected_capacity") or {}
            point = select_capacity_point(
                recomposed_curve, int(selected.get("capacity_bytes", 0))
            )
            if point is None or point != selected.get("point"):
                raise ValueError(
                    f"multi-einsum capacity point is invalid for {chain_id}"
                )
            region_ids = {
                str(region_by_layer[layer_id]["id"])
                for layer_id in layer_ids
                if layer_id in region_by_layer
            }
            applicability = result.get("formal_applicability") or {}
            if (
                applicability.get("applicable") is not True
                or applicability.get("layer_ids") != layer_ids
                or len(region_ids) != 1
                or applicability.get("region") != next(iter(region_ids))
                or applicability.get("operand_provenance")
                != "graph_inputs_and_internal_chain_edges"
                or applicability.get("reason") != "verified_linear_matmul_tiled_fusion"
            ):
                raise ValueError(f"multi-einsum applicability drifted for {chain_id}")
            first = descriptors[0]
            last = descriptors[-1]
            compulsory_elements = int(first["m"]) * int(first["k"])
            compulsory_elements += sum(
                int(item["k"]) * int(item["n"]) for item in descriptors
            )
            compulsory_elements += int(last["m"]) * int(last["n"])
            compulsory_bytes = float(compulsory_elements * word_bytes)
            solver_bytes = float(point["dram_bytes"])
            if float(result.get("audited_dram_bytes", -1)) != solver_bytes:
                raise ValueError(f"multi-einsum audited traffic drifted for {chain_id}")
            if float(result.get("modeled_compulsory_bytes", -1)) != compulsory_bytes:
                raise ValueError(
                    f"multi-einsum compulsory traffic drifted for {chain_id}"
                )
            solver_excesses.append(max(0.0, solver_bytes - compulsory_bytes))
            applicable_layer_count += len(layer_ids)

        supported_region_compositions = {
            MULTI_EINSUM_LAYOUT_COMPOSITION,
            MULTI_EINSUM_BATCH_COMPOSITION,
            MULTI_EINSUM_FANOUT_COMPOSITION,
        }
        for region_id, expected_region in expected_region_ids.items():
            result = recorded_regions[region_id]
            if result.get("solver") != MULTI_EINSUM_SOLVER:
                raise ValueError(
                    f"multi-einsum region solver identity mismatch for {region_id}"
                )
            if result.get("commit") != OROJENESIS_COMMIT:
                raise ValueError(f"Orojenesis revision mismatch for {region_id}")
            if result.get("toolchain") != toolchain:
                raise ValueError(f"Orojenesis toolchain drifted for {region_id}")
            if result.get("composition") not in supported_region_compositions:
                raise ValueError(
                    f"multi-einsum region composition mismatch for {region_id}"
                )
            word_bits = int(result.get("word_bits", 0))
            if word_bits <= 0 or word_bits % 8:
                raise ValueError(
                    f"invalid multi-einsum region word width for {region_id}"
                )
            word_bytes = word_bits // 8
            expected_problem = multi_einsum_region_problem(expected_region)
            if result.get("problem") != expected_problem:
                raise ValueError(f"multi-einsum region problem drifted for {region_id}")
            descriptors = expected_problem["nodes"]
            expected_sweeps = [
                {
                    "node_id": str(descriptor["id"]),
                    "row_tiles": multi_einsum_row_tiles(int(descriptor["m"])),
                    "role": multi_einsum_region_mapper_role(
                        expected_problem, str(descriptor["id"])
                    ),
                }
                for descriptor in descriptors
            ]
            if result.get("sweeps") != expected_sweeps:
                raise ValueError(
                    f"multi-einsum region sweep set drifted for {region_id}"
                )
            expected_environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
            if result.get("environment") != expected_environment:
                raise ValueError(
                    f"multi-einsum region environment drifted for {region_id}"
                )
            required_files = {
                "region.yaml",
                "architecture.yaml",
                "environment.yaml",
                "curve",
            }
            row_tiles_by_node: dict[str, list[int]] = {}
            for node_index, descriptor in enumerate(descriptors):
                node_id = str(descriptor["id"])
                row_tiles = multi_einsum_row_tiles(int(descriptor["m"]))
                row_tiles_by_node[node_id] = row_tiles
                required_files.add(f"problem-node-{node_index}.yaml")
                for row_tile in row_tiles:
                    prefix = f"node-{node_index}-m-{row_tile}"
                    required_files.update(
                        {
                            f"{prefix}-architecture.yaml",
                            f"{prefix}-mapper.yaml",
                            f"{prefix}-problem.yaml",
                            f"{prefix}-raw",
                        }
                    )
            evidence_files = result.get("evidence_files") or {}
            if set(evidence_files) != required_files:
                raise ValueError(
                    f"incomplete multi-einsum region evidence for {region_id}"
                )
            region_files: dict[str, Path] = {}
            for name, evidence in evidence_files.items():
                relative = _relative_path(
                    str((evidence or {}).get("path", "")),
                    f"orojenesis.{region_id}.{name}.path",
                )
                candidate = (analysis_path.parent / relative).resolve()
                if (
                    analysis_path.parent not in candidate.parents
                    or not candidate.is_file()
                ):
                    raise ValueError(
                        f"multi-einsum region evidence is missing: {relative}"
                    )
                digest = cls._parse_sha256(
                    (evidence or {}).get("sha256"),
                    f"orojenesis.{region_id}.{name}.sha256",
                )
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
                    raise ValueError(
                        f"multi-einsum region evidence SHA-256 mismatch: {relative}"
                    )
                region_files[name] = candidate
            if (
                yaml.safe_load(region_files["region.yaml"].read_text())
                != expected_problem
            ):
                raise ValueError(f"multi-einsum region file drifted for {region_id}")
            expected_architecture = OrojenesisRunner.multi_architecture(word_bits)
            if (
                yaml.safe_load(region_files["architecture.yaml"].read_text())
                != expected_architecture
            ):
                raise ValueError(
                    f"multi-einsum region architecture drifted for {region_id}"
                )
            if (
                yaml.safe_load(region_files["environment.yaml"].read_text())
                != expected_environment
            ):
                raise ValueError(
                    f"multi-einsum region environment evidence drifted for {region_id}"
                )
            region_raw_paths: dict[str, list[Path]] = {}
            for node_index, descriptor in enumerate(descriptors):
                node_id = str(descriptor["id"])
                layer_problem = multi_einsum_layer_problem(descriptor)
                problem_name = f"problem-node-{node_index}.yaml"
                if (
                    yaml.safe_load(region_files[problem_name].read_text())
                    != layer_problem
                ):
                    raise ValueError(
                        f"multi-einsum region node problem drifted for {region_id}"
                    )
                node_raw_paths: list[Path] = []
                role = multi_einsum_region_mapper_role(expected_problem, node_id)
                for row_tile in row_tiles_by_node[node_id]:
                    prefix = f"node-{node_index}-m-{row_tile}"
                    if (
                        yaml.safe_load(
                            region_files[f"{prefix}-architecture.yaml"].read_text()
                        )
                        != expected_architecture
                    ):
                        raise ValueError(
                            f"multi-einsum region sweep architecture drifted for {region_id}"
                        )
                    expected_mapper = OrojenesisRunner.multi_mapper_config(
                        row_tile, role=role
                    )
                    if (
                        yaml.safe_load(
                            region_files[f"{prefix}-mapper.yaml"].read_text()
                        )
                        != expected_mapper
                    ):
                        raise ValueError(
                            f"multi-einsum region mapper drifted for {region_id}"
                        )
                    if (
                        yaml.safe_load(
                            region_files[f"{prefix}-problem.yaml"].read_text()
                        )
                        != layer_problem
                    ):
                        raise ValueError(
                            f"multi-einsum region sweep problem drifted for {region_id}"
                        )
                    raw_path = region_files[f"{prefix}-raw"]
                    parse_multi_mapping_records(raw_path, word_bytes=word_bytes)
                    node_raw_paths.append(raw_path)
                region_raw_paths[node_id] = node_raw_paths
            recomposed_curve = compose_multi_einsum_region_curve(
                expected_problem,
                region_raw_paths,
                row_tiles_by_node=row_tiles_by_node,
                word_bytes=word_bytes,
            )
            if recomposed_curve != result.get("curve"):
                raise ValueError(f"multi-einsum region curve drifted for {region_id}")
            serialized_curve = parse_multi_einsum_region_curve(
                region_files["curve"], word_bytes=word_bytes
            )
            if serialized_curve != recomposed_curve:
                raise ValueError(
                    f"serialized multi-einsum region curve drifted for {region_id}"
                )
            selected = result.get("selected_capacity") or {}
            point = select_capacity_point(
                recomposed_curve, int(selected.get("capacity_bytes", 0))
            )
            if point is None or point != selected.get("point"):
                raise ValueError(
                    f"multi-einsum region capacity point is invalid for {region_id}"
                )
            layer_ids = [str(item) for item in expected_problem["schedule"]]
            fusion_region_ids = {
                str(region_by_layer[layer_id]["id"])
                for layer_id in layer_ids
                if layer_id in region_by_layer
            }
            applicability = result.get("formal_applicability") or {}
            if (
                applicability.get("applicable") is not True
                or applicability.get("layer_ids") != layer_ids
                or len(fusion_region_ids) != 1
                or applicability.get("region") != next(iter(fusion_region_ids))
                or applicability.get("operand_provenance")
                != "graph_inputs_and_verified_internal_region_edges"
                or applicability.get("reason") != "verified_matmul_region_tiled_fusion"
            ):
                raise ValueError(
                    f"multi-einsum region applicability drifted for {region_id}"
                )
            by_id = {str(item["id"]): item for item in descriptors}
            compulsory_elements = sum(
                int(by_id[root]["m"]) * int(by_id[root]["k"])
                for root in expected_problem["roots"]
            )
            compulsory_elements += sum(
                int(item["k"]) * int(item["n"]) for item in descriptors
            )
            compulsory_elements += sum(
                int(by_id[leaf]["m"]) * int(by_id[leaf]["n"])
                for leaf in expected_problem["leaves"]
            )
            compulsory_bytes = float(compulsory_elements * word_bytes)
            solver_bytes = float(point["dram_bytes"])
            if float(result.get("audited_dram_bytes", -1)) != solver_bytes:
                raise ValueError(
                    f"multi-einsum region audited traffic drifted for {region_id}"
                )
            if float(result.get("modeled_compulsory_bytes", -1)) != compulsory_bytes:
                raise ValueError(
                    f"multi-einsum region compulsory traffic drifted for {region_id}"
                )
            solver_excesses.append(max(0.0, solver_bytes - compulsory_bytes))
            applicable_layer_count += len(layer_ids)

        expected_io_bytes = fused_bytes + max(solver_excesses, default=0.0)
        expected_coverage = {
            "applicable_layers": applicable_layer_count,
            "total_layers": len(einsum_layers),
        }
        if einsum_layers and solver.get("formal_coverage") != expected_coverage:
            raise ValueError("Orojenesis formal coverage drifted")
        if einsum_layers and not solver_excesses:
            raise ValueError("formal analysis has no composable tile-aware layer")
        if io_lower_bound_bytes != expected_io_bytes:
            raise ValueError(
                "analysis I/O lower bound drifted from Orojenesis evidence"
            )
        profile = ArchitectureProfile.load(metadata.get("architecture") or {})
        expected_resource_seconds = profile.resource_seconds(resource_work)
        if serialized_resource_seconds != expected_resource_seconds:
            raise ValueError(
                "analysis resource seconds drifted from architecture limits"
            )
        expected_compute_resource = (
            max(
                sorted(expected_resource_seconds),
                key=expected_resource_seconds.__getitem__,
            )
            if expected_resource_seconds
            else None
        )
        if total.get("compute_resource") != expected_compute_resource:
            raise ValueError("analysis compute resource bottleneck drifted")
        expected_seconds = profile.theoretical_seconds_by_resources(
            resource_work, expected_io_bytes
        )
        if lower_bound_seconds != expected_seconds:
            raise ValueError("analysis time bound drifted from audited I/O evidence")
        architecture_identity = profile.to_dict()
        architecture_identity.pop("source", None)
        return cls(
            path=path,
            sha256=expected_analysis_sha,
            source_graph=source_graph,
            source_graph_sha256=expected_graph_sha,
            flops=flops,
            fused_bytes=fused_bytes,
            macs_by_precision=macs_by_precision,
            resource_work=resource_work,
            lower_bound_seconds=lower_bound_seconds,
            architecture_hash=canonical_hash(architecture_identity),
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
        required_matched_ratio: float,
        max_error_cap: float | None,
        allow_negative_inf: bool,
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
            required_matched_ratio=required_matched_ratio,
            max_error_cap=max_error_cap,
            allow_negative_inf=allow_negative_inf,
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
        if int(artifact.get("schema_version", 0)) != 2:
            raise ValueError("compatibility artifact must use schema_version=2")
        if artifact.get("fallbacks_used") != []:
            raise ValueError("compatibility artifact must not use fallbacks")
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
        if status == "compatible" and reason_code != "compatible":
            raise ValueError("compatible evidence has an inconsistent reason_code")
        if status != "compatible" and reason_code == "compatible":
            raise ValueError("non-compatible evidence has an inconsistent reason_code")
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
    required_matched_ratio: float = 1.0
    max_error_cap: float | None = None
    allow_negative_inf: bool = False

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
        required_matched_ratio = float(
            local_tolerance.get("required_matched_ratio", 1.0)
        )
        max_error_cap_raw = local_tolerance.get("max_error_cap")
        max_error_cap = (
            float(max_error_cap_raw) if max_error_cap_raw is not None else None
        )
        allow_negative_inf = bool(local_tolerance.get("allow_negative_inf", False))
        if not all(
            math.isfinite(value)
            for value in (workload_atol, workload_rtol, required_matched_ratio)
        ):
            raise ValueError("workload tolerances must be finite")
        if workload_atol < 0 or workload_rtol < 0:
            raise ValueError("workload tolerances must be non-negative")
        if not 0.0 <= required_matched_ratio <= 1.0:
            raise ValueError("required_matched_ratio must be between zero and one")
        if max_error_cap is not None and (
            not math.isfinite(max_error_cap) or max_error_cap < 0
        ):
            raise ValueError("max_error_cap must be finite and non-negative")
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
            status_evidence = CompatibilityArtifact.load(
                compatibility_data, source_root
            )
            if status_evidence.status != status:
                raise ValueError(
                    "workload status disagrees with compatibility artifact"
                )
            return cls(
                name=name,
                status=status,
                uuid=str(data.get("uuid", "")) or None,
                parameters=parameters,
                compatibility=status_evidence,
                atol=workload_atol,
                rtol=workload_rtol,
                required_matched_ratio=required_matched_ratio,
                max_error_cap=max_error_cap,
                allow_negative_inf=allow_negative_inf,
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
                required_matched_ratio=required_matched_ratio,
                max_error_cap=max_error_cap,
                allow_negative_inf=allow_negative_inf,
            )
        else:
            raise ValueError(
                "compatible workloads require a replayable verification artifact"
            )
        compatibility: CompatibilityArtifact | None = None
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
            required_matched_ratio=required_matched_ratio,
            max_error_cap=max_error_cap,
            allow_negative_inf=allow_negative_inf,
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
        if not math.isfinite(self.atol) or not math.isfinite(self.rtol):
            raise ValueError("tolerances must be finite")
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
