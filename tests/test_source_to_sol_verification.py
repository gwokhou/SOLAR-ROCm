# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from solar.analysis.graph_analyzer import (
    EinsumGraphAnalyzer,
    contraction_operands_are_graph_external,
)
from solar.analysis.orojenesis import (
    MULTI_EINSUM_COMPOSITION,
    MULTI_EINSUM_SOLVER,
    OROJENESIS_COMMIT,
    OrojenesisRunner,
    compose_multi_einsum_curve,
    compose_multi_einsum_region_curve,
    multi_einsum_layer_problem,
    multi_einsum_mapper_role,
    multi_einsum_problem,
    multi_einsum_region_mapper_role,
    multi_einsum_region_problem,
    multi_einsum_row_tiles,
)
from solar.benchmark.models import AnalysisArtifact, BenchmarkSpec, canonical_hash
from solar.einsum import ConversionError, PyTorchToEinsum, annotate_semantics
from solar.rocm import ArchitectureProfile
from solar.verification import (
    VerificationError,
    create_verification_artifact,
    replay_verification_artifact,
)
from solar.verification.einsum import _assert_close


class _DeterministicOrojenesisRunner:
    """Small evidence-producing stand-in for the pinned external binary."""

    def __init__(self, minimum_accesses: int = 80):
        self.minimum_accesses = minimum_accesses
        self.toolchain_identity = {
            "schema_version": 1,
            "verification_mode": "test_fixture",
            "source": {
                "repository": "https://github.com/NVlabs/timeloop.git",
                "commit": OROJENESIS_COMMIT,
                "tree_git_oid": "1" * 40,
                "archive_sha256": "2" * 64,
            },
            "artifact": {
                "path": "bin/timeloop-mapper",
                "sha256": "3" * 64,
            },
        }

    def run_layer(self, layer, output_dir, *, word_bits):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        problem = OrojenesisRunner.problem_for_layer(layer)
        dimensions = list(problem["problem"]["shape"]["dimensions"])
        spaces = [item["name"] for item in problem["problem"]["shape"]["data-spaces"]]
        documents = {
            "problem.yaml": problem,
            "architecture.yaml": OrojenesisRunner.architecture(word_bits),
            "mapper.yaml": OrojenesisRunner.mapper_config(dimensions, spaces),
        }
        for name, value in documents.items():
            (output_dir / name).write_text(yaml.safe_dump(value, sort_keys=False))
        curve_path = output_dir / "timeloop-mapper.oaves.csv"
        curve_path.write_text(
            f"64,2,{self.minimum_accesses + 20},map,None,10\n"
            f"128,4,{self.minimum_accesses},map,None,10\n"
        )
        evidence_files = {
            name: {
                "path": name,
                "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
            }
            for name in documents
        }
        evidence_files["curve"] = {
            "path": curve_path.name,
            "sha256": hashlib.sha256(curve_path.read_bytes()).hexdigest(),
        }
        return {
            "solver": "NVlabs/timeloop oaves_keep_max",
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "word_bits": word_bits,
            "curve": OrojenesisRunner.parse_curve(
                curve_path, word_bytes=word_bits // 8
            ),
            "evidence_files": evidence_files,
        }

    @staticmethod
    def _mapping_row(
        descriptor: dict,
        *,
        word_bytes: int,
        row_tile: int,
        layer_index: int,
    ) -> str:
        m_size = int(descriptor["m"])
        k_size = int(descriptor["k"])
        n_size = int(descriptor["n"])
        weight_accesses = k_size * n_size + 100 + layer_index
        input_accesses = m_size * k_size
        output_accesses = m_size * n_size
        row: list[object] = [0] * 24
        row[0] = (row_tile * (k_size + n_size) + k_size * n_size) * word_bytes
        row[1] = 1
        row[2] = weight_accesses + input_accesses + output_accesses
        row[3] = f"layer-{layer_index}-m-{row_tile}"
        row[5] = m_size * k_size * n_size
        row[6] = k_size * n_size * word_bytes
        row[10] = row_tile * k_size * word_bytes
        row[11] = row_tile * n_size * word_bytes
        row[21] = weight_accesses
        row[22] = input_accesses
        row[23] = output_accesses
        return ",".join(str(value) for value in row) + "\n"

    def run_multi_chain(self, chain, output_dir, *, word_bits):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        word_bytes = word_bits // 8
        problem = multi_einsum_problem(chain)
        descriptors = problem["chain"]["layers"]
        row_tiles = multi_einsum_row_tiles(int(descriptors[0]["m"]))
        architecture = OrojenesisRunner.multi_architecture(word_bits)
        environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
        paths: dict[str, Path] = {}

        def write_yaml(name: str, value: dict) -> Path:
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(value, sort_keys=False))
            paths[name] = path
            return path

        write_yaml("chain.yaml", problem)
        write_yaml("architecture.yaml", architecture)
        write_yaml("environment.yaml", environment)
        raw_paths: list[list[Path]] = []
        sweeps: list[dict] = []
        for layer_index, descriptor in enumerate(descriptors):
            layer_problem = multi_einsum_layer_problem(descriptor)
            write_yaml(f"problem-layer-{layer_index}.yaml", layer_problem)
            layer_raw_paths: list[Path] = []
            for row_tile in row_tiles:
                prefix = f"layer-{layer_index}-m-{row_tile}"
                write_yaml(f"{prefix}-architecture.yaml", architecture)
                write_yaml(
                    f"{prefix}-mapper.yaml",
                    OrojenesisRunner.multi_mapper_config(
                        row_tile,
                        role=multi_einsum_mapper_role(layer_index, len(descriptors)),
                    ),
                )
                write_yaml(f"{prefix}-problem.yaml", layer_problem)
                raw_path = output_dir / prefix / "timeloop-mapper.oaves.csv"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    self._mapping_row(
                        descriptor,
                        word_bytes=word_bytes,
                        row_tile=row_tile,
                        layer_index=layer_index,
                    )
                )
                paths[f"{prefix}-raw"] = raw_path
                layer_raw_paths.append(raw_path)
            raw_paths.append(layer_raw_paths)
            sweeps.append(
                {
                    "layer_id": str(descriptor["id"]),
                    "row_tiles": row_tiles,
                    "role": multi_einsum_mapper_role(layer_index, len(descriptors)),
                }
            )
        curve = compose_multi_einsum_curve(
            raw_paths, row_tiles=row_tiles, word_bytes=word_bytes
        )
        curve_path = output_dir / "multi-einsum-curve.csv"
        with curve_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for point in curve:
                writer.writerow(
                    [
                        point["buffer_bytes"],
                        point["operational_intensity"],
                        point["dram_accesses_words"],
                        json.dumps(point["mappings"], separators=(",", ":")),
                        point["row_tile"],
                    ]
                )
        paths["curve"] = curve_path
        return {
            "solver": MULTI_EINSUM_SOLVER,
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "composition": MULTI_EINSUM_COMPOSITION,
            "word_bits": word_bits,
            "environment": environment,
            "problem": problem,
            "sweeps": sweeps,
            "curve": curve,
            "evidence_files": {
                name: {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in paths.items()
            },
        }

    def run_multi_region(self, region, output_dir, *, word_bits):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        word_bytes = word_bits // 8
        problem = multi_einsum_region_problem(region)
        architecture = OrojenesisRunner.multi_architecture(word_bits)
        environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
        paths: dict[str, Path] = {}

        def write_yaml(name: str, value: dict) -> Path:
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(value, sort_keys=False))
            paths[name] = path
            return path

        write_yaml("region.yaml", problem)
        write_yaml("architecture.yaml", architecture)
        write_yaml("environment.yaml", environment)
        raw_paths: dict[str, list[Path]] = {}
        row_tiles_by_node: dict[str, list[int]] = {}
        sweeps: list[dict] = []
        for node_index, descriptor in enumerate(problem["nodes"]):
            node_id = str(descriptor["id"])
            row_tiles = multi_einsum_row_tiles(int(descriptor["m"]))
            row_tiles_by_node[node_id] = row_tiles
            role = multi_einsum_region_mapper_role(problem, node_id)
            node_problem = multi_einsum_layer_problem(descriptor)
            write_yaml(f"problem-node-{node_index}.yaml", node_problem)
            node_raw_paths: list[Path] = []
            for row_tile in row_tiles:
                prefix = f"node-{node_index}-m-{row_tile}"
                write_yaml(f"{prefix}-architecture.yaml", architecture)
                write_yaml(
                    f"{prefix}-mapper.yaml",
                    OrojenesisRunner.multi_mapper_config(row_tile, role=role),
                )
                write_yaml(f"{prefix}-problem.yaml", node_problem)
                raw_path = output_dir / prefix / "timeloop-mapper.oaves.csv"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    self._mapping_row(
                        descriptor,
                        word_bytes=word_bytes,
                        row_tile=row_tile,
                        layer_index=node_index,
                    )
                )
                paths[f"{prefix}-raw"] = raw_path
                node_raw_paths.append(raw_path)
            raw_paths[node_id] = node_raw_paths
            sweeps.append({"node_id": node_id, "row_tiles": row_tiles, "role": role})
        curve = compose_multi_einsum_region_curve(
            problem,
            raw_paths,
            row_tiles_by_node=row_tiles_by_node,
            word_bytes=word_bytes,
        )
        curve_path = output_dir / "multi-einsum-region-curve.csv"
        with curve_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for point in curve:
                writer.writerow(
                    [
                        point["buffer_bytes"],
                        point["operational_intensity"],
                        point["dram_accesses_words"],
                        json.dumps(point["mappings"], separators=(",", ":")),
                    ]
                )
        paths["curve"] = curve_path
        return {
            "solver": MULTI_EINSUM_SOLVER,
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "composition": problem["composition"],
            "word_bits": word_bits,
            "environment": environment,
            "problem": problem,
            "sweeps": sweeps,
            "curve": curve,
            "evidence_files": {
                name: {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in paths.items()
            },
        }


def _write_reference(path: Path) -> None:
    path.write_text(
        "import torch\n"
        "def get_inputs(workload, device):\n"
        "    g = torch.Generator(device=device).manual_seed(workload['seed'])\n"
        "    n = workload['n']\n"
        "    return (torch.randn((n, n), generator=g, device=device), "
        "torch.randn((n, n), generator=g, device=device))\n"
        "def run(a, b): return torch.mm(a, b)\n"
    )


def _graph(n: int = 3) -> dict:
    dtype = "torch.float32"
    return annotate_semantics(
        {
            "layers": {
                "start_a": {
                    "type": "start",
                    "tensor_names": {"inputs": [], "outputs": ["a"]},
                    "tensor_shapes": {"inputs": [], "outputs": [[n, n]]},
                    "tensor_dtypes": {"inputs": [], "outputs": [dtype]},
                },
                "start_b": {
                    "type": "start",
                    "tensor_names": {"inputs": [], "outputs": ["b"]},
                    "tensor_shapes": {"inputs": [], "outputs": [[n, n]]},
                    "tensor_dtypes": {"inputs": [], "outputs": [dtype]},
                },
                "matmul": {
                    "type": "matmul",
                    "einsum_equation": "MK,KN->MN",
                    "elementwise_op": "mul",
                    "reduction_op": "add",
                    "is_real_einsum": True,
                    "is_einsum_supportable": True,
                    "tensor_names": {"inputs": ["a", "b"], "outputs": ["output"]},
                    "tensor_types": {
                        "inputs": ["input", "input"],
                        "outputs": ["output"],
                    },
                    "tensor_shapes": {
                        "inputs": [[n, n], [n, n]],
                        "outputs": [[n, n]],
                    },
                    "tensor_dtypes": {
                        "inputs": [dtype, dtype],
                        "outputs": [dtype],
                    },
                    "connections": {"inputs": ["start_a", "start_b"], "outputs": []},
                },
            }
        },
        strict=True,
    )


def _multi_graph(n: int = 3) -> dict:
    graph = _graph(n)
    layers = graph["layers"]
    first = layers.pop("matmul")
    first["tensor_names"]["outputs"] = ["hidden"]
    first["connections"] = {
        "inputs": ["start_a", "start_b"],
        "outputs": ["second"],
    }
    layers["first"] = first
    layers["start_a"]["connections"] = {"inputs": [], "outputs": ["first"]}
    layers["start_b"]["connections"] = {"inputs": [], "outputs": ["first"]}
    layers["start_c"] = {
        "type": "start",
        "tensor_names": {"inputs": [], "outputs": ["c"]},
        "tensor_shapes": {"inputs": [], "outputs": [[n, n]]},
        "tensor_dtypes": {"inputs": [], "outputs": ["torch.float32"]},
        "connections": {"inputs": [], "outputs": ["second"]},
    }
    layers["second"] = {
        "type": "matmul",
        "einsum_equation": "MK,KN->MN",
        "elementwise_op": "mul",
        "reduction_op": "add",
        "is_real_einsum": True,
        "is_einsum_supportable": True,
        "tensor_names": {"inputs": ["hidden", "c"], "outputs": ["output"]},
        "tensor_types": {"inputs": ["intermediate", "input"], "outputs": ["output"]},
        "tensor_shapes": {
            "inputs": [[n, n], [n, n]],
            "outputs": [[n, n]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float32", "torch.float32"],
            "outputs": ["torch.float32"],
        },
        "connections": {"inputs": ["first", "start_c"], "outputs": []},
    }
    return annotate_semantics(graph, strict=True)


def _layout_multi_graph(n: int = 3) -> dict:
    graph = _multi_graph(n)
    layers = graph["layers"]
    first = layers["first"]
    first["einsum_equation"] = "MK,NK->MN"
    first["semantic_op"]["equation"] = "MK,NK->MN"
    first["connections"]["outputs"] = ["view"]
    layers["view"] = {
        "type": "view",
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "einsum_equation": "AB->AB",
        "semantic_op": {
            "kind": "aten",
            "target": "view",
            "overload": "default",
            "arguments": [{"tensor": 0}],
            "kwargs": {"shape": [n, n]},
            "effects": {
                "mutates": [],
                "aliases": [{"output": 0, "input": 0, "conditional": False}],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {"inputs": ["hidden"], "outputs": ["viewed"]},
        "tensor_shapes": {"inputs": [[n, n]], "outputs": [[n, n]]},
        "tensor_dtypes": {
            "inputs": ["torch.float32"],
            "outputs": ["torch.float32"],
        },
        "connections": {"inputs": ["first"], "outputs": ["second"]},
    }
    layers["second"]["tensor_names"]["inputs"][0] = "viewed"
    layers["second"]["connections"]["inputs"][0] = "view"
    return graph


def _create_attestation(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    reference = tmp_path / "reference.py"
    graph_path = tmp_path / "einsum_graph.yaml"
    verification = tmp_path / "verification.yaml"
    _write_reference(reference)
    graph_path.write_text(yaml.safe_dump(_graph(), sort_keys=False))
    artifact = create_verification_artifact(
        reference_path=reference,
        reference_entry_point="run",
        input_factory_name="get_inputs",
        graph_path=graph_path,
        workload_name="tiny",
        workload_parameters={"n": 3},
        output_path=verification,
        atol=1e-5,
        rtol=1e-5,
    )
    return reference, graph_path, verification, artifact


def test_in_toto_artifact_is_replayable_and_hash_bound(tmp_path: Path):
    reference, graph_path, _, artifact = _create_attestation(tmp_path)
    assert artifact["_type"] == "https://in-toto.io/Statement/v1"
    assert len(artifact["predicate"]["cases"]) == 9
    replay_verification_artifact(
        artifact,
        reference_path=reference,
        graph_path=graph_path,
        workload_name="tiny",
        workload_parameters={"n": 3},
        atol=1e-5,
        rtol=1e-5,
    )


def test_replay_rejects_reference_or_graph_tampering(tmp_path: Path):
    reference, graph_path, _, artifact = _create_attestation(tmp_path)
    reference.write_text(reference.read_text().replace("torch.mm", "torch.add"))
    with pytest.raises(VerificationError, match="reference SHA-256"):
        replay_verification_artifact(
            artifact,
            reference_path=reference,
            graph_path=graph_path,
            workload_name="tiny",
            workload_parameters={"n": 3},
            atol=1e-5,
            rtol=1e-5,
        )


def test_attestation_uses_official_nonfinite_and_ratio_policy() -> None:
    import torch

    negative_inf = torch.tensor([float("-inf"), 1.0])
    with pytest.raises(VerificationError, match="non-finite"):
        _assert_close(negative_inf, negative_inf, 0.0, 0.0)
    _assert_close(
        negative_inf,
        negative_inf,
        0.0,
        0.0,
        allow_negative_inf=True,
    )
    actual = torch.ones(100)
    expected = actual.clone()
    actual[-1] = 10.0
    _assert_close(
        actual,
        expected,
        0.0,
        0.0,
        required_matched_ratio=0.99,
    )
    with pytest.raises(VerificationError, match="exceeds cap"):
        _assert_close(
            actual,
            expected,
            0.0,
            0.0,
            required_matched_ratio=0.99,
            max_error_cap=5.0,
        )


def test_schema_v3_benchmark_requires_and_replays_verification(tmp_path: Path):
    reference, graph_path, verification, _ = _create_attestation(tmp_path)
    graph_digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    analysis_path = tmp_path / "derived" / "analysis.yaml"
    analysis = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path / "derived",
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(),
        require_orojenesis=True,
    )
    assert analysis is not None
    assert analysis["total"]["io_lower_bound_bytes"] > analysis["total"]["fused_bytes"]
    benchmark = {
        "schema_version": 3,
        "name": "trusted",
        "precision": "fp32",
        "reference": {
            "source": reference.name,
            "entry_point": "run",
            "input_factory": "get_inputs",
        },
        "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "cache_policy": "cold",
        "workloads": [
            {
                "name": "tiny",
                "status": "compatible",
                "parameters": {"n": 3},
                "analysis": {
                    "path": str(analysis_path.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                    "source_graph": graph_path.name,
                    "source_graph_sha256": graph_digest,
                },
                "verification": {
                    "path": verification.name,
                    "sha256": hashlib.sha256(verification.read_bytes()).hexdigest(),
                },
            }
        ],
    }
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(yaml.safe_dump(benchmark, sort_keys=False))
    loaded = BenchmarkSpec.load(benchmark_path)
    assert loaded.workloads[0].verification is not None
    architecture_identity = ArchitectureProfile.load("RX_9060_XT").to_dict()
    architecture_identity.pop("source", None)
    assert loaded.workloads[0].analysis is not None
    assert loaded.workloads[0].analysis.architecture_hash == canonical_hash(
        architecture_identity
    )
    curve_path = (
        analysis_path.parent / "orojenesis" / "matmul" / ("timeloop-mapper.oaves.csv")
    )
    original_curve = curve_path.read_text()
    curve_path.write_text("128,4,1,map,None,10\n")
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        BenchmarkSpec.load(benchmark_path)
    curve_path.write_text(original_curve)
    benchmark["workloads"][0].pop("verification")
    benchmark_path.write_text(yaml.safe_dump(benchmark, sort_keys=False))
    with pytest.raises(ValueError, match="require.*replayable verification"):
        BenchmarkSpec.load(benchmark_path)


def test_strict_conversion_and_analysis_fail_closed(tmp_path: Path):
    incomplete = {
        "layers": {
            "unknown": {
                "type": "unknown",
                "einsum_equation": "",
                "is_einsum_supportable": False,
                "tensor_shapes": {"inputs": [[2]], "outputs": [[2]]},
            }
        }
    }
    with pytest.raises(ConversionError, match="unsupported operation"):
        PyTorchToEinsum._validate_exact_graph(incomplete)
    path = tmp_path / "einsum_graph.yaml"
    path.write_text(yaml.safe_dump(incomplete))
    with pytest.raises(
        ValueError, match="strict analysis requires executable semantics"
    ):
        EinsumGraphAnalyzer().analyze_graph(path, tmp_path / "out", strict=True)


def test_orojenesis_curve_controls_formal_io_and_time_bound(tmp_path: Path) -> None:
    graph_path = tmp_path / "einsum_graph.yaml"
    graph_path.write_text(yaml.safe_dump(_graph(32), sort_keys=False))
    low = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path / "low",
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(5_000),
        require_orojenesis=True,
    )
    high = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        tmp_path / "high",
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(20_000),
        require_orojenesis=True,
    )
    assert low is not None and high is not None
    assert high["total"]["io_lower_bound_bytes"] > low["total"]["io_lower_bound_bytes"]
    assert high["total"]["lower_bound_seconds"] > low["total"]["lower_bound_seconds"]
    assert high["metadata"]["orojenesis"]["formal_coverage"] == {
        "applicable_layers": 1,
        "total_layers": 1,
    }


def test_multi_einsum_artifact_recomposes_raw_mapping_evidence(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "multi_graph.yaml"
    graph_path.write_text(yaml.safe_dump(_multi_graph(), sort_keys=False))
    output_dir = tmp_path / "derived"
    analysis = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        output_dir,
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(),
        require_orojenesis=True,
    )
    assert analysis is not None
    solver = analysis["metadata"]["orojenesis"]
    assert solver["layers"] == {}
    assert list(solver["chains"]) == ["chain_0"]
    assert solver["formal_coverage"] == {
        "applicable_layers": 2,
        "total_layers": 2,
    }
    fusion_decision = analysis["metadata"]["fusion"]["decisions"][0]
    assert fusion_decision["reason"] == "verified_multi_einsum_chain"
    assert analysis["total"]["io_lower_bound_bytes"] > analysis["total"]["fused_bytes"]

    analysis_path = output_dir / "analysis.yaml"
    artifact_data = {
        "path": str(analysis_path.relative_to(tmp_path)),
        "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "source_graph": graph_path.name,
        "source_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    }
    loaded = AnalysisArtifact.load(artifact_data, tmp_path)
    assert loaded.lower_bound_seconds == analysis["total"]["lower_bound_seconds"]

    tampered = yaml.safe_load(analysis_path.read_text())
    tampered_curve = tampered["metadata"]["orojenesis"]["chains"]["chain_0"]["curve"]
    tampered_curve[0]["dram_accesses_words"] += 1
    analysis_path.write_text(yaml.safe_dump(tampered, sort_keys=False))
    artifact_data["sha256"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="multi-einsum curve drifted"):
        AnalysisArtifact.load(artifact_data, tmp_path)

    resigned = yaml.safe_load(yaml.safe_dump(analysis, sort_keys=False))
    chain_result = resigned["metadata"]["orojenesis"]["chains"]["chain_0"]
    raw_evidence = chain_result["evidence_files"]["layer-0-m-1-raw"]
    raw_path = analysis_path.parent / raw_evidence["path"]
    row = next(csv.reader([raw_path.read_text().strip()]))
    row[2] = str(float(row[2]) + 1)
    row[21] = str(float(row[21]) + 1)
    raw_path.write_text(",".join(row) + "\n")
    raw_evidence["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    analysis_path.write_text(yaml.safe_dump(resigned, sort_keys=False))
    artifact_data["sha256"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="multi-einsum curve drifted"):
        AnalysisArtifact.load(artifact_data, tmp_path)


def test_extended_multi_einsum_region_is_independently_replayed(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "layout_multi_graph.yaml"
    graph_path.write_text(yaml.safe_dump(_layout_multi_graph(), sort_keys=False))
    output_dir = tmp_path / "region-derived"
    analysis = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
        output_dir,
        precision="fp32",
        copy_graph=False,
        strict=True,
        architecture="RX_9060_XT",
        orojenesis_runner=_DeterministicOrojenesisRunner(),
        require_orojenesis=True,
    )
    assert analysis is not None
    solver = analysis["metadata"]["orojenesis"]
    assert solver["layers"] == {}
    assert solver["chains"] == {}
    assert list(solver["regions"]) == ["region_0"]
    assert solver["regions"]["region_0"]["problem"]["kind"] == (
        "linear_matmul_with_axis_maps"
    )
    assert solver["formal_coverage"] == {
        "applicable_layers": 2,
        "total_layers": 2,
    }

    analysis_path = output_dir / "analysis.yaml"
    artifact_data = {
        "path": str(analysis_path.relative_to(tmp_path)),
        "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "source_graph": graph_path.name,
        "source_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    }
    loaded = AnalysisArtifact.load(artifact_data, tmp_path)
    assert loaded.lower_bound_seconds == analysis["total"]["lower_bound_seconds"]

    tampered = yaml.safe_load(analysis_path.read_text())
    region_result = tampered["metadata"]["orojenesis"]["regions"]["region_0"]
    region_result["curve"][0]["dram_accesses_words"] += 1
    analysis_path.write_text(yaml.safe_dump(tampered, sort_keys=False))
    artifact_data["sha256"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="multi-einsum region curve drifted"):
        AnalysisArtifact.load(artifact_data, tmp_path)


def test_solver_applicability_traces_only_unconditional_aliases() -> None:
    graph = _graph()
    matmul = graph["layers"]["matmul"]
    transpose = {
        "type": "transpose",
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "tensor_names": {"inputs": ["b"], "outputs": ["bt"]},
        "tensor_shapes": {"inputs": [[3, 3]], "outputs": [[3, 3]]},
        "tensor_dtypes": {
            "inputs": ["torch.float32"],
            "outputs": ["torch.float32"],
        },
        "module_args": {
            "call_arguments": [{"tensor": 0}, {"value": 0}, {"value": 1}],
            "call_kwargs": {},
        },
        "connections": {"inputs": ["start_b"], "outputs": ["matmul"]},
    }
    graph["layers"]["transpose"] = transpose
    matmul["tensor_names"]["inputs"][1] = "bt"
    graph = annotate_semantics(graph, strict=True)
    assert contraction_operands_are_graph_external(matmul, graph["layers"])

    graph["layers"]["transpose"]["semantic_op"]["effects"]["aliases"][0][
        "conditional"
    ] = True
    assert not contraction_operands_are_graph_external(matmul, graph["layers"])
