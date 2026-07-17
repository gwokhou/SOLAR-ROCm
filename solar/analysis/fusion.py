# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conservative legality and capacity analysis for SOLAR fusion regions."""

# Fusion is deliberately a single ordered proof pass over the DAG so every
# legal/illegal edge and capacity consequence is emitted together.
# pylint: disable=too-few-public-methods,too-many-return-statements,missing-function-docstring,too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx

from solar.common.constants import dtype_bytes
from solar.einsum.semantics import validate_semantic_graph
from solar.rocm.architecture import MemoryLevel


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _tensor_bytes(shape: Sequence[int], dtype: str) -> int:
    return int(_product(shape) * dtype_bytes(dtype))


class FusionPlanner:
    """Build maximal regions while treating unproven fusion as illegal."""

    def __init__(self, graph: Mapping[str, Any]):
        validate_semantic_graph(graph)
        self.graph = graph
        self.layers = {
            str(key): value
            for key, value in (graph.get("layers") or {}).items()
            if str(value.get("type", "")).lower() != "start"
        }

    @staticmethod
    def _barrier(layer: Mapping[str, Any]) -> str | None:
        semantic = layer["semantic_op"]
        effects = semantic.get("effects") or {}
        target = str(semantic.get("target", ""))
        if effects.get("mutates"):
            return "mutation"
        if effects.get("aliases"):
            return "observable_alias"
        if effects.get("atomic"):
            return "atomic"
        if effects.get("opaque_library_call"):
            return "opaque_library_call"
        if target in {
            "sum",
            "mean",
            "prod",
            "amax",
            "amin",
            "argmax",
            "argmin",
            "logsumexp",
        }:
            return "synchronizing_reduction"
        outputs = (layer.get("tensor_names") or {}).get("outputs") or []
        if len(outputs) != 1:
            return "multi_output_not_proven_safe"
        return None

    def plan(self, hierarchy: Sequence[MemoryLevel]) -> dict[str, Any]:
        dag = nx.DiGraph()
        dag.add_nodes_from(self.layers)
        for layer_id, layer in self.layers.items():
            for consumer in (layer.get("connections") or {}).get("outputs") or []:
                if consumer in self.layers:
                    dag.add_edge(layer_id, consumer)
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("fusion requires an acyclic semantic graph")

        parent = {node: node for node in dag.nodes}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        decisions: list[dict[str, Any]] = []
        for producer, consumer in dag.edges:
            producer_reason = self._barrier(self.layers[producer])
            consumer_reason = self._barrier(self.layers[consumer])
            reason = producer_reason or consumer_reason
            legal = reason is None
            if legal:
                union(producer, consumer)
            decisions.append(
                {
                    "producer": producer,
                    "consumer": consumer,
                    "legal": legal,
                    "reason": reason or "pure_dependency",
                }
            )

        groups: dict[str, list[str]] = defaultdict(list)
        order = list(nx.topological_sort(dag))
        for node in order:
            groups[find(node)].append(node)

        producers: dict[str, str] = {}
        consumers: dict[str, set[str]] = defaultdict(set)
        tensor_metadata: dict[str, tuple[list[int], str]] = {}
        for node, layer in self.layers.items():
            names = layer.get("tensor_names") or {}
            shapes = layer.get("tensor_shapes") or {}
            dtypes = layer.get("tensor_dtypes") or {}
            for name, shape, dtype in zip(
                names.get("outputs") or [],
                shapes.get("outputs") or [],
                dtypes.get("outputs") or [],
            ):
                producers[str(name)] = node
                tensor_metadata[str(name)] = (list(shape), str(dtype))
            for name, shape, dtype in zip(
                names.get("inputs") or [],
                shapes.get("inputs") or [],
                dtypes.get("inputs") or [],
            ):
                consumers[str(name)].add(node)
                tensor_metadata.setdefault(str(name), (list(shape), str(dtype)))

        regions: list[dict[str, Any]] = []
        for index, nodes in enumerate(groups.values()):
            node_set = set(nodes)
            external_inputs: set[str] = set()
            external_outputs: set[str] = set()
            internal: set[str] = set()
            unfused_bytes = 0
            for node in nodes:
                layer = self.layers[node]
                shapes = layer.get("tensor_shapes") or {}
                dtypes = layer.get("tensor_dtypes") or {}
                for shape, dtype in zip(
                    shapes.get("inputs") or [], dtypes.get("inputs") or []
                ):
                    unfused_bytes += _tensor_bytes(shape, str(dtype))
                for shape, dtype in zip(
                    shapes.get("outputs") or [], dtypes.get("outputs") or []
                ):
                    unfused_bytes += _tensor_bytes(shape, str(dtype))
                for name in (layer.get("tensor_names") or {}).get("inputs") or []:
                    if producers.get(str(name)) not in node_set:
                        external_inputs.add(str(name))
                for name in (layer.get("tensor_names") or {}).get("outputs") or []:
                    name = str(name)
                    uses = consumers.get(name) or set()
                    if uses and uses.issubset(node_set):
                        internal.add(name)
                    else:
                        external_outputs.add(name)

            fused_bytes = sum(
                _tensor_bytes(*tensor_metadata[name])
                for name in external_inputs | external_outputs
                if name in tensor_metadata
            )
            # A tensor is live from its producer until its last consumer in the
            # region.  This is an auditable capacity lower bound, not a claimed
            # physical allocation schedule.
            position = {node: offset for offset, node in enumerate(nodes)}
            events: dict[int, int] = defaultdict(int)
            for name in internal:
                size = _tensor_bytes(*tensor_metadata[name])
                start = position[producers[name]]
                end = max(
                    position[consumer]
                    for consumer in consumers[name]
                    if consumer in node_set
                )
                events[start] += size
                events[end + 1] -= size
            live = 0
            peak_live = 0
            for offset in range(len(nodes) + 1):
                live += events[offset]
                peak_live = max(peak_live, live)

            capacities: dict[str, Any] = {}
            spill_lower_bound = 0
            for level in hierarchy:
                capacity = level.capacity_bytes
                spill = None if capacity is None else max(0, peak_live - capacity)
                if spill is not None:
                    spill_lower_bound = max(spill_lower_bound, 2 * spill)
                capacities[level.name] = {
                    "scope": level.scope,
                    "capacity_bytes": capacity,
                    "peak_live_bytes": peak_live,
                    "spill_bytes_lower_bound": None if spill is None else 2 * spill,
                    "source": level.source,
                }
            regions.append(
                {
                    "id": f"region_{index}",
                    "layers": nodes,
                    "external_inputs": sorted(external_inputs),
                    "external_outputs": sorted(external_outputs),
                    "unfused_bytes": unfused_bytes,
                    "fused_bytes": fused_bytes,
                    "prefetched_bytes": fused_bytes + spill_lower_bound,
                    "peak_live_bytes": peak_live,
                    "capacity": capacities,
                }
            )
        return {"decisions": decisions, "regions": regions}


__all__ = ["FusionPlanner"]
