# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyze an einsum graph into hardware-independent metrics.

This module implements the **second stage** of the Solar pipeline:

  `einsum_graph.yaml`  ->  `analysis.yaml`

The output `analysis.yaml` is intended to be hardware-independent and includes:
- per-layer: macs, flops (= 2 * macs), tensor dtypes, and exact byte traffic
- totals across the graph

Memory access models (elements are diagnostic; byte totals use each tensor dtype):
- ``unfused``: every per-operation input and output access;
- ``fused``: compulsory, deduplicated graph-external I/O;
- ``prefetched`` / ``io_lower_bound``: compulsory I/O plus the safely composable
  excess traffic selected from a capacity-constrained Orojenesis curve.

Formal schema-v3 analysis also emits conservative fusion regions, hierarchy
capacity pressure, the pinned solver inputs/raw curve and their SHA-256 hashes,
and separate compute/memory overlap components.  Unsupported multi-einsum
composition is never approximated into a scored bound.

Note: input_elements includes all inputs to an operation (including weights/biases).
Weights are treated as inputs since they are just another operand to the computation.

Note: "start" nodes are filtered out before analysis as they represent model inputs,
not actual computation. Their outputs are treated as external inputs to the graph.

See SOL_GUIDE.md for detailed explanation of the three SOL models.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from solar.einsum import EinsumAnalyzer
from solar.einsum.semantics import (
    EINSUM_GRAPH_SCHEMA_VERSION,
    SemanticGraphError,
    validate_semantic_graph,
)
from solar.analysis.fusion import FusionPlanner
from solar.analysis.orojenesis import OrojenesisRunner, select_capacity_point
from solar.rocm.architecture import ArchitectureProfile
from solar.common.constants import (
    BYTES_PER_ELEMENT,
    DEFAULT_PRECISION,
    dtype_bytes,
    normalize_dtype,
)
from solar.common.types import TensorShapes
from solar.common.utils import ensure_directory, NoAliasDumper

PathLike = Union[str, Path]


def _product(shape: List[int]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return int(out)


def contraction_operands_are_graph_external(
    layer: dict[str, Any], layers: dict[str, Any]
) -> bool:
    """Return whether every contraction operand traces to a graph input.

    Unconditional aliasing views are transparent. Conditional aliases such as
    reshape/contiguous are not, because a formal proof must cover layouts that
    require materialization.
    """
    producers = {
        str(name): (str(layer_id), producer, output_index)
        for layer_id, producer in layers.items()
        for output_index, name in enumerate(
            (producer.get("tensor_names") or {}).get("outputs") or []
        )
    }

    def traces_external(name: str, visited: set[str]) -> bool:
        if name in visited:
            return False
        produced = producers.get(name)
        if produced is None:
            return True
        _, producer, output_index = produced
        if str(producer.get("type", "")).lower() == "start":
            return True
        semantic = producer.get("semantic_op") or {}
        effects = semantic.get("effects") or {}
        aliases = [
            alias
            for alias in effects.get("aliases") or []
            if int(alias.get("output", -1)) == output_index
            and not bool(alias.get("conditional", False))
        ]
        if len(aliases) != 1:
            return False
        input_index = int(aliases[0].get("input", -1))
        input_names = (producer.get("tensor_names") or {}).get("inputs") or []
        if input_index not in range(len(input_names)):
            return False
        return traces_external(str(input_names[input_index]), visited | {name})

    return all(
        traces_external(str(name), set())
        for name in (layer.get("tensor_names") or {}).get("inputs") or []
    )


class EinsumGraphAnalyzer:
    """Analyze `einsum_graph.yaml` and write `analysis.yaml`."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.einsum_analyzer = EinsumAnalyzer(debug=debug)

    def analyze_graph(
        self,
        einsum_graph_path: PathLike,
        output_dir: PathLike,
        *,
        precision: str = DEFAULT_PRECISION,
        copy_graph: bool = True,
        strict: bool = False,
        architecture: str | Path | ArchitectureProfile | None = None,
        orojenesis_runner: OrojenesisRunner | None = None,
        require_orojenesis: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Analyze an einsum graph and write `analysis.yaml`.

        Args:
            einsum_graph_path: Path to `einsum_graph.yaml`.
            output_dir: Directory to write `analysis.yaml` into.
            precision: Tensor precision for byte calculations (e.g., fp32, bf16).
            copy_graph: If True, copy the einsum graph into output dir using the
                canonical name `einsum_graph.yaml`.
            strict: Reject unsupported layers and every implicit dtype fallback.

        Returns:
            Analysis dict, or None on failure.
        """
        src = Path(einsum_graph_path)
        # Stage 2.5 fallback: prefer einsum_graph_reordered.yaml if present in the same dir
        reordered = src.parent / "einsum_graph_reordered.yaml"
        if src.name == "einsum_graph.yaml" and reordered.exists():
            if self.debug:
                print(f"Debug: using reordered graph {reordered}")
            src = reordered
        out_dir = ensure_directory(output_dir)

        if not src.exists():
            if self.debug:
                print(f"Debug: einsum graph not found: {src}")
            return None

        try:
            with open(src) as f:
                graph = yaml.safe_load(f) or {}
        except Exception as exc:
            if self.debug:
                print(f"Debug: failed reading einsum graph: {exc}")
            return None

        semantic_graph = (
            int(graph.get("schema_version", 0)) == EINSUM_GRAPH_SCHEMA_VERSION
        )
        if strict and not semantic_graph:
            raise ValueError(
                "strict analysis requires executable semantics: "
                f"einsum graph must use latest schema_version={EINSUM_GRAPH_SCHEMA_VERSION}"
            )
        semantic_complete = False
        if semantic_graph:
            try:
                validate_semantic_graph(graph)
                semantic_complete = True
            except SemanticGraphError as exc:
                if strict:
                    raise ValueError(
                        f"strict analysis requires executable semantics: {exc}"
                    ) from exc

        if copy_graph:
            try:
                dst = out_dir / "einsum_graph.yaml"
                if src.resolve() != dst.resolve():
                    dst.write_text(src.read_text())
            except Exception:
                if self.debug:
                    print("Debug: failed to copy einsum_graph.yaml")

        all_layers: Dict[str, Any] = graph.get("layers") or {}
        if strict:
            failures: List[str] = []
            for layer_id, layer in all_layers.items():
                layer_type = str(layer.get("type", "")).lower()
                if layer_type != "start":
                    if layer.get("is_einsum_supportable") is not True:
                        failures.append(f"{layer_id}: unsupported operation")
                    semantic = layer.get("semantic_op") or {}
                    if semantic.get("kind") == "einsum" and not layer.get(
                        "einsum_equation"
                    ):
                        failures.append(f"{layer_id}: empty einsum equation")
                shapes = layer.get("tensor_shapes") or {}
                dtypes = layer.get("tensor_dtypes") or {}
                for side in ("inputs", "outputs"):
                    if len(shapes.get(side) or []) != len(dtypes.get(side) or []):
                        failures.append(
                            f"{layer_id}: missing explicit {side} dtype metadata"
                        )
            if failures:
                raise ValueError(
                    "strict analysis refused an untrusted graph:\n- "
                    + "\n- ".join(failures)
                )
        requested_precision = normalize_dtype(precision)
        element_size = BYTES_PER_ELEMENT[requested_precision]

        fallback_precision = requested_precision
        used_dtype_fallback = False

        # Filter out "start" nodes - they represent model inputs, not computation
        # Keep track of start node IDs for reference
        _BOOL_DTYPES = {"torch.bool", "bool"}
        start_node_ids: Set[str] = set()
        bool_start_node_ids: Set[str] = set()
        layers_in: Dict[str, Any] = {}

        for layer_id, layer in all_layers.items():
            op_type = str(layer.get("type", "")).lower()
            if op_type == "start":
                start_node_ids.add(layer_id)
                # Detect bool-typed start nodes from tensor_dtypes
                out_dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
                if out_dtypes and all(str(d) in _BOOL_DTYPES for d in out_dtypes):
                    bool_start_node_ids.add(layer_id)
            else:
                layers_in[layer_id] = layer

        if self.debug:
            print(f"Debug: Filtered out {len(start_node_ids)} start nodes")
            if bool_start_node_ids:
                print(
                    f"Debug: Found {len(bool_start_node_ids)} bool-typed start nodes: {bool_start_node_ids}"
                )
            print(f"Debug: Analyzing {len(layers_in)} computation nodes")

        # Build tensor producer/consumer maps using tensor_names from the
        # einsum graph.  A tensor is intermediate if it is produced by one op
        # AND consumed by another op.
        all_layer_ids: Set[str] = set(layers_in.keys())

        # Zero-copy view layers are transparent for memory accounting.  The
        # fused model should see through them when deciding whether a tensor is
        # graph-internal or external.
        _TRANSPARENT_OPS = {
            "expand",
            "expand_as",
            "view",
            "reshape",
            "contiguous",
            "transpose",
            "permute",
            "t",
            "unsqueeze",
            "squeeze",
            "flatten",
            "unfold",
            "unflatten",
            "chunk",
            "split",
            "tensor_split",
        }
        transparent_layer_ids: Set[str] = set()
        for layer_id, layer in layers_in.items():
            if str(layer.get("type", "")).lower() in _TRANSPARENT_OPS:
                transparent_layer_ids.add(layer_id)

        tensor_producers: Dict[str, str] = {}  # tensor_name -> producer_layer_id
        tensor_consumers: Dict[str, Set[str]] = (
            {}
        )  # tensor_name -> set of consumer_layer_ids

        # Pass 1: gather all produced tensor names (order-independent).
        for layer_id, layer in layers_in.items():
            t_names = layer.get("tensor_names") or {}
            for oname in t_names.get("outputs") or []:
                tensor_producers[oname] = layer_id

        # Pass 2: gather consumers for tensors produced somewhere in graph.
        for layer_id, layer in layers_in.items():
            t_names = layer.get("tensor_names") or {}
            for iname in t_names.get("inputs") or []:
                if iname in tensor_producers:
                    tensor_consumers.setdefault(iname, set()).add(layer_id)

        # Identify intermediate tensors: produced by one op AND consumed by another
        intermediate_tensors: Set[str] = set()
        for tensor_name in tensor_producers:
            if (
                tensor_name in tensor_consumers
                and len(tensor_consumers[tensor_name]) > 0
            ):
                intermediate_tensors.add(tensor_name)

        def _trace_source_through_views(layer_id: str) -> str:
            """Trace backward through transparent view layers to the real source."""
            visited: Set[str] = set()
            current = layer_id
            while current in transparent_layer_ids and current not in visited:
                visited.add(current)
                conns = (layers_in[current].get("connections") or {}).get(
                    "inputs"
                ) or []
                if not conns:
                    break
                current = conns[0]
            return current

        def _has_real_consumer(layer_id: str) -> bool:
            """Return true if output reaches a non-transparent graph layer."""
            visited: Set[str] = set()
            queue = [layer_id]
            while queue:
                lid = queue.pop(0)
                if lid in visited:
                    continue
                visited.add(lid)
                conns = (layers_in.get(lid, {}).get("connections") or {}).get(
                    "outputs"
                ) or []
                for out_id in conns:
                    if out_id in transparent_layer_ids:
                        queue.append(out_id)
                    elif out_id in all_layer_ids:
                        return True
            return False

        if self.debug:
            print(f"Debug: Found {len(intermediate_tensors)} intermediate tensors")
            for t in sorted(intermediate_tensors)[:10]:
                print(f"  - {t}")
            if transparent_layer_ids:
                print(f"Debug: {len(transparent_layer_ids)} transparent view layers")

        # TEMPORARY FIX: Propagate bool-ness from start nodes through graph.
        # A computation layer is "bool" if ALL its inputs come from bool
        # sources (bool start nodes or other bool layers).
        _bool_layers: Set[str] = set()
        if bool_start_node_ids:
            # Process layers in topological-ish order (inputs before outputs)
            # by iterating until no more layers are added.
            changed = True
            while changed:
                changed = False
                for layer_id, layer in layers_in.items():
                    if layer_id in _bool_layers:
                        continue
                    conns = layer.get("connections") or {}
                    inp_ids = list(conns.get("inputs") or [])
                    if not inp_ids:
                        continue
                    # All inputs must be bool (start or layer)
                    if all(
                        inp in bool_start_node_ids or inp in _bool_layers
                        for inp in inp_ids
                    ):
                        _bool_layers.add(layer_id)
                        changed = True

            if self.debug and _bool_layers:
                print(
                    f"Debug: Skipping memory for {len(_bool_layers)} bool-derived layers: {sorted(_bool_layers)}"
                )

        # Detect orphaned subgraphs: chains rooted at tensors created
        # outside RecorderTensor tracking (e.g. torch.zeros() with no
        # subclass args).  The source-tracing code (Step 4 below)
        # classifies inputs from non-existent nodes as external DRAM I/O.
        # For dead-end layers whose input traces to a genuinely orphaned
        # source, this phantom traffic should be zeroed.
        #
        # A layer's input is "orphaned" when the source traces to a
        # non-existent ID AND the layer produces no live output.
        # For scatter/setitem ops, if the TARGET (first input) traces to
        # an orphaned source AND the output is dead-end, the write is
        # phantom too.
        #
        # Pre-compute which layers are dead ends (no live output path).
        _dead_end_layers: Set[str] = set()
        for layer_id in layers_in:
            if not _has_real_consumer(layer_id):
                _dead_end_layers.add(layer_id)

        # Track which layers are flagged as orphaned (for metadata).
        _orphaned_layers: Set[str] = set()

        layers_out: Dict[str, Any] = {}
        total_macs = 0  # contracted-operation MACs (matmul, conv)
        total_flops = 0  # 2 * total_macs
        total_other_ops = 0  # scalar/vector elementwise and reduction ops
        total_unfused_elems = 0  # Σ (all input + output elems) per op
        total_intermediate_elems = 0  # Σ intermediate activation elems
        total_unfused_bytes = 0.0
        total_intermediate_bytes = 0.0
        macs_by_precision: Dict[str, int] = defaultdict(int)

        # Deduplicated external (non-intermediate) tensor tracking for the
        # fused / fused_prefetched model.  When the same external tensor
        # (e.g. model input x) fans out to multiple ops, it is read from
        # DRAM once.  We track by tensor_name → max element count.
        unique_external_inputs: Dict[str, int] = {}
        unique_external_outputs: Dict[str, int] = {}
        unique_external_input_bytes: Dict[str, float] = {}
        unique_external_output_bytes: Dict[str, float] = {}

        for layer_id, layer in layers_in.items():
            op_type = str(layer.get("type", "unknown"))
            equation = str(layer.get("einsum_equation", "") or "")
            if "is_real_einsum" not in layer:
                raise ValueError(
                    f"Layer '{layer_id}' (type={op_type}) is missing 'is_real_einsum' field. "
                    f"All layers in the einsum graph must specify is_real_einsum: true/false."
                )
            is_real_einsum = bool(layer["is_real_einsum"])
            tensor_shapes: Dict[str, Any] = layer.get("tensor_shapes") or {}
            tensor_types: Dict[str, Any] = layer.get("tensor_types") or {}
            tensor_dtypes: Dict[str, Any] = layer.get("tensor_dtypes") or {}
            tensor_names: Dict[str, Any] = layer.get("tensor_names") or {}
            connections: Dict[str, Any] = layer.get("connections") or {}
            input_layer_ids = list(connections.get("inputs") or [])
            output_layer_ids = list(connections.get("outputs") or [])

            ts = TensorShapes(
                inputs=tensor_shapes.get("inputs", []),
                outputs=tensor_shapes.get("outputs", []),
            )

            ops_cost = 0
            try:
                if is_real_einsum and equation:
                    ops_cost = int(
                        self.einsum_analyzer.get_compute_cost(
                            op_type, ts, equation=equation
                        )
                    )
                else:
                    ops_cost = int(self.einsum_analyzer.get_compute_cost(op_type, ts))
            except Exception:
                ops_cost = 0

            # Zero-compute operations: no ALU work, only pointer/metadata
            # manipulation or pure memory copies.
            #
            # View/reshape ops: pointer manipulation, zero cost
            # Slice/select ops: pointer offset, zero cost
            # Scatter/index ops: in-place writes, zero compute
            # Embedding: table lookup, zero MACs
            # Memory ops (cat, repeat, stack, chunk, split): move data but
            #   have zero *compute* cost — bounded by memory bandwidth, not
            #   scalar/vector throughput. Their memory cost is already captured by
            #   input_elems/output_elems; assigning them other_ops would
            _ZERO_COMPUTE_OPS = {
                # Embedding
                "embedding",
                "embedding_bag",
                # View / reshape (pointer manipulation)
                "expand",
                "expand_as",
                "view",
                "reshape",
                "contiguous",
                "transpose",
                "permute",
                "t",
                "unsqueeze",
                "squeeze",
                "flatten",
                "unfold",
                "unflatten",
                # Slice / select (pointer offset)
                "__getitem__",
                "narrow",
                "slice",
                "select",
                # Scatter / in-place write
                "__setitem__",
                "scatter",
                "scatter_",
                "index_copy",
                "index_copy_",
                "index_put",
                "index_put_",
                # Memory-only ops (data movement, zero ALU compute)
                "cat",
                "concat",
                "stack",
                "chunk",
                "split",
                "tensor_split",
                "repeat",
                "repeat_interleave",
                "tile",
                "roll",
                "flip",
                "pad",
                "constant_pad_nd",
                "clone",
                "copy_",
                # Type conversion (zero compute)
                "to",
                "type",
                "type_as",
                "float",
                "half",
                "bfloat16",
                "int",
            }
            if op_type in _ZERO_COMPUTE_OPS:
                ops_cost = 0
                is_real_einsum = False

            if is_real_einsum:
                macs = ops_cost
                other_ops = 0
            else:
                macs = 0
                other_ops = ops_cost

            flops = int(2 * macs)

            input_shapes = tensor_shapes.get("inputs") or []
            output_shapes = tensor_shapes.get("outputs") or []
            input_type_list = tensor_types.get("inputs") or []
            output_type_list = tensor_types.get("outputs") or []
            input_dtype_list = list(tensor_dtypes.get("inputs") or [])
            output_dtype_list = list(tensor_dtypes.get("outputs") or [])

            # ── Step 1: Compute per-tensor sizes from shapes ──
            input_sizes: List[int] = []
            output_sizes: List[int] = []
            for shp in input_shapes:
                input_sizes.append(_product(shp) if isinstance(shp, list) else 0)
            for shp in output_shapes:
                output_sizes.append(_product(shp) if isinstance(shp, list) else 0)

            # memory_reads[i]  = DRAM read elements for input tensor i
            # memory_writes[i] = DRAM write elements for output tensor i
            # Initialised to raw tensor sizes; special-case ops override below.
            memory_reads: List[int] = list(input_sizes)
            memory_writes: List[int] = list(output_sizes)

            # ── Step 2: Override memory_reads/writes for special-case ops ──

            _ZERO_COPY_VIEW_OPS = {
                "expand",
                "expand_as",
                "view",
                "reshape",
                "contiguous",
                "transpose",
                "permute",
                "t",
                "unsqueeze",
                "squeeze",
                "flatten",
                "unfold",
                "unflatten",
                # chunk/split return views into the source tensor
                "chunk",
                "split",
                "tensor_split",
            }
            _SLICE_VIEW_OPS = {
                "__getitem__",
                "narrow",
                "slice",
                "select",
            }
            _SCATTER_OPS = {
                "__setitem__",
                "scatter",
                "scatter_",
                "index_copy",
                "index_copy_",
                "index_put",
                "index_put_",
            }

            # For embedding (table lookup), only the gathered rows are read
            # from the weight matrix, not the entire vocabulary table.
            # Input shapes are [indices_shape, weight_shape] where weight is
            # [vocab_size, embedding_dim].  The actual DRAM read is just the
            # rows selected by indices — which equals the output shape
            # [batch, seq, embedding_dim].  Use min(input, output) to handle
            # both small token counts (gathered rows << full table) and large
            # token counts (most/all rows accessed, full table is the bound).
            if op_type in ("embedding", "embedding_bag"):
                total_output = sum(output_sizes)
                gathered = min(sum(input_sizes), total_output)
                memory_reads = [0] * len(input_sizes)
                if input_sizes:
                    memory_reads[-1] = gathered
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # View/reshape ops produce zero-copy aliases — they never
            # materialize data to DRAM.  The downstream consumer accounts
            # for the actual read, so these ops contribute 0 memory.
            if op_type in _ZERO_COPY_VIEW_OPS:
                memory_reads = [0] * len(input_sizes)
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # Slicing/selection ops return a view into the source tensor.
            # The actual memory read is the output slice size, not the
            # full source.  Set read = output size so the downstream
            # consumer accounts for reading the slice.
            elif op_type in _SLICE_VIEW_OPS:
                out_total = sum(output_sizes)
                memory_reads = [out_total] if input_sizes else []
                memory_reads += [0] * max(0, len(input_sizes) - 1)
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # Scatter/index-write ops (__setitem__, scatter, index_copy)
            # write a slice into a large target tensor.  Memory cost is
            # the values being written, not the full target.  The smallest
            # input shape is typically the values/indices; use that as
            # the write cost and set output to the same (in-place update).
            elif op_type in _SCATTER_OPS:
                if len(input_sizes) >= 2:
                    slice_elems = max(sorted(input_sizes)[:-1])
                elif input_sizes:
                    slice_elems = min(input_sizes)
                elif output_sizes:
                    slice_elems = min(output_sizes)
                else:
                    slice_elems = 0
                memory_reads = [0] * len(input_sizes)
                memory_writes = [slice_elems] if output_sizes else []
                memory_writes += [0] * max(0, len(output_sizes) - 1)
                other_ops = 0

            # Orphaned dead-end layers: ALL inputs trace (through views)
            # to non-existent sources AND the layer has no live output.
            # For scatter/setitem: if the TARGET (first input) traces to a
            # non-existent source and the output is dead-end, the write is
            # phantom — zero all memory.
            # Standalone if (not elif) so it overrides any prior op-type branch.
            if layer_id in _dead_end_layers and input_layer_ids:
                _is_orphan = False
                _SCATTER_TARGET_OPS_INLINE = {
                    "__setitem__",
                    "scatter",
                    "scatter_",
                    "index_copy",
                    "index_copy_",
                    "index_put",
                    "index_put_",
                }

                def _source_is_orphan(cid: str) -> bool:
                    src = (
                        _trace_source_through_views(cid)
                        if cid in transparent_layer_ids
                        else cid
                    )
                    return src not in all_layer_ids and src not in start_node_ids

                if all(_source_is_orphan(c) for c in input_layer_ids):
                    _is_orphan = True

                if (
                    not _is_orphan
                    and op_type in _SCATTER_TARGET_OPS_INLINE
                    and _source_is_orphan(input_layer_ids[0])
                ):
                    _is_orphan = True

                if _is_orphan:
                    memory_reads = [0] * len(input_sizes)
                    memory_writes = [0] * len(output_sizes)
                    other_ops = 0
                    _orphaned_layers.add(layer_id)

            # ── Step 3: Derive totals from corrected per-tensor counts ──
            input_elems = int(sum(memory_reads))
            output_elems = int(sum(memory_writes))
            unfused_elems = input_elems + output_elems
            if any(
                count > 0 and i >= len(input_dtype_list)
                for i, count in enumerate(memory_reads)
            ) or any(
                count > 0 and i >= len(output_dtype_list)
                for i, count in enumerate(memory_writes)
            ):
                used_dtype_fallback = True
            input_bytes = [
                float(count)
                * dtype_bytes(
                    input_dtype_list[i] if i < len(input_dtype_list) else None,
                    fallback_precision,
                )
                for i, count in enumerate(memory_reads)
            ]
            output_bytes = [
                float(count)
                * dtype_bytes(
                    output_dtype_list[i] if i < len(output_dtype_list) else None,
                    fallback_precision,
                )
                for i, count in enumerate(memory_writes)
            ]
            unfused_bytes = float(sum(input_bytes) + sum(output_bytes))

            compute_precisions = []
            for dtype in input_dtype_list:
                normalized = normalize_dtype(dtype, fallback_precision)
                if normalized in {
                    "fp64",
                    "fp32",
                    "tf32",
                    "bf16",
                    "fp16",
                    "fp8",
                    "nvfp4",
                    "int8",
                    "int4",
                }:
                    compute_precisions.append(normalized)
            compute_precision = (
                max(
                    compute_precisions,
                    key=lambda value: BYTES_PER_ELEMENT[value],
                )
                if compute_precisions
                else fallback_precision
            )
            if macs:
                macs_by_precision[compute_precision] += int(macs)

            # ── Step 4: Classify inputs as external vs graph-internal ──
            # Uses memory_reads (already corrected) so no re-scanning needed.
            # Classify each input tensor:
            #   - "weight"        → always external (DRAM read every time)
            #   - graph-internal  → intermediate activation (fusable, skip in fused model)
            #   - other           → external model input (DRAM read)
            #
            # graph-internal = not a weight AND produced by a non-view op in
            # the graph. Transparent views are traced back to their source.
            input_name_list = tensor_names.get("inputs") or []
            graph_internal_input_elems = 0  # intermediate activations from other ops
            external_input_elems = 0  # weights + model-level inputs (always DRAM)
            graph_internal_input_bytes = 0.0
            external_input_bytes = 0.0

            for i, mem_read in enumerate(memory_reads):
                if mem_read <= 0:
                    continue
                itype = input_type_list[i] if i < len(input_type_list) else "weight"
                iname = input_name_list[i] if i < len(input_name_list) else ""

                if itype == "weight":
                    is_graph_internal = False
                elif iname in tensor_producers:
                    producer_id = tensor_producers[iname]
                    source_id = _trace_source_through_views(producer_id)
                    is_graph_internal = (
                        source_id in all_layer_ids
                        and source_id not in transparent_layer_ids
                    )
                else:
                    is_graph_internal = False

                if is_graph_internal:
                    graph_internal_input_elems += mem_read
                    graph_internal_input_bytes += input_bytes[i]
                else:
                    external_input_elems += mem_read
                    external_input_bytes += input_bytes[i]
                    if iname:
                        unique_external_inputs[iname] = max(
                            unique_external_inputs.get(iname, 0), mem_read
                        )
                        unique_external_input_bytes[iname] = max(
                            unique_external_input_bytes.get(iname, 0.0),
                            input_bytes[i],
                        )

            intermediate_input_elems = int(graph_internal_input_elems)
            model_input_elems = int(external_input_elems)
            input_is_intermediate = graph_internal_input_elems > 0

            # Classify outputs: intermediate if consumed by a real
            # non-transparent op, or by views that lead to one.
            output_name_list = tensor_names.get("outputs") or []
            output_is_intermediate = False
            for oname in output_name_list:
                for consumer_id in tensor_consumers.get(oname) or set():
                    if consumer_id not in transparent_layer_ids:
                        output_is_intermediate = True
                        break
                    if _has_real_consumer(consumer_id):
                        output_is_intermediate = True
                        break
                if output_is_intermediate:
                    break

            # Intermediate output elems: written to cache (fused) not DRAM
            intermediate_output_elems = output_elems if output_is_intermediate else 0
            intermediate_output_bytes = (
                float(sum(output_bytes)) if output_is_intermediate else 0.0
            )
            # Total intermediate elems for this layer (inputs + outputs)
            layer_intermediate_elems = (
                intermediate_input_elems + intermediate_output_elems
            )
            layer_intermediate_bytes = (
                graph_internal_input_bytes + intermediate_output_bytes
            )

            # Model output elems: final graph outputs that must go to DRAM
            model_output_elems = output_elems if not output_is_intermediate else 0
            model_output_bytes = (
                float(sum(output_bytes)) if not output_is_intermediate else 0.0
            )
            # Per-op model I/O: external inputs + model outputs (no intermediates)
            model_io_elems = model_input_elems + model_output_elems
            model_io_bytes = external_input_bytes + model_output_bytes

            # Track unique external outputs for deduplication.
            if not output_is_intermediate:
                for i, oname in enumerate(output_name_list):
                    elems = memory_writes[i] if i < len(memory_writes) else 0
                    byte_count = output_bytes[i] if i < len(output_bytes) else 0.0
                    unique_external_outputs[oname] = max(
                        unique_external_outputs.get(oname, 0), int(elems)
                    )
                    unique_external_output_bytes[oname] = max(
                        unique_external_output_bytes.get(oname, 0.0), byte_count
                    )

            # Per-op fused elements: only non-intermediate DRAM traffic
            fused_elems = int(model_io_elems)
            fused_bytes = float(model_io_bytes)

            layers_out[layer_id] = {
                "type": op_type,
                "einsum_equation": equation,
                "is_real_einsum": is_real_einsum,
                "macs": macs,
                "other_ops": other_ops,
                "flops": flops,
                "compute_precision": compute_precision if macs else None,
                "unfused_elements": unfused_elems,
                "unfused_bytes": unfused_bytes,
                "orojenesis_elements": None,
                "fused_elements": fused_elems,
                "fused_bytes": fused_bytes,
                "tensor_shapes": {
                    "inputs": [s for s in input_shapes if isinstance(s, list)],
                    "outputs": [s for s in output_shapes if isinstance(s, list)],
                },
                "tensor_sizes": {
                    "inputs": input_sizes,
                    "outputs": output_sizes,
                },
                "memory_elements": {
                    "inputs": memory_reads,
                    "outputs": memory_writes,
                },
                "memory_bytes": {
                    "inputs": input_bytes,
                    "outputs": output_bytes,
                },
                "tensor_dtypes": {
                    "inputs": input_dtype_list,
                    "outputs": output_dtype_list,
                },
                "tensor_types": {
                    "inputs": list(input_type_list),
                    "outputs": list(output_type_list),
                },
                "input_elements": input_elems,
                "output_elements": output_elems,
                "intermediate_elements": layer_intermediate_elems,
                "intermediate_bytes": layer_intermediate_bytes,
                "model_io_elements": model_io_elems,
                "model_io_bytes": model_io_bytes,
                "input_is_intermediate": input_is_intermediate,
                "output_is_intermediate": output_is_intermediate,
                "is_orphaned": layer_id in _orphaned_layers,
                "connections": {"inputs": input_layer_ids, "outputs": output_layer_ids},
            }

            total_macs += macs
            total_other_ops += other_ops
            total_flops += flops
            total_unfused_elems += unfused_elems
            total_intermediate_elems += layer_intermediate_elems
            total_unfused_bytes += unfused_bytes
            total_intermediate_bytes += layer_intermediate_bytes

        # Deduplicated graph-level external I/O: when the same tensor
        # (e.g. model input x) fans out to multiple ops, count it once.
        # Used for both fused and fused_prefetched totals.
        total_fused_prefetched_elems = int(
            sum(unique_external_inputs.values()) + sum(unique_external_outputs.values())
        )
        # fused_elements == fused_prefetched_elements (same dedup logic)
        total_fused_elems = total_fused_prefetched_elems
        total_fused_bytes = float(
            sum(unique_external_input_bytes.values())
            + sum(unique_external_output_bytes.values())
        )

        # model_io_elements: raw per-op sum (may double-count shared inputs).
        # Kept for diagnostic / per-layer inspection.
        total_model_io_elems = sum(
            layer.get("model_io_elements", 0) for layer in layers_out.values()
        )
        total_model_io_bytes = float(
            sum(layer.get("model_io_bytes", 0) for layer in layers_out.values())
        )

        fusion: dict[str, Any] | None = None
        orojenesis: dict[str, Any] = {
            "status": "not_applicable" if not semantic_graph else "not_requested",
            "layers": {},
        }
        profile: ArchitectureProfile | None = None
        audited_fused_bytes = total_fused_bytes
        audited_prefetched_bytes = total_fused_bytes
        formal_bound = False
        lower_bound_components: dict[str, float] | None = None
        if semantic_graph and semantic_complete:
            if isinstance(architecture, ArchitectureProfile):
                profile = architecture
            elif architecture is not None:
                profile = ArchitectureProfile.load(architecture)
            hierarchy = profile.memory_hierarchy if profile is not None else ()
            fusion = FusionPlanner(graph).plan(hierarchy)
            # The compulsory HBM lower bound is graph external I/O.  Region
            # boundaries and on-chip capacity pressure are not automatically
            # HBM traffic because values may remain in another cache level.
            audited_fused_bytes = total_fused_bytes
            audited_prefetched_bytes = audited_fused_bytes

            einsum_layers = {
                layer_id: layer
                for layer_id, layer in layers_in.items()
                if (layer.get("semantic_op") or {}).get("kind") == "einsum"
            }
            if einsum_layers and orojenesis_runner is None and require_orojenesis:
                raise ValueError(
                    "strict formal analysis requires the pinned Orojenesis toolchain"
                )
            if orojenesis_runner is not None:
                orojenesis["status"] = "complete"
                last_cache = None
                if profile is not None:
                    known = [
                        level
                        for level in profile.memory_hierarchy
                        if level.capacity_bytes is not None and level.name != "vram"
                    ]
                    last_cache = max(
                        known,
                        key=lambda level: int(level.capacity_bytes or 0),
                        default=None,
                    )
                for layer_id, layer in einsum_layers.items():
                    tensor_dtypes = layer.get("tensor_dtypes") or {}
                    dtypes = [
                        *(tensor_dtypes.get("inputs") or []),
                        *(tensor_dtypes.get("outputs") or []),
                    ]
                    # Timeloop uses one global word width.  The minimum tensor
                    # width is conservative for mixed-dtype equations; using
                    # the maximum would overstate the communication bound.
                    bits = min(
                        (int(dtype_bytes(str(dtype)) * 8) for dtype in dtypes),
                        default=int(element_size * 8),
                    )
                    result = orojenesis_runner.run_layer(
                        layer, out_dir / "orojenesis" / layer_id, word_bits=bits
                    )
                    if last_cache is not None:
                        point = select_capacity_point(
                            result["curve"], int(last_cache.capacity_bytes or 0)
                        )
                        if point is None and require_orojenesis:
                            raise ValueError(
                                f"Orojenesis produced no point within "
                                f"{last_cache.name} capacity for {layer_id}"
                            )
                        result["selected_capacity"] = {
                            "level": last_cache.name,
                            "capacity_bytes": last_cache.capacity_bytes,
                            "point": point,
                        }
                    for evidence in result.get("evidence_files", {}).values():
                        evidence["path"] = str(
                            Path("orojenesis") / layer_id / str(evidence["path"])
                        )
                    orojenesis["layers"][layer_id] = result
            elif not einsum_layers:
                orojenesis["status"] = "not_applicable"

            # Consume solver traffic only when the single-einsum proof is
            # applicable to a complete graph endpoint region.  More complex
            # producer/consumer fusion requires the official multi-einsum
            # solver and is rejected in formal mode instead of approximated.
            if einsum_layers and orojenesis["status"] == "complete":
                region_by_layer = {
                    layer_id: region
                    for region in fusion["regions"]
                    for layer_id in region["layers"]
                }
                solver_excesses: list[float] = []
                for layer_id, layer in einsum_layers.items():
                    result = orojenesis["layers"][layer_id]
                    point = (result.get("selected_capacity") or {}).get("point")
                    region = region_by_layer[layer_id]
                    graph_input_contraction = contraction_operands_are_graph_external(
                        layer, all_layers
                    )
                    applicable = bool(point and graph_input_contraction)
                    result["formal_applicability"] = {
                        "applicable": applicable,
                        "region": region["id"],
                        "graph_input_operands": graph_input_contraction,
                        "reason": (
                            "graph_input_contraction"
                            if applicable
                            else "internal_operand_requires_multi_einsum_composition"
                        ),
                    }
                    if not applicable:
                        continue
                    assert point is not None
                    solver_bytes = float(point["dram_bytes"])
                    word_bytes = int(result["word_bits"]) // 8
                    names = layer.get("tensor_names") or {}
                    shapes = layer.get("tensor_shapes") or {}
                    modeled_tensors: dict[str, list[int]] = {}
                    for side in ("inputs", "outputs"):
                        for name, shape in zip(
                            names.get(side) or [], shapes.get(side) or []
                        ):
                            modeled_tensors[str(name)] = list(shape)
                    compulsory_bytes = float(
                        sum(_product(shape) for shape in modeled_tensors.values())
                        * word_bytes
                    )
                    result["audited_dram_bytes"] = solver_bytes
                    result["modeled_compulsory_bytes"] = compulsory_bytes
                    solver_excesses.append(max(0.0, solver_bytes - compulsory_bytes))
                # A maximum is composable without assuming that independent
                # region traffic cannot share cache residency.
                audited_prefetched_bytes = audited_fused_bytes + max(
                    solver_excesses, default=0.0
                )
                applicable_layers = sum(
                    bool((result.get("formal_applicability") or {}).get("applicable"))
                    for result in orojenesis["layers"].values()
                )
                orojenesis["formal_coverage"] = {
                    "applicable_layers": applicable_layers,
                    "total_layers": len(orojenesis["layers"]),
                }
                formal_bound = bool(solver_excesses) and all(
                    (result.get("selected_capacity") or {}).get("point") is not None
                    for result in orojenesis["layers"].values()
                )
            elif not einsum_layers:
                formal_bound = True

        lower_bound_seconds = None
        if profile is not None and semantic_graph and semantic_complete:
            compute_seconds = sum(
                2.0 * float(macs) / profile.peak_for(precision_name)
                for precision_name, macs in macs_by_precision.items()
            )
            fused_memory_seconds = (
                audited_fused_bytes / profile.memory_bandwidth_bytes_per_second
            )
            prefetched_memory_seconds = (
                audited_prefetched_bytes / profile.memory_bandwidth_bytes_per_second
            )
            lower_bound_seconds = max(compute_seconds, prefetched_memory_seconds)
            lower_bound_components = {
                "compute_seconds": compute_seconds,
                "fused_memory_seconds": fused_memory_seconds,
                "fused_unoverlapped_seconds": compute_seconds + fused_memory_seconds,
                "prefetched_memory_seconds": prefetched_memory_seconds,
                "prefetched_overlapped_seconds": lower_bound_seconds,
            }
        if require_orojenesis and not formal_bound:
            raise ValueError(
                "formal analysis did not produce a complete tile-aware bound"
            )

        analysis: Dict[str, Any] = {
            "schema_version": 3 if semantic_graph else 2,
            "layers": layers_out,
            "total": {
                "num_layers": len(layers_out),
                "num_start_nodes_filtered": len(start_node_ids),
                "macs": int(total_macs),
                "other_ops": int(total_other_ops),
                "flops": int(total_flops),
                "macs_by_precision": dict(sorted(macs_by_precision.items())),
                "unfused_elements": int(total_unfused_elems),
                "unfused_bytes": total_unfused_bytes,
                "orojenesis_elements": (
                    None
                    if not orojenesis["layers"]
                    else sum(
                        float(layer_result["selected_capacity"]["point"]["dram_bytes"])
                        / element_size
                        for layer_result in orojenesis["layers"].values()
                        if (layer_result.get("selected_capacity") or {}).get("point")
                    )
                ),
                "fused_elements": int(total_fused_elems),
                "fused_bytes": audited_fused_bytes,
                "fused_prefetched_elements": total_fused_prefetched_elems,
                "fused_prefetched_bytes": audited_prefetched_bytes,
                "prefetched_bytes": audited_prefetched_bytes,
                "io_lower_bound_bytes": audited_prefetched_bytes,
                "lower_bound_seconds": lower_bound_seconds,
                "lower_bound_components": lower_bound_components,
                "model_io_elements": int(total_model_io_elems),
                "model_io_bytes": total_model_io_bytes,
                "intermediate_elements": int(total_intermediate_elems),
                "intermediate_bytes": total_intermediate_bytes,
                "num_intermediate_tensors": len(intermediate_tensors),
                "num_orphaned_layers": len(_orphaned_layers),
            },
            "metadata": {
                "precision": requested_precision,
                "fallback_precision": fallback_precision,
                "bytes_per_element": element_size,
                "dtype_accounting": (
                    "fallback_global" if used_dtype_fallback else "per_tensor"
                ),
                "source_graph": str(src),
                "source_graph_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "fusion": fusion,
                "orojenesis": orojenesis,
                "bound_kind": (
                    "capacity_constrained_tile_aware_v1"
                    if formal_bound and profile is not None
                    else "diagnostic"
                ),
                "architecture": profile.to_dict() if profile is not None else None,
            },
        }

        out_path = out_dir / "analysis.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                analysis,
                f,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
            )

        if self.debug:
            print(f"✅ Wrote analysis: {out_path}")

        return analysis

    # Maps metadata orig_dtypes keywords to Solar precision names
    _QUANT_DTYPE_MAP = {
        "nvfp4": "nvfp4",
        "float4_e2m1fn_x2": "nvfp4",
        "fp8": "fp8",
        "float8_e4m3fn": "fp8",
        "float8_e5m2": "fp8",
        "float8_e4m3fnuz": "fp8",
        "float8_e5m2fnuz": "fp8",
    }

    def _resolve_quant_precision(self, einsum_graph_path: Path) -> Optional[str]:
        """Search for metadata.yaml near the einsum graph and return quant precision.

        Walks up from the einsum_graph_path looking for metadata.yaml
        (max 3 levels). Picks highest-throughput quant dtype (nvfp4 > fp8).
        """
        search_dir = einsum_graph_path.parent
        for _ in range(3):
            candidate = search_dir / "metadata.yaml"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        meta = yaml.safe_load(f) or {}
                except Exception:
                    return None

                best = None
                for conv in meta.get("dtype_conversions") or []:
                    orig = str(conv.get("orig_dtypes", "")).lower()
                    for keyword, prec in self._QUANT_DTYPE_MAP.items():
                        if keyword in orig:
                            if best is None or BYTES_PER_ELEMENT.get(
                                prec, 99
                            ) < BYTES_PER_ELEMENT.get(best, 99):
                                best = prec
                            break
                return best
            search_dir = search_dir.parent
        return None


__all__ = ["EinsumGraphAnalyzer"]
