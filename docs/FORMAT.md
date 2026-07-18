<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Solar YAML Format Reference

This document defines the exact dictionary schema for every YAML file produced by the Solar pipeline. All stages **must** follow these schemas; any code that emits or reads these files should reference this document.

---

## Pipeline Stages

```
Model.py
  │  solar.cli.process_model
  ▼
pytorch_graph.yaml          (Stage 1: torchview extraction)
  │  solar.cli.toeinsum_model
  ▼
einsum_graph.yaml           (Stage 2: einsum conversion)
einsum_graph_renamed.yaml   (Stage 2b: BFS rank rename)
  │  solar.cli.analyze_model
  ▼
analysis.yaml               (Stage 3: metrics + optional formal architecture bound)
  │  solar.cli.predict_perf_model
  ▼
perf_<arch>.yaml            (Stage 4: roofline performance prediction)
```

---

## 1. `pytorch_graph.yaml` — Torchview Graph

Produced by `solar.graph.pytorch_processor`.
Each layer is keyed by its hierarchical torchview node ID.

### Top-level

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Name of the model class |
| `layers` | `dict[str, Layer]` | All nodes (ops + tensor nodes) |

### Layer dict

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `str` | yes | Operation or tensor type (e.g. `matmul`, `conv2d`, `auxiliary-tensor`, `parameter-tensor`, `output-tensor`) |
| `node_class` | `str` | yes | Torchview class: `FunctionNode`, `TensorNode`, etc. |
| `input_shapes` | `list[list[int]]` | yes | Shapes of each input tensor |
| `output_shapes` | `list[list[int]]` | yes | Shapes of each output tensor |
| `input_dtypes` | `list[str]` | yes | PyTorch dtype strings (e.g. `torch.float32`) |
| `output_dtypes` | `list[str]` | yes | PyTorch dtype strings |
| `input_types` | `list[str]` | yes | Semantic role: `input`, `weight`, `bias` |
| `output_types` | `list[str]` | yes | Semantic role: `output` |
| `module_args` | `dict` | yes | Op-specific args (`function_name`, `stride`, `padding`, `groups`, `hierarchical_name`, `raw_attributes`, etc.) |
| `connections` | `dict` | yes | `{inputs: [str], outputs: [str]}` — node IDs |

---

## 2. `einsum_graph.yaml` — Einsum Graph

Produced by `solar.einsum.pytorch_to_einsum`.
This is the canonical intermediate representation. **All layers — including subgraph expansions — must use this exact schema.**

### Top-level

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `int` | `3` for executable official graphs |
| `model_name` | `str` | Name of the model |
| `layers` | `dict[str, Layer]` | All layers (start nodes + op nodes) |
| `outputs` | `list[str]` | Optional explicit ordered graph output tensor names |

### Layer dict (every layer)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `str` | yes | Op type: `start`, `matmul`, `conv2d`, `add`, `relu`, `view`, etc. |
| `einsum_equation` | `str` | yes | Extended einsum string (e.g. `MK,KN->MN`, `BC(P+R)(Q+S),OCRS->BOPQ`, `AB->AB`) |
| `elementwise_op` | `str` | yes | Pointwise operation: `mul`, `add`, `sub`, `div`, `relu`, `copy`, etc. |
| `reduction_op` | `str` | yes | Reduction operation: `add`, `max`, `min`, `mul`, `none` |
| `is_real_einsum` | `bool` | yes | `true` for MAC-producing ops (matmul, conv); `false` for elementwise/view/reduction |
| `is_einsum_supportable` | `bool` | yes | Whether the op can be represented as an einsum |
| `tensor_names` | `dict` | yes | `{inputs: [str], outputs: [str]}` — unique tensor name per slot (e.g. `Model.matmul.Weight`, `Model.matmul.Output`) |
| `tensor_types` | `dict` | yes | `{inputs: [str], outputs: [str]}` — semantic role per slot: `input`, `weight`, `bias`, `output` |
| `tensor_shapes` | `dict` | yes | `{inputs: [list[int]], outputs: [list[int]]}` — shape per slot |
| `tensor_dtypes` | `dict` | yes | `{inputs: [str], outputs: [str]}` — explicit dtype per slot |
| `connections` | `dict` | yes | `{inputs: [str], outputs: [str]}` — layer IDs (not tensor names) |
| `semantic_op` | `dict` | yes | Executable `kind`, target/equation, ordered arguments, kwargs, and effects |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `taco_expression` | `str` | TACO index notation equivalent |
| `raw_attributes` | `str` | Original torchview raw_attributes string |
| `graph_signature` | `dict` | AOT joint-graph inputs/outputs, saved tensors, gradients, and mutation maps |

### Key conventions

- **`tensor_names`**, **`tensor_types`**, and **`tensor_shapes`** are parallel arrays: `tensor_names.inputs[i]` describes the same tensor as `tensor_shapes.inputs[i]` and `tensor_types.inputs[i]`.
- **`connections`** references layer IDs (dict keys), not tensor names.
- **`start` nodes** represent model inputs. They have `tensor_shapes.inputs: []` and `tensor_shapes.outputs: [[shape]]`. They are filtered out before analysis.
- **Subgraph expansions** (e.g. grouped conv → reshape_input + reshape_weight + conv + reshape_output) must emit the same schema as regular layers. Never use flat `input_shapes`/`output_shapes` keys — those belong to `pytorch_graph.yaml` only.

---

## 3. `analysis.yaml` — Analysis and Formal Bound

Produced by `solar.analysis.graph_analyzer`.

### Top-level

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `int` | `3` for formal executable-semantic analysis |
| `layers` | `dict[str, Layer]` | Per-layer analysis (start nodes excluded) |
| `total` | `dict` | Graph-wide totals |
| `metadata` | `dict` | Precision, provenance, fusion, solver, architecture and bound kind |

### Layer dict

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Op type |
| `einsum_equation` | `str` | Einsum string |
| `is_real_einsum` | `bool` | MAC-producing or not |
| `macs` | `int` | Multiply-accumulate operations (non-zero only for `is_real_einsum: true`) |
| `other_ops` | `int` | Scalar/vector elementwise and reduction operations |
| `flops` | `int` | `2 * macs` |
| `resources` | `dict` | Versioned per-layer resource classification: `work`, `classification`, exact formulas, and an exemption reason where applicable |
| `unfused_elements` | `int` | `input_elements + output_elements` (all DRAM traffic if nothing fused) |
| `orojenesis_elements` | `float \| null` | Diagnostic selected solver traffic in fallback element units |
| `fused_elements` | `int` | External I/O only (intermediates excluded) |
| `tensor_shapes` | `dict` | `{inputs: [...], outputs: [...]}` — copied from einsum graph |
| `tensor_sizes` | `dict` | `{inputs: [int], outputs: [int]}` — product of each shape |
| `memory_elements` | `dict` | `{inputs: [int], outputs: [int]}` — corrected DRAM reads/writes per tensor |
| `tensor_types` | `dict` | `{inputs: [str], outputs: [str]}` — `input`, `weight`, `output` |
| `input_elements` | `int` | Sum of `memory_elements.inputs` |
| `output_elements` | `int` | Sum of `memory_elements.outputs` |
| `intermediate_elements` | `int` | Elements that stay in cache (fusable) |
| `model_io_elements` | `int` | External inputs + model outputs (no intermediates) |
| `input_is_intermediate` | `bool` | True if any input comes from another op in the graph |
| `output_is_intermediate` | `bool` | True if output is consumed by another op |
| `connections` | `dict` | `{inputs: [str], outputs: [str]}` |

### `total` dict

| Field | Type | Description |
|-------|------|-------------|
| `num_layers` | `int` | Number of computation layers (excludes start) |
| `num_start_nodes_filtered` | `int` | Start nodes removed |
| `macs` | `int` | Sum of all layer MACs |
| `other_ops` | `int` | Sum of all layer other_ops |
| `flops` | `int` | `2 * macs` |
| `macs_by_precision` | `dict[str, int]` | Matrix work grouped by actual operation precision |
| `resource_work` | `dict[str, dict[str, int]]` | Hardware-independent work grouped by AMD resource and mode |
| `resource_seconds` | `dict[str, float]` | Per-resource `sum(work / published_rate)` using the bound profile |
| `compute_resource` | `str \| null` | Resource with the largest formal compute time |
| `unfused_elements` | `int` | Sum across layers |
| `orojenesis_elements` | `float \| null` | Sum of selected solver traffic in fallback element units |
| `fused_elements` | `int` | Deduplicated external I/O (shared tensors counted once) |
| `fused_bytes` | `float` | Deduplicated graph-external compulsory I/O |
| `fused_prefetched_elements` | `int` | Compatibility element count for graph-external I/O |
| `fused_prefetched_bytes` | `float` | Formal tile-aware traffic when available |
| `io_lower_bound_bytes` | `float` | Compulsory I/O plus safely composable solver excess |
| `lower_bound_seconds` | `float \| null` | `max(max(resource_seconds), io_lower_bound / bandwidth)` |
| `lower_bound_components` | `dict \| null` | Per-resource compute, bottleneck resource, fused/prefetched memory, and overlap components |
| `model_io_elements` | `int` | Per-op sum (may double-count shared inputs; diagnostic) |
| `intermediate_elements` | `int` | Total fusable elements |
| `num_intermediate_tensors` | `int` | Count of intermediate tensor names |

### `metadata` dict

| Field | Type | Description |
|-------|------|-------------|
| `precision` | `str` | e.g. `fp16`, `fp32`, `bf16`, `fp8` |
| `bytes_per_element` | `float` | Bytes per tensor element for this precision |
| `source_graph` | `str` | Path to the einsum graph that was analyzed |
| `source_graph_sha256` | `str` | Hash of the exact source graph |
| `fusion` | `dict \| null` | Edge legality decisions, regions, liveness, and hierarchy pressure |
| `orojenesis` | `dict` | Hash-bound mapper toolchain provenance plus pinned single-layer, canonical-chain, and extended-region problem/mapper/raw mapping evidence, selected points and coverage |
| `architecture` | `dict \| null` | Architecture profile used for the formal bound |
| `resource_model` | `dict` | Version, modeled/exempt/unclassified coverage, and fail-closed state |
| `bound_kind` | `str` | `capacity_constrained_tile_aware_v1` or `diagnostic` |

---

## 4. `perf_<arch>.yaml` — Roofline Performance Prediction

Produced by `solar.perf.perf_model`.

### Top-level sections

| Section | Description |
|---------|-------------|
| `arch` | Hardware config used |
| `workload` | Model-level compute/memory summary |
| `unfused` | Whole-graph roofline using all operation tensor traffic |
| `fused` | Whole-graph roofline using deduplicated external I/O |
| `fused_prefetched` | Tile-aware traffic when formal evidence exists; otherwise diagnostic fallback |
| `memory_breakdown` | Weight vs activation vs intermediate split |
| `speedup` | Ratio between models |
| `memory_reduction` | Fraction of memory saved by fusion |

### `arch` dict

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Architecture config name (for example `Radeon_RX_9060_XT`) |
| `vendor` | `str` | Architecture vendor; bundled and accepted profiles use `AMD` |
| `gfx_target` | `str` | AMD ROCm target, for example `gfx1200` |
| `clock_hz` | `float` | Clock frequency in Hz |
| `memory_bandwidth_bytes_per_second` | `float` | Published memory bandwidth |
| `throughput_precision` | `str` | Precision used for matrix throughput |
| `operations_per_cycle` | `float` | Matrix-operation throughput used for cycle diagnostics |
| `scalar_operations_per_cycle` | `float` | Scalar/vector throughput used for diagnostics |
| `peak_ops_per_second` | `mapping` | Precision-keyed published operation throughput |
| `resource_model_version` | `str` | Version matching analysis resource counters |
| `resource_limits` | `mapping` | Published upper rate for every resource/mode |
| `resource_limit_sources` | `mapping` | Source URL or derivation for each resource limit |
| `calibration_exempt_modes` | `mapping` | Unmeasured resource modes with mandatory limitation reasons |
| `precision_support` | `mapping` | Per-precision hardware, software-maturity, evidence, and calibration policy |
| `profile_revision` | `str` | Immutable resource-profile revision |
| `audit_evidence` | `mapping` | Hash-bound local hardware audit identity |
| `ridge_point` | `float` | Arithmetic intensity at roofline knee |

### `workload` dict

| Field | Type | Description |
|-------|------|-------------|
| `total_macs` | `int` | Total MACs from analysis |
| `total_other_ops` | `int` | Total elementwise/reduction ops |
| `total_flops` | `int` | `2 * total_macs` |
| `resource_model_version` | `str` | Version used by the source analysis |
| `resource_work` | `mapping` | Exact graph resource counters |
| `resource_cycles` | `mapping` | Per-resource time converted using the profile clock |
| `bytes_per_element` | `float` | Effective bytes per element |
| `quant_orig_dtype` | `str` | *(optional)* Original quantized dtype if metadata present |

### Roofline model dicts (`unfused`, `fused`, `fused_prefetched`)

| Field | Type | Description |
|-------|------|-------------|
| `description` | `str` | Human-readable model description |
| `memory_elements` | `int` | Total DRAM-accessed elements |
| `memory_bytes` | `int` | `memory_elements * bytes_per_element` |
| `compute_matrix_cycles` | `int` | Matrix-operation cycles (informational) |
| `compute_scalar_cycles` | `int` | Largest non-MFMA resource time expressed in cycles |
| `compute_cycles` | `int` | Maximum across all modeled AMD resource cycles used in SOL |
| `memory_cycles` | `int` | Memory time converted to diagnostic cycles using the normalized profile |
| `total_cycles` | `int` | `max(compute_cycles, memory_cycles)` |
| `runtime_ms` | `float` | `max(max(resource_cycles), memory_cycles) / clock_hz * 1000` |
| `arithmetic_intensity` | `float` | `total_macs / memory_bytes` |
| `bottleneck` | `str` | `"compute"` or `"memory"` |

### `memory_breakdown` dict

| Field | Type | Description |
|-------|------|-------------|
| `weight_elements` | `int` | Total weight tensor elements |
| `weight_bytes` | `int` | Weight bytes |
| `model_io_elements` | `int` | Model input/output elements |
| `model_io_bytes` | `int` | Model I/O bytes |
| `intermediate_elements` | `int` | Fusable intermediate elements |
| `intermediate_bytes` | `int` | Intermediate bytes |

### `speedup` / `memory_reduction` dicts

| Field | Type | Description |
|-------|------|-------------|
| `fused_vs_unfused` | `float` | Speedup or memory fraction saved |
| `fused_prefetched_vs_unfused` | `float` | |
| `fused_prefetched_vs_fused` | `float` | |
