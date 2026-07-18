<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SOL (Speed-of-Light) Performance Models Guide

This guide explains the three roofline-based performance models used in Solar for predicting DNN execution time.

## Overview

Solar computes three SOL (Speed-of-Light) performance estimates based on the roofline model. Each model makes different assumptions about memory access patterns:

| Model | Memory Accesses | Roofline Application | Use Case |
|-------|-----------------|---------------------|----------|
| **Unfused** | All tensors (Input + Weight + Output) per op | Whole-graph roofline on summed totals | Baseline / worst case |
| **Fused** | Weights + model I/O only (intermediates excluded) | Whole-graph roofline on summed totals | Operator fusion |
| **Fused+Prefetched** | Compulsory graph I/O + capacity-constrained tile-aware excess | Whole-graph roofline on summed totals | Formal lower bound / perfect overlap |

> **Implementation note**: All three models apply a single whole-graph roofline:
> `max(total_compute_cycles, total_memory_cycles)`. In schema-v3 formal analysis,
> `fused` is the compulsory graph-external floor while `fused_prefetched` is the
> audited `io_lower_bound_bytes` produced from pinned Orojenesis evidence. Without
> a formal solver result, the latter falls back to `fused` for diagnostic output
> only and cannot be used for official scoring.

## Roofline Model Basics

The roofline model predicts performance based on two hardware limits:
- **Compute bound**: Limited by published peak compute throughput (operations/second)
- **Memory bound**: Limited by memory bandwidth (bytes/second)

```
runtime_seconds = max(compute_seconds, memory_seconds)
```

Where:
- `resource_seconds[r] = Σmode(resource_work[r, mode] / published_rate[r, mode])`
- `compute_seconds = max_r(resource_seconds[r])`
- `memory_seconds = total_memory_bytes / memory_bandwidth_bytes_per_second`
- **Arithmetic Intensity** = total_macs / total_memory_bytes

The versioned AMD resource set is MFMA, VALU, SFU, reduction, atomic,
scan/sort, and conversion. This makes casts, dequantization, accumulation,
normalization, indexed updates, and other non-matrix work part of the formal
bound. Operations sharing a resource serialize; independent resources may
overlap. Official analysis rejects an unclassified executable compute node.

The YAML output also reports cycles for diagnostics. Solar derives those from
the normalized AMD profile clock; the formal lower bound is the per-second
equation above.

---

## 1. Unfused SOL

### Description
All tensor accesses (inputs, weights, outputs) for every operation are assumed to come from DRAM. This produces the highest memory traffic estimate.

### Memory Calculation ([`graph_analyzer.py`](../solar/analysis/graph_analyzer.py))
```
Per layer:  unfused_elements_i = input_elems_i + output_elems_i
Total:      unfused_elements   = Σ_i unfused_elements_i
            unfused_bytes      = unfused_elements × bytes_per_element
```

### Roofline Application ([`perf_model.py`](../solar/perf/perf_model.py))
```
unfused_runtime_ms = max(
    max_r(resource_seconds[r]),
    unfused_bytes / memory_bandwidth_bytes_per_second,
) × 1000
```

A single whole-graph roofline is applied to graph-level totals. Intermediate tensors are counted as DRAM traffic (read by consumer, written by producer).

### When to Use
- Baseline performance estimate
- No kernel fusion
- Memory-bound workloads with poor data reuse
- Debugging / understanding memory bottlenecks

### Example
For a simple `Linear → ReLU → Linear` network:
```
Layer 1 (Linear): Read Input + Weight, Write Output → DRAM
Layer 2 (ReLU):   Read Input (from DRAM), Write Output → DRAM
Layer 3 (Linear): Read Input (from DRAM) + Weight, Write Output → DRAM
```

---

## 2. Fused SOL

### Description
Intermediate tensor accesses are **excluded** from memory cost. Only weights and model-boundary I/O (global inputs/outputs) are counted.

### Memory Calculation ([`graph_analyzer.py`](../solar/analysis/graph_analyzer.py))
```
Graph level:
  external_inputs  = deduplicate(weight and model-input tensors)
  external_outputs = deduplicate(outputs with no downstream consumer)
  fused_elements   = Σ external_inputs + Σ external_outputs
  fused_bytes      = fused_elements × bytes_per_element
```

### Roofline Application ([`perf_model.py`](../solar/perf/perf_model.py))
```
fused_runtime_ms = max(
    max_r(resource_seconds[r]),
    fused_bytes / memory_bandwidth_bytes_per_second,
) × 1000
```

A single whole-graph roofline is applied. Intermediate tensors are assumed to stay in cache/registers.

### When to Use
- Operator fusion scenarios (for example, MIOpen or compiler-fused kernels)
- When intermediate tensors fit in L2 cache
- Realistic estimate for modern GPU execution

### Example
For a simple `Linear → ReLU → Linear` network:
```
Layer 1 (Linear): Read Model_Input + Weight, intermediate stays in cache
Layer 2 (ReLU):   No DRAM access (intermediate in cache)
Layer 3 (Linear): Read Weight, Write Model_Output
```

---

## 3. Fused+Prefetched SOL

### Description
A **single roofline** is applied to the entire graph. Total compute is combined
with the compulsory graph I/O plus safely composable tile-aware excess traffic,
assuming perfect overlap between compute and memory operations.

### Memory Calculation ([`graph_analyzer.py`](../solar/analysis/graph_analyzer.py))
```
compulsory_bytes = deduplicated graph-external tensor bytes
solver_excess_i  = max(0, selected_OAVES_dram_bytes_i
                           - modeled_compulsory_bytes_i)
io_lower_bound_bytes = compulsory_bytes + max(safely_composable solver_excess_i)
fused_prefetched_bytes = io_lower_bound_bytes
```

The maximum excess is composable without assuming independent regions cannot
share cache residency. Exact einsums whose operands come from internal
non-alias producers require a multi-einsum proof and are marked non-applicable;
formal scoring fails if no solver layer is safely composable.

### Roofline Application ([`perf_model.py`](../solar/perf/perf_model.py))
```
fused_prefetched_runtime_ms = max(
    max_r(resource_seconds[r]),
    fused_prefetched_bytes / memory_bandwidth_bytes_per_second,
) × 1000
```

### When to Use
- Best-case performance estimate
- Highly optimized implementations with prefetching
- Compute-bound workloads
- Upper bound on achievable performance

### Example
For a simple `Linear → ReLU → Linear` network:
```
Total FLOPs  = FLOPs(Linear1) + FLOPs(ReLU) + FLOPs(Linear2)
Total Memory = Model_Input + Weight1 + Weight2 + Model_Output
Single roofline applied to (Total FLOPs, Total Memory)
```

---

## 4. Comparison Summary

| Aspect | Unfused | Fused | Fused+Prefetched |
|--------|---------|-------|------------------|
| Intermediate tensors | Counted | Excluded | Reflected through solver excess |
| Roofline granularity | Whole graph | Whole graph | Whole graph |
| Memory assumption | All from DRAM | Compulsory floor only | Capacity-constrained tile reuse |
| Typical speedup | 1.0x (baseline) | Diagnostic | Formal target |
| Realism | Conservative traffic scenario | Optimistic floor | Audited lower bound |

## 5. Output Fields

### analysis.yaml
```yaml
total:
  macs: 1000000                    # Total multiply-accumulate operations
  flops: 2000000                   # Total floating-point operations (2 × MACs)
  other_ops: 50000                 # Total scalar/vector elementwise/reduction ops
  resource_work:                   # Exact hardware-independent counters
    mfma: {fp16->fp32: 2000000}
    reduction: {fp32: 50000}
  resource_seconds:                # Work divided by published profile rates
    mfma: 0.000001
    reduction: 0.000002
  compute_resource: reduction      # Maximum resource time determines compute SOL
  unfused_elements: 25000000       # Unfused memory elements (all tensor I/O)
  fused_elements: 10000000         # Fused memory elements (intermediates excluded)
  fused_prefetched_elements: 12000000  # Tile-aware I/O lower-bound elements
  fused_bytes: 20000000                # Explicit per-tensor compulsory bytes
  io_lower_bound_bytes: 24000000       # Compulsory + solver excess
  lower_bound_seconds: 0.000012        # Formal compute/memory lower bound
  model_io_elements: 2500000       # Model input/output elements
  intermediate_elements: 15000000  # Intermediate tensor elements
  weight_elements: 7500000         # Weight tensor elements
```

### perf_<arch>.yaml
```yaml
unfused:
  memory_bytes: 50000000
  compute_cycles: 2645
  memory_cycles: 49049
  total_cycles: 49049
  runtime_ms: 0.025
  arithmetic_intensity: 0.04
  bottleneck: memory

fused:
  memory_bytes: 20000000
  compute_cycles: 2645
  memory_cycles: 19619
  total_cycles: 19619
  runtime_ms: 0.010
  arithmetic_intensity: 0.1
  bottleneck: memory

fused_prefetched:
  memory_bytes: 24000000
  compute_cycles: 2645
  memory_cycles: 23543
  total_cycles: 23543
  runtime_ms: 0.012
  arithmetic_intensity: 0.083
  bottleneck: memory

speedup:
  fused_vs_unfused: 2.5
  fused_prefetched_vs_unfused: 2.08
  fused_prefetched_vs_fused: 0.83
```

## 6. Known Boundaries

All three diagnostic views use a whole-graph roofline. `unfused` and `fused`
are useful comparisons, but only schema-v3 `fused_prefetched` backed by a
complete `capacity_constrained_tile_aware_v1` analysis is accepted as a formal
benchmark denominator.

The current formal composition uses pinned single-einsum OAVES proofs. It does
not add excess traffic across independent regions and does not approximate an
einsum fed by a materializing internal producer. Those cases require an
official multi-einsum solver; until then they are recorded as non-applicable
and a graph with no applicable proof is rejected from scoring.


## Practical Guidance

1. **Use Unfused** when:
   - Evaluating baseline performance
   - No fusion optimizations available
   - Debugging memory bottlenecks

2. **Use Fused** when:
   - Inspecting the compulsory graph-external I/O floor
   - Comparing how much traffic a legal fusion plan can remove
   - Diagnosing solver excess relative to compulsory bytes

3. **Use Fused+Prefetched** when:
   - The analysis has `bound_kind: capacity_constrained_tile_aware_v1`
   - Setting the formal workload TSOL target
   - Auditing the selected capacity point and compute/memory overlap

## References

- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures.
- AMD ROCm documentation on operator fusion and library kernels
- PyTorch compile / TorchInductor fusion strategies
