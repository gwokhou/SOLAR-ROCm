<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Verification Guide

SOLAR has two maintained verification layers:

1. graph/einsum regression tests verify that PyTorch operations are converted
   into the expected equations, shapes, costs, and memory accounting; and
2. the ROCm evaluator executes a candidate and its benchmark reference on the
   same AMD device, comparing outputs before any timing or score is accepted.

The original source and SPDX notices remain intact throughout the analysis
pipeline. Verification does not require translating PyTorch's public
`torch.cuda` API names: ROCm PyTorch intentionally exposes HIP devices through
that compatibility namespace.

## Graph and einsum verification

Run the complete deterministic suite from the repository root:

```bash
bash scripts/run_tests.sh all
```

For a focused conversion and cost check:

```bash
.venv/bin/python -m pytest \
  tests/test_einsum_analyzer.py \
  tests/test_pytorch_to_einsum_regressions.py \
  tests/test_graph_analyzer_regression.py \
  tests/test_integration.py -v
```

These tests cover equation generation, rank renaming, tensor shapes, MAC/FLOP
counts, fused/unfused memory totals, orphan pruning, zero-compute operations,
and end-to-end analysis/performance output.

The maintained example pipelines provide an additional artifact-level check:

```bash
bash scripts/run_tests.sh examples
```

The runner verifies that each example produces `pytorch_graph.yaml`,
`einsum_graph_renamed.yaml`, `analysis.yaml`, and the default AMD performance
prediction.

## Executable ROCm correctness

`solar-evaluate` loads the benchmark reference and candidate from separately
hashed manifests. Before timing, it constructs inputs with three distinct
seeds and compares every output using the benchmark's declared `atol` and
`rtol`. A failed comparison prevents latency and SOL score publication.

Quick local verification:

```bash
.venv/bin/solar-evaluate \
  --benchmark examples/rocm_matmul/benchmark.yaml \
  --solution examples/rocm_matmul/triton_solution.yaml \
  --timing-profile quick \
  --no-lock-clocks \
  --output /tmp/solar-triton-verification.yaml
```

The same benchmark includes three independently hashed AMD paths:

- `solution.yaml`: PyTorch on ROCm;
- `triton_solution.yaml`: Triton ROCm JIT; and
- `hip_solution.yaml`: a HIP C++ extension built for the detected `gfx*`
  target.

Replace `--solution` to verify each implementation. A successful smoke result
has top-level `failure: null` and every workload has `correct: true`. The
`quick` timing profile is intentionally not publishable; use `standard` or
`official` with verified AMD-SMI `STABLE_PEAK` locking for baseline or score
artifacts.

For untrusted sources, run the same checks in the pinned container:

```bash
.venv/bin/solar-evaluate \
  --untrusted \
  --image solar-rocm:7.2 \
  --benchmark examples/rocm_matmul/benchmark.yaml \
  --solution examples/rocm_matmul/triton_solution.yaml \
  --timing-profile quick \
  --no-lock-clocks \
  --output /tmp/solar-container-verification.yaml
```

See [`ROCM_BENCHMARKING.md`](ROCM_BENCHMARKING.md) for timing stability,
cache policy, clock evidence, baseline compatibility, and publishability
requirements.

## Environment verification

Confirm that the selected AMD profile matches the detected device and that all
optional ROCm backends are visible:

```bash
.venv/bin/solar-rocm-doctor --json
```

For the default profile, the report should identify the RX 9060 XT as
`gfx1200`; `pytorch_rocm`, `triton_rocm`, `hipcc`, `rocprofv3`, `amd_smi`,
`hipblas`, `miopen`, `ck`, and `rocwmma` should be available. Other AMD targets
are accepted when a matching `--arch-config` and solution `gfx_targets` entry
are supplied.

## Legacy verifier wrappers

`solar/cli/verify_einsum.py` and `verify_einsum_model.py` are preserved from
the original source tree for compatibility, but their external
`solar_verifier` execution package and the old
`run_kernelbench_einsum_verification.sh` wrapper are not shipped in this
repository. They are therefore not registered as supported console commands.
Use the maintained pytest and ROCm evaluator paths above rather than relying on
those optional legacy wrappers.
