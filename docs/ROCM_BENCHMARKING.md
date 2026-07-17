<!-- SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ROCm executable benchmarking

SOLAR keeps the paper's published lower bound authoritative:

`TSOL = max(Σp(2 × MACp / published peak throughputp), fused bytes / published memory bandwidth)`

Calibration reports measured throughput separately as `calibrated_solar`; it
never changes the formal denominator. A score is emitted only when correctness
passes, the baseline environment matches exactly, and AMD-SMI reports
`STABLE_PEAK`:

`score = (Tb - TSOL) / ((Tk - TSOL) + (Tb - TSOL))`

`Tb <= TSOL` and `Tk < TSOL` are audit failures and are not clipped.

## Package contracts

`benchmark.yaml` defines a reference Python source with `get_inputs(workload,
device)` and `run(*inputs)`, one or more workloads, tolerance, precision, and
either `cold` or `application` cache behavior. The latest and only accepted
benchmark contract is `schema_version: 2`. Each workload must reference a
schema-v2 `analysis.yaml`, its source `einsum_graph.yaml`, and a hash-bound
`verification.yaml` in-toto statement. The statement binds the reference
source, graph, workload parameters, tolerance, and replayable numerical cases;
the loader reruns those cases before TSOL or SOL Score can be emitted. Manual
workload-level FLOPs and fused-byte totals are rejected. The
artifact contains per-tensor byte traffic and MAC totals grouped by actual
operation precision; the evaluator records the artifact, graph, benchmark,
reference, architecture, solution, and environment identities in the bound
report chain. Loading a benchmark deterministically reruns the analyzer on the
bound graph and compares FLOPs, fused bytes, and the precision breakdown. A
schema-v2 benchmark is rejected if any traffic-bearing tensor lacks an explicit
dtype; neighboring quantization sidecars cannot override the artifact.

The evaluator injects changing integer seeds and runs ten fresh correctness
rounds before timing. Timed calls receive non-repeating aligned tensor
addresses; every timed output is checked after its device event, followed by
ten fresh post-timing rounds.
Input mutation, persistent monkey patches, and evaluator-visible worker threads
are candidate failures.

`solution.yaml` selects `pytorch`, `triton`, `hip_cpp`, `hipblas`, `miopen`,
`ck`, or `rocwmma`; declares one or more `gfx*` targets; and lists relative source paths plus
SHA-256. Python entry points use `file.py::function`. Native backends also
provide an argv-style `compile.command`, with `{python}`, `{staging}`, and
`{gfx_target}` placeholders, and must produce one top-level Python extension
`.so`. The detected target must match both `--arch-config` and the solution.

`baseline.yaml` binds workload latency to exact benchmark, environment, timing,
cache, gfx target, clock evidence, and baseline solution hashes. There is no
implicit or nearest-match baseline selection.

The resulting `evaluation.yaml` contains raw samples, p20/p50/p80/p95, IQR,
mean, population standard deviation, theoretical and calibrated bounds,
correctness, score, capabilities, and structured failure states.

## Timing profiles

- `standard`: one untimed initialization; warmup ≥10 calls and ≥200 ms;
  measurement ≥30 samples and ≥1 s; continue until IQR/median ≤5%, capped at
  10 s or 100,000 samples.
- `official`: the same initialization/warmup followed by five independently
  stable windows, each ≥20 samples and ≥600 ms. The reported latency is the
  median of the five window medians.
- `quick`: 25 ms warmup and 100 ms sampling. It cannot create a formal baseline
  or SOL Score.

Cold-cache measurement touches at least twice the larger of the profile's L2
and last-level cache before every timed call. For the RX 9060 XT profile this
means 64 MiB, covering its declared 32 MiB last-level cache. Cold-cache mode
disables batching so no call can reuse another call's cache state. Evaluator
timing also disables batching for application-cache runs because output
validation and unique-address inputs are per invocation.
Input construction and cache clearing are outside device-event intervals. Raw
samples are retained without outlier deletion. Run `rocprofv3` separately so profiler
overhead cannot contaminate score timing.

## Isolation and native libraries

Trusted development solutions may run locally. Use `solar-evaluate --untrusted`
for external or publishable solutions; it mounts sources read-only in
`solar-rocm:7.2` and exposes only `/dev/kfd`, `/dev/dri`, and the selected output
directory. Entrypoints must name a declared staged source; static scanning adds
defense in depth against process, network, and arbitrary file access.
Missing hipBLAS, MIOpen, CK, or rocWMMA headers are reported as capability
failures instead of falling back to another backend.

## Included end-to-end examples

`examples/rocm_matmul` contains hash-verified PyTorch, Triton, and HIP C++
solutions for the same FP16 benchmark. The HIP manifest compiles a CPython
extension with `{python}` and `{gfx_target}`; the loader imports native modules
under their build artifact name so the exported `PyInit_*` symbol matches.
These examples exercise correctness gating and device-event timing on a real
ROCm device rather than only validating YAML contracts.
