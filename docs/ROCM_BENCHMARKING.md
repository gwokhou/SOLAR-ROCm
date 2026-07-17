<!-- SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ROCm executable benchmarking

SOLAR keeps the paper's published lower bound authoritative:

`TSOL = max(total FLOPs / published peak throughput, fused bytes / published memory bandwidth)`

Calibration reports measured throughput separately as `calibrated_solar`; it
never changes the formal denominator. A score is emitted only when correctness
passes, the baseline environment matches exactly, and AMD-SMI reports
`STABLE_PEAK`:

`score = (Tb - TSOL) / ((Tk - TSOL) + (Tb - TSOL))`

`Tb <= TSOL` and `Tk < TSOL` are audit failures and are not clipped.

## Package contracts

`benchmark.yaml` defines a reference Python source with `get_inputs(workload,
device)` and `run(*inputs)`, one or more workloads, FLOPs, fused bytes,
tolerance, precision, and either `cold` or `application` cache behavior.
The evaluator injects a changing integer `seed` into the workload mapping and
runs three correctness seeds before timing.

`solution.yaml` selects `pytorch`, `triton`, `hip_cpp`, `hipblas`, `miopen`,
`ck`, or `rocwmma`; declares `gfx1200`; and lists relative source paths plus
SHA-256. Python entry points use `file.py::function`. Native backends also
provide an argv-style `compile.command`, with `{staging}` and `{gfx_target}`
placeholders, and must produce one top-level Python extension `.so`.

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

Cold-cache measurement touches at least twice the 4 MiB L2 before every timed
call and disables batching so no call can reuse another call's cache state.
Input construction and cache clearing are outside device-event intervals. Raw
samples are retained without outlier deletion. Kernels faster than 1 ms use an
adaptive blocked sample in application-cache mode and report both its batch
size and normalized per-call latency. Run `rocprofv3` separately so profiler
overhead cannot contaminate score timing.

## Isolation and native libraries

Trusted development solutions may run locally. Use `solar-evaluate --untrusted`
for external or publishable solutions; it mounts sources read-only in
`solar-rocm:7.2` and exposes only `/dev/kfd`, `/dev/dri`, and the selected output
directory. Entrypoints must name a declared staged source; static scanning adds
defense in depth against process, network, and arbitrary file access.
Missing hipBLAS, MIOpen, CK, or rocWMMA headers are reported as capability
failures instead of falling back to another backend.
