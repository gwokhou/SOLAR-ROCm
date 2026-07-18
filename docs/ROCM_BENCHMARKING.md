<!-- SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ROCm executable benchmarking

SOLAR keeps published architecture limits authoritative and accounts for every
executable compute node with the versioned `amd_resource_v1` model:

`resource_time[r] = Σmode(resource_work[r, mode] / published_rate[r, mode])`

`TSOL = max(max_r(resource_time[r]), io_lower_bound_bytes / published memory bandwidth)`

The complete formal resource set is MFMA, VALU, SFU, reduction, atomic,
scan/sort, and conversion (including casts, dequantization, and accumulation
modes). Work sharing a resource is summed; different resources and memory are
allowed perfect overlap. An executable node that is neither exactly modeled
nor explicitly memory/view-only makes official analysis fail closed. MAC/FLOP
totals remain useful diagnostics, but they are not a substitute for non-MFMA
work in TSOL.

`io_lower_bound_bytes` is the deduplicated graph-external compulsory traffic
plus the safely composable capacity-constrained Orojenesis excess. Plain
`fused_bytes` remains a diagnostic compulsory-I/O component.

Calibration reports measured throughput separately as `calibrated_solar`; it
never changes the formal denominator. A score is emitted only when correctness
passes, the baseline environment matches exactly, and AMD-SMI reports
`STABLE_PEAK`:

`score = (Tb - TSOL) / ((Tk - TSOL) + (Tb - TSOL))`

`Tb <= TSOL` and `Tk < TSOL` are audit failures and are not clipped. In a
publishable, clock-locked run, a correct measured p50 below TSOL is emitted as
`status: bound_violation` with hashes, timings, and the observed/theoretical
ratio; the score is withheld pending a resource/profile audit.

## Package contracts

`benchmark.yaml` defines a reference Python source with `get_inputs(workload,
device)` and `run(*inputs)`, one or more workloads, tolerance, precision, and
either `cold` or `application` cache behavior. The latest and only accepted
benchmark contract is `schema_version: 3`. Each compatible workload must reference a
schema-v3 `analysis.yaml`, its source `einsum_graph.yaml`, and a hash-bound
`verification.yaml` in-toto statement. The statement binds the reference
source, graph, workload parameters, tolerance, and replayable numerical cases;
the loader reruns those cases before TSOL or SOL Score can be emitted. Manual
workload-level FLOPs and fused-byte totals are rejected. The
artifact contains per-tensor byte traffic and MAC totals grouped by actual
operation precision; the evaluator records the artifact, graph, benchmark,
reference, architecture, solution, and environment identities in the bound
report chain. Loading a benchmark deterministically reruns the analyzer on the
bound graph and compares FLOPs, fused bytes, and the precision breakdown. It
also revalidates pinned single-einsum inputs/raw curves and, for supported
MatMul regions, rebuilds each fusion-friendly mapper sweep and recomposes the
joint curve from mapping-level raw OAVES records, including exact axis maps,
broadcast-batch flattening, and fanout schedules. It requires the
analysis architecture identity to equal the evaluator profile. A schema-v3
benchmark is rejected if any traffic-bearing tensor lacks an explicit dtype or
executable semantic record; neighboring quantization sidecars cannot override
the artifact.

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
correctness, score, capabilities, `bound_audit`, and structured failure states.

## RX 9060 XT resource audit

The bundled `gfx1200` profile revision `rx9060xt-amd-resource-v2` binds its
published limits to the locked-clock local audit at
`configs/arch/evidence/RX_9060_XT_resource_audit.yaml` (SHA-256
`11b8adc226b3b4f0d52830dde6fd112b1b376adffa81f1204514fa8d34114c1b`).
The evidence records ROCm/device identity, source hashes, clock state, raw
timings, stability, and measured-to-published ratios for every resource plus
HBM. Its `precision_support` matrix separates Radeon PyTorch production types,
rocWMMA/library support with a verified local framework path, and ISA-only
types. Required locked-clock probes cover FP32, FP16, BF16, both gfx1200 OCP
FP8 encodings, and INT8. INT4 is not probed: RDNA4 exposes IU4 WMMA, but ROCm
7.2 rocWMMA has no INT4 type and PyTorch 2.11 has no validated gfx1200 INT4
matrix API, so its published peak is explicitly source-only and exempt.
Measured calibration audits conservative published ceilings; it never replaces
a formal denominator. See [the precision support matrix](QUANT_SUPPORT.md).

Reproduce it on the same GPU with:

```bash
solar-calibrate-rocm \
  --arch-config RX_9060_XT \
  --timing-profile official \
  --output /tmp/RX_9060_XT_resource_audit.yaml
```

Evidence from other hardware must use its own architecture profile and audit.
No cross-hardware extrapolation is claimed by the bundled local audit.

## Pinned official representative corpus

`configs/corpus/RX_9060_XT_SOL_EXECBENCH.yaml` pins fifteen entries from
`nvidia/SOL-ExecBench` revision
`63699402f003496acc3af4eb534a5304a8ac1ea9`, including attention, norm, MoE,
SSM, convolution, MatMul, FP32/BF16/FP16/OCP FP8, forward/backward, dynamic
shapes, and structured inputs. Fourteen compatible entries have replayable
formal artifacts and independent FLOP, byte, and resource-counter goldens. The official OCP
E4M3 block-scale workload is formally scored as native gfx1200 FP8 with FP32
accumulation. NVFP4 is retained as an explicit
`unsupported_quantization_format` result; it is not converted, shrunk, sent to
CPU, or otherwise replaced.

Rebuild all selected source-to-SOL artifacts from that exact revision with:

```bash
solar-build-sol-execbench-corpus \
  configs/corpus/RX_9060_XT_SOL_EXECBENCH.yaml \
  --dataset-root /path/to/pinned-official-dataset \
  --output /tmp/RX_9060_XT_formal_artifacts \
  --orojenesis-home /path/to/pinned/Orojenesis
```

The builder groups multiple workloads from one problem, uses the manifest's
hash-bound architecture profile, and writes a `build-index.yaml`. Exit status 2
from the underlying per-problem builder is accepted only as terminal
incompatibility evidence; failed or unchecked work stops the batch.

Given an independently obtained copy of that exact official revision, audit it
with:

```bash
solar-audit-sol-execbench-corpus \
  configs/corpus/RX_9060_XT_SOL_EXECBENCH.yaml \
  --dataset-root /path/to/pinned-official-dataset \
  --artifact-root /path/to/formal-artifacts \
  --output /tmp/RX_9060_XT_SOL_EXECBENCH_audit.yaml
```

The checked local result is
`configs/corpus/evidence/RX_9060_XT_SOL_EXECBENCH_audit.yaml`. Every
compatibility record must contain `fallbacks_used: []`; incompatibility and OOM
are outcomes to record, not reasons to mutate the workload. The checked gate is
`passed: true`: all fifteen entries have terminal evidence, all fourteen
compatible entries are formally attested, and the per-axis minimums, critical
cross-axis combinations, L2/last-level-cache footprint classes, and fixed
M=1/M=8828 shape pair all pass. The report also binds the raw profile SHA-256
and canonical architecture hash.

`configs/corpus/RX_9060_XT_CONFORMANCE.yaml` is a separate repository-local
suite for accept/reject contracts such as layout bridges, batch flattening,
fanout, alias rejection, artifact replay, and toolchain tampering. These cases
run in pytest and never contribute to the official corpus denominator.

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
