<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Quantized kernel support

Solar separates shape-level graph analysis from architecture-specific runtime
prediction. Graph extraction may retain quantization conversion metadata even
when PyTorch's `meta` device cannot execute the original quantized operation.
Performance prediction then resolves the recorded format against the selected
architecture profile; it never silently substitutes an unrelated format.

## AMD ROCm behavior

The bundled RX 9060 XT profile publishes FP32, FP16, BF16, generic FP8, INT8,
and INT4 rooflines. It explicitly aliases the ROCm/PyTorch FNUZ spellings
`float8_e4m3fnuz` and `float8_e5m2fnuz` to that published FP8 roofline.

Non-ROCm `float8_e4m3fn`, `float8_e5m2`, and NVFP4 spellings are not aliases on
the AMD profile. Selecting them, or discovering them in `metadata.yaml`, raises
an unsupported-precision error. This prevents an incompatible format from
being scored as if it were an AMD format with the same storage width.

INT4 is supported as an explicit theoretical precision. It is not a synonym
for FP4. A generic `fp4` label is intentionally rejected unless a future
architecture profile publishes an explicit FP4 throughput.

## Metadata contract

When preprocessing outside this repository rewrites an unsupported dtype for
shape-only tracing, it may place the original dtype in a nearby
`metadata.yaml`:

```yaml
dtype_conversions:
  - operation: source_dtype_replacement
    orig_dtypes: float8_e4m3fnuz
    new_dtypes: int8
    reason: shape-only graph extraction
```

Solar searches up to three parent directories for this file. The analysis
stage records the quantization family and byte width; the performance stage
re-reads the exact original label and checks it against the selected profile's
`precision_aliases`.

## Current boundary

This repository does not contain the historical SolBench postprocessor or the
`run_solbenchv3*.sh` wrappers referenced by older upstream documentation.
External preprocessing is supported through the metadata contract above, but
those absent tools are not presented as maintained functionality.

Executable correctness and timing depend on the submitted solution backend.
Publishing a theoretical FP8/INT8/INT4 roofline does not by itself guarantee
that an arbitrary PyTorch operator implements that dtype on every ROCm target.
