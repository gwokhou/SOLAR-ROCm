<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Quantized kernel support

Solar separates shape-level graph analysis from architecture-specific runtime
prediction. Graph extraction may retain quantization conversion metadata even
when PyTorch's `meta` device cannot execute the original quantized operation.
Performance prediction then resolves the recorded format against the selected
architecture profile; it never silently substitutes an unrelated format.

## AMD ROCm behavior

The bundled profile is intentionally specific to RX 9060 XT (`gfx1200`) and
the tested ROCm 7.2 / PyTorch 2.11 stack. “Hardware native”, “available in a
ROCm library”, and “listed as production-supported by the Radeon PyTorch
matrix” are separate claims.

### RX 9060 XT precision support matrix

| Precision / encoding | gfx1200 hardware | ROCm 7.2 software status | Local PyTorch path | Profile and calibration policy |
| --- | --- | --- | --- | --- |
| FP32 | Native | Radeon PyTorch production matrix | `torch.mm`, verified | Published and locked-clock calibrated |
| FP16 | Native WMMA | Radeon PyTorch production matrix; rocWMMA | `torch.mm`, verified | Published and locked-clock calibrated |
| BF16 | Native WMMA | rocWMMA supports gfx12; omitted from the Radeon PyTorch production datatype list | `torch.mm`, verified on the pinned stack | Published and calibrated, but recorded as library-level / empirical framework support |
| OCP FP8 E4M3/E5M2 | Native WMMA input, FP32 accumulation/output | HIP and rocWMMA support gfx12; omitted from the Radeon PyTorch production datatype list | private `torch._scaled_mm`, both encodings verified | Published and calibrated; treated as partial/experimental framework support |
| INT8 | Native WMMA | Radeon PyTorch production datatype; rocWMMA supports gfx12 | private `torch._int_mm`, verified | Published and locked-clock calibrated |
| INT4 | Native IU4 WMMA instruction | No INT4 type in the ROCm 7.2 rocWMMA precision table and no validated PyTorch matrix API | None | Published ISA/product peak only; explicitly exempt from measured calibration |
| FP8 FNUZ E4M3/E5M2 | Not a gfx1200 encoding | HIP limits FNUZ to gfx94x | `_scaled_mm` returns `HIPBLAS_STATUS_NOT_SUPPORTED` locally | Rejected, never aliased to OCP FP8 |
| NVFP4 / generic FP4 | Not an RX 9060 XT format in this profile | No compatible path used by this project | None | Rejected, with no substitution |
| FP64 / TF32 matrix | No gfx12 combination in the ROCm 7.2 rocWMMA table | Outside the pinned corpus/profile scope | Not probed | Not published by this profile; support is deferred rather than inferred |

The authoritative software distinctions come from the
[ROCm 7.2 Radeon Linux support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html),
[rocWMMA 7.2.1 precision table](https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.2.1/api-reference/api-reference-guide.html),
[HIP FP8 device table](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.5/reference/fp8_numbers.html),
and the [RDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).
The local probe column is narrower: it says only that the exact operation ran
on the audited host, not that every PyTorch operator supports that dtype.

The profile aliases PyTorch's gfx1200 OCP spellings `float8_e4m3fn` and
`float8_e5m2` to the `fp8` roofline. The FNUZ spellings
`float8_e4m3fnuz` and `float8_e5m2fnuz` are rejected. This prevents an
incompatible encoding from being scored merely because it also occupies eight
bits.

INT4 is an explicit theoretical integer precision, not a synonym for FP4. Its
calibration exemption records the missing ROCm/PyTorch matrix path; a generic
`fp4` label remains unsupported.

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
Publishing a theoretical BF16/FP8/INT8/INT4 roofline does not by itself
guarantee that an arbitrary PyTorch operator implements that dtype on every
ROCm target. The profile's `precision_support` field and the locked-clock audit
preserve that distinction in machine-readable form.

The pinned official representative corpus deliberately retains one NVIDIA
OCP E4M3 FP8 problem as a native, formally attested case and one NVFP4 problem
as incompatible RX 9060 XT evidence. Its audit records
`unsupported_quantization_format` and `fallbacks_used: []` for NVFP4.
Substituting INT4, FP16/BF16, smaller shapes, or CPU execution would audit a
different workload and is therefore forbidden.
