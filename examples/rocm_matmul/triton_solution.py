# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for start_k in range(0, n, block_k):
        a_offsets = offsets_m[:, None] * n + start_k + offsets_k[None, :]
        b_offsets = (start_k + offsets_k[:, None]) * n + offsets_n[None, :]
        a = tl.load(
            a_ptr + a_offsets,
            mask=(offsets_m[:, None] < n) & (start_k + offsets_k[None, :] < n),
            other=0.0,
        )
        b = tl.load(
            b_ptr + b_offsets,
            mask=(start_k + offsets_k[:, None] < n) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(a, b)

    output_offsets = offsets_m[:, None] * n + offsets_n[None, :]
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=(offsets_m[:, None] < n) & (offsets_n[None, :] < n),
    )


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("the example expects equally sized square matrices")
    n = a.shape[0]
    output = torch.empty((n, n), dtype=a.dtype, device=a.device)
    block_m = 32
    block_n = 32
    grid = (triton.cdiv(n, block_m), triton.cdiv(n, block_n))
    _matmul_kernel[grid](
        a,
        b,
        output,
        n=n,
        block_m=block_m,
        block_n=block_n,
        block_k=32,
    )
    return output
