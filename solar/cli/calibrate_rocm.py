# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Run a locked-clock RX 9060 XT resource audit."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

from solar.benchmark import ResourceProbe, RocmCalibrator
from solar.benchmark.clock_lock import (
    ClockLockLease,
    acquire_clock_lock,
    query_clock_level,
)
from solar.rocm import ArchitectureProfile


def _build_probes(matrix_size: int) -> dict[str, ResourceProbe]:
    import torch

    n = matrix_size
    fp16_a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    fp16_b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    bf16_a = fp16_a.to(torch.bfloat16)
    bf16_b = fp16_b.to(torch.bfloat16)
    fp32_a = fp16_a.float()
    fp32_b = fp16_b.float()
    fp8_scale = torch.ones((), device="cuda", dtype=torch.float32)
    fp8_e4m3_a = fp16_a.to(torch.float8_e4m3fn)
    fp8_e4m3_b = fp16_b.to(torch.float8_e4m3fn).T
    fp8_e5m2_a = fp16_a.to(torch.float8_e5m2)
    fp8_e5m2_b = fp16_b.to(torch.float8_e5m2).T
    int8_a = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
    int8_b = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
    vector_n = max(n * n, 1 << 20)
    vector = torch.randn((vector_n,), device="cuda", dtype=torch.float32)
    vector_b = torch.randn_like(vector)
    integer = torch.randint(0, 1024, (vector_n,), device="cuda", dtype=torch.int32)
    atomic_out = torch.zeros((max(vector_n // 4, 1),), device="cuda")
    atomic_index = torch.randint(
        0, atomic_out.numel(), (vector_n,), device="cuda", dtype=torch.int64
    )
    atomic_src = torch.randn((vector_n,), device="cuda")
    memory_out = torch.empty_like(vector)

    return {
        "mfma_fp16_fp32": ResourceProbe(
            lambda: torch.mm(fp16_a, fp16_b), 2.0 * n**3, "mfma", "fp16->fp32"
        ),
        "mfma_bf16_fp32": ResourceProbe(
            lambda: torch.mm(bf16_a, bf16_b), 2.0 * n**3, "mfma", "bf16->fp32"
        ),
        "mfma_fp32_fp32": ResourceProbe(
            lambda: torch.mm(fp32_a, fp32_b), 2.0 * n**3, "mfma", "fp32->fp32"
        ),
        "mfma_fp8_e4m3_fp32": ResourceProbe(
            lambda: torch._scaled_mm(
                fp8_e4m3_a,
                fp8_e4m3_b,
                fp8_scale,
                fp8_scale,
                out_dtype=torch.float32,
            ),
            2.0 * n**3,
            "mfma",
            "fp8->fp32",
        ),
        "mfma_fp8_e5m2_fp32": ResourceProbe(
            lambda: torch._scaled_mm(
                fp8_e5m2_a,
                fp8_e5m2_b,
                fp8_scale,
                fp8_scale,
                out_dtype=torch.float32,
            ),
            2.0 * n**3,
            "mfma",
            "fp8->fp32",
        ),
        "mfma_int8_int32": ResourceProbe(
            lambda: torch._int_mm(int8_a, int8_b),
            2.0 * n**3,
            "mfma",
            "int8->int32",
        ),
        "valu_fp32": ResourceProbe(
            lambda: torch.add(vector, vector_b), float(vector_n), "valu", "fp32"
        ),
        "valu_integer": ResourceProbe(
            lambda: torch.add(integer, 1), float(vector_n), "valu", "integer"
        ),
        "sfu_fp32": ResourceProbe(
            lambda: torch.exp(vector), float(vector_n), "sfu", "fp32"
        ),
        "reduction_fp32": ResourceProbe(
            lambda: torch.sum(vector), float(vector_n - 1), "reduction", "fp32"
        ),
        "atomic_fp32": ResourceProbe(
            lambda: atomic_out.scatter_add_(0, atomic_index, atomic_src),
            float(vector_n),
            "atomic",
            "fp32",
        ),
        "scan_fp32": ResourceProbe(
            lambda: torch.cumsum(vector, dim=0),
            float(vector_n),
            "scan_sort",
            "fp32",
        ),
        "conversion_fp32_fp16": ResourceProbe(
            lambda: vector.to(torch.float16),
            float(vector_n),
            "conversion",
            "fp32->fp16",
        ),
        "conversion_fp32_fp8": ResourceProbe(
            lambda: vector.to(torch.float8_e4m3fn),
            float(vector_n),
            "conversion",
            "fp32->fp8",
        ),
        "conversion_fp8_fp32": ResourceProbe(
            lambda: vector.to(torch.float8_e4m3fn).float(),
            float(vector_n),
            "conversion",
            "fp8->fp32",
        ),
        "memory_hbm": ResourceProbe(
            lambda: memory_out.copy_(vector),
            float(2 * vector.numel() * vector.element_size()),
            "memory",
            "hbm",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure and audit every formal AMD compute resource"
    )
    parser.add_argument("--output", default="calibration.yaml")
    parser.add_argument(
        "--arch-config",
        default="RX_9060_XT",
        help="AMD architecture profile name or YAML path",
    )
    parser.add_argument(
        "--timing-profile",
        choices=("official", "standard", "quick"),
        default="official",
    )
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument(
        "--no-lock-clocks",
        action="store_true",
        help="produce diagnostic-only evidence; official mode will reject it",
    )
    args = parser.parse_args()
    if args.matrix_size <= 0:
        parser.error("--matrix-size must be positive")

    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lease = ClockLockLease(False, False)
    if not args.no_lock_clocks:
        lease = acquire_clock_lock()
    try:
        artifact = RocmCalibrator(
            architecture=ArchitectureProfile.load(args.arch_config)
        ).calibrate(
            _build_probes(args.matrix_size),
            timing_profile=args.timing_profile,
            clocks_locked=lease.locked,
            clock_levels=query_clock_level(),
            probe_source_sha256=source_sha,
        )
        Path(args.output).write_text(
            yaml.safe_dump(artifact.to_dict(), sort_keys=False), encoding="utf-8"
        )
        print(f"ROCm resource audit written to {args.output}")
    finally:
        lease.release()


if __name__ == "__main__":
    main()
