# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Run diagnostic compute and memory calibration on the selected AMD target."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from solar.benchmark import RocmCalibrator
from solar.rocm import ArchitectureProfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure diagnostic ROCm throughput")
    parser.add_argument("--output", default="calibration.yaml")
    parser.add_argument(
        "--arch-config",
        default="RX_9060_XT",
        help="AMD architecture profile name or YAML path",
    )
    parser.add_argument(
        "--timing-profile",
        choices=("standard", "official", "quick"),
        default="standard",
    )
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args()
    import torch

    n = args.matrix_size
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    out = torch.empty_like(a)
    operations = {
        "fp16_flops": (lambda: torch.mm(a, b), float(2 * n**3)),
        "memory_bytes": (lambda: out.copy_(a), float(2 * a.numel() * a.element_size())),
    }
    artifact = RocmCalibrator(
        architecture=ArchitectureProfile.load(args.arch_config)
    ).calibrate(operations, timing_profile=args.timing_profile)
    Path(args.output).write_text(yaml.safe_dump(artifact.to_dict(), sort_keys=False))
    print(f"Diagnostic calibration written to {args.output}")


if __name__ == "__main__":
    main()
