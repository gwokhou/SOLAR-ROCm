# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Report ROCm evaluator readiness."""

from __future__ import annotations

import argparse
import json

from solar.rocm import ArchitectureProfile, RocmEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SOLAR ROCm readiness")
    parser.add_argument(
        "--arch-config",
        default="RX_9060_XT",
        help="AMD architecture profile name or YAML path",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args()
    environment = RocmEnvironment.detect()
    profile = ArchitectureProfile.load(args.arch_config)
    payload = {"environment": environment.to_dict(), "architecture": profile.to_dict()}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Device: {environment.device_name or 'unavailable'}")
        print(f"Target: {environment.gfx_target or 'unavailable'}")
        print(
            "Compute units: "
            f"{environment.normalized_compute_units or 'unknown'} "
            f"(PyTorch reports {environment.pytorch_compute_units or 'unknown'} WGPs/CUs)"
        )
        print(
            f"ROCm/HIP: {environment.rocm_version or environment.hip_version or 'unavailable'}"
        )
        for name, capability in sorted(environment.capabilities.items()):
            marker = "OK" if capability.available else "MISSING"
            print(f"[{marker}] {name}: {capability.detail}")
    if (
        not environment.supported_target
        or profile.vendor.upper() != "AMD"
        or environment.gfx_target != profile.gfx_target
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
