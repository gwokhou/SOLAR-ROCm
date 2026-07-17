# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

import sys

from torch.utils.cpp_extension import load

if len(sys.argv) != 3:
    raise SystemExit("usage: build_hip_extension.py GFX_TARGET BUILD_DIRECTORY")

load(
    name="solar_hip_matmul",
    sources=["hip_extension.cu"],
    extra_cuda_cflags=["-O3", f"--offload-arch={sys.argv[1]}"],
    build_directory=sys.argv[2],
    verbose=True,
)
