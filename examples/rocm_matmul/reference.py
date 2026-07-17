# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

import torch


def get_inputs(workload, device):
    n = int(workload["n"])
    generator = torch.Generator(device=device).manual_seed(int(workload["seed"]))
    return (
        torch.randn((n, n), dtype=torch.float16, device=device, generator=generator),
        torch.randn((n, n), dtype=torch.float16, device=device, generator=generator),
    )


def run(a, b):
    return torch.mm(a, b)
