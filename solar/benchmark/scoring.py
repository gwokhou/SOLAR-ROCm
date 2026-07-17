# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOL Score validation and calculation."""

from __future__ import annotations


def calculate_sol_score(
    baseline_ms: float, candidate_ms: float, theoretical_ms: float
) -> float:
    """Calculate the paper's score without clipping invalid measurements."""
    if baseline_ms <= theoretical_ms:
        raise ValueError("baseline latency must be greater than theoretical SOL")
    if candidate_ms < theoretical_ms:
        raise ValueError("candidate latency below theoretical SOL requires an audit")
    denominator = (candidate_ms - theoretical_ms) + (baseline_ms - theoretical_ms)
    if denominator <= 0:
        raise ValueError("SOL Score denominator must be positive")
    return (baseline_ms - theoretical_ms) / denominator


__all__ = ["calculate_sol_score"]
