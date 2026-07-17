# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm kernel evaluation, calibration, and SOL scoring."""

from solar.benchmark.calibration import CalibrationArtifact, RocmCalibrator
from solar.benchmark.evaluator import EvaluationReport, RocmEvaluator
from solar.benchmark.models import BaselineRegistry, BenchmarkSpec, SolutionSpec

__all__ = [
    "BaselineRegistry",
    "BenchmarkSpec",
    "CalibrationArtifact",
    "EvaluationReport",
    "RocmCalibrator",
    "RocmEvaluator",
    "SolutionSpec",
]
