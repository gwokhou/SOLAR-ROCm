# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Solar: PyTorch Model Analysis Toolkit.

This package provides tools for analyzing PyTorch model graphs,
generating einsum representations, and performing performance analysis.

Package structure:
- solar.einsum: Einsum conversion (ops, converters, analyzer)
- solar.perf: Performance modeling
- solar.analysis: Hardware-independent analysis (graph analyzer, model analyzer)
- solar.graph: PyTorch graph processing
"""

__version__ = "2.0.0"

# Keep the historical top-level API without importing torchview when users only
# need ROCm diagnostics, YAML validation, or performance modeling.
_LAZY_IMPORTS = {
    "EinsumAnalyzer": ("solar.einsum", "EinsumAnalyzer"),
    "PyTorchToEinsum": ("solar.einsum", "PyTorchToEinsum"),
    "PyTorchEinsumConverter": ("solar.einsum", "PyTorchEinsumConverter"),
    "BenchmarkEinsumConverter": ("solar.einsum", "BenchmarkEinsumConverter"),
    "EinsumGraphAnalyzer": ("solar.analysis", "EinsumGraphAnalyzer"),
    "ModelAnalyzer": ("solar.analysis", "ModelAnalyzer"),
    "EinsumGraphPerfModel": ("solar.perf", "EinsumGraphPerfModel"),
    "PyTorchProcessor": ("solar.graph", "PyTorchProcessor"),
    "TorchviewProcessor": ("solar.graph", "TorchviewProcessor"),
    "EinsumOp": ("solar.einsum.ops", "EinsumOp"),
    "EinsumOperand": ("solar.einsum.ops", "EinsumOperand"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    # Einsum
    "EinsumAnalyzer",
    "PyTorchToEinsum",
    "PyTorchEinsumConverter",  # Backward compatibility
    "BenchmarkEinsumConverter",
    # Analysis
    "EinsumGraphAnalyzer",
    "ModelAnalyzer",
    # Performance
    "EinsumGraphPerfModel",
    # Graph processing
    "PyTorchProcessor",
    "TorchviewProcessor",
    # Types
    "EinsumOp",
    "EinsumOperand",
]
