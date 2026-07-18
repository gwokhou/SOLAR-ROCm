# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-independent graph, fusion, and Orojenesis analysis APIs."""

# Public attributes are populated by ``__getattr__`` to keep lightweight mapper
# contract tests from importing the full graph-analysis dependency tree.
# pylint: disable=undefined-all-variable

from __future__ import annotations

from importlib import import_module

_LAZY_IMPORTS = {
    "EinsumGraphAnalyzer": ("solar.analysis.graph_analyzer", "EinsumGraphAnalyzer"),
    "ModelAnalyzer": ("solar.analysis.model_analyzer", "ModelAnalyzer"),
    "FusionPlanner": ("solar.analysis.fusion", "FusionPlanner"),
    "OrojenesisError": ("solar.analysis.orojenesis", "OrojenesisError"),
    "OrojenesisRunner": ("solar.analysis.orojenesis", "OrojenesisRunner"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_LAZY_IMPORTS)
