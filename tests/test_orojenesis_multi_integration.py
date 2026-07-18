# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Opt-in integration coverage for the pinned multi-einsum mapper binary."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import pytest

from solar.analysis.orojenesis import (
    MULTI_EINSUM_COMPOSITION,
    OROJENESIS_COMMIT,
    OROJENESIS_PROVENANCE_FILENAME,
    OrojenesisRunner,
    find_multi_einsum_regions,
    parse_multi_einsum_curve,
    parse_multi_einsum_region_curve,
    parse_multi_mapping_records,
    select_capacity_point,
)


def _matmul(
    left: str,
    right: str,
    output: str,
    *,
    m_size: int = 2,
    k_size: int = 2,
    n_size: int = 2,
) -> dict:
    return {
        "type": "matmul",
        "semantic_op": {
            "kind": "einsum",
            "target": "matmul",
            "equation": "MK,KN->MN",
            "arguments": [{"tensor": 0}, {"tensor": 1}],
            "kwargs": {},
            "effects": {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {"inputs": [left, right], "outputs": [output]},
        "tensor_shapes": {
            "inputs": [[m_size, k_size], [k_size, n_size]],
            "outputs": [[m_size, n_size]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float16", "torch.float16"],
            "outputs": ["torch.float16"],
        },
    }


_OROJENESIS_HOME = os.environ.get("SOLAR_OROJENESIS_HOME")
_REQUIRE_INTEGRATION = os.environ.get("SOLAR_REQUIRE_OROJENESIS_INTEGRATION") == "1"


@pytest.mark.orojenesis_real
def test_pinned_multi_einsum_mapper_emits_replayable_joint_curve(
    tmp_path: Path,
) -> None:
    if not _OROJENESIS_HOME:
        if _REQUIRE_INTEGRATION:
            pytest.fail("required real Orojenesis integration home is missing")
        pytest.skip("set SOLAR_OROJENESIS_HOME to run the pinned mapper test")
    runner = OrojenesisRunner(_OROJENESIS_HOME, timeout_seconds=600)
    identity = runner.toolchain_identity
    mapper = Path(_OROJENESIS_HOME) / "bin" / "timeloop-mapper"
    assert identity["source"]["commit"] == OROJENESIS_COMMIT
    assert (
        identity["artifact"]["sha256"]
        == hashlib.sha256(mapper.read_bytes()).hexdigest()
    )
    if identity["verification_mode"] == "provenance_manifest":
        assert (Path(_OROJENESIS_HOME) / OROJENESIS_PROVENANCE_FILENAME).is_file()

    single = runner.run_layer(
        _matmul("x", "w", "y"),
        tmp_path / "single",
        word_bits=16,
    )
    assert single["curve"]
    assert single["toolchain"] == identity

    result = runner.run_multi_chain(
        [
            ("first", _matmul("x", "w1", "hidden")),
            ("second", _matmul("hidden", "w2", "output")),
        ],
        tmp_path,
        word_bits=16,
    )
    assert result["commit"] == OROJENESIS_COMMIT
    assert result["toolchain"] == identity
    assert result["composition"] == MULTI_EINSUM_COMPOSITION
    assert result["environment"] == {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
    assert result["curve"]
    assert all(len(point["mappings"]) == 2 for point in result["curve"])
    assert min(point["dram_accesses_words"] for point in result["curve"]) >= 16

    evidence = result["evidence_files"]
    for name, item in evidence.items():
        path = tmp_path / item["path"]
        assert path.is_file(), name
        if name.endswith("-raw"):
            assert parse_multi_mapping_records(path, word_bytes=2)
    assert (
        parse_multi_einsum_curve(tmp_path / evidence["curve"]["path"], word_bytes=2)
        == result["curve"]
    )
    assert select_capacity_point(result["curve"], 1 << 20) is not None

    first = _matmul("x", "w1", "hidden")
    first["semantic_op"]["equation"] = "MK,NK->MN"
    first["tensor_shapes"]["inputs"][1] = [2, 2]
    extended_layers = {
        "x": {
            "type": "start",
            "tensor_names": {"inputs": [], "outputs": ["x"]},
        },
        "w1": {
            "type": "start",
            "tensor_names": {"inputs": [], "outputs": ["w1"]},
        },
        "w2": {
            "type": "start",
            "tensor_names": {"inputs": [], "outputs": ["w2"]},
        },
        "first": first,
        "second": copy.deepcopy(_matmul("hidden", "w2", "output")),
    }
    regions = find_multi_einsum_regions(extended_layers)
    assert len(regions) == 1
    extended = runner.run_multi_region(regions[0], tmp_path / "extended", word_bits=16)
    assert extended["curve"]
    assert extended["toolchain"] == identity
    extended_curve = extended["evidence_files"]["curve"]
    assert (
        parse_multi_einsum_region_curve(
            tmp_path / "extended" / extended_curve["path"], word_bytes=2
        )
        == extended["curve"]
    )
