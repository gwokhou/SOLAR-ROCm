# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest
import yaml

from solar.perf import EinsumGraphPerfModel
from solar.rocm import ArchitectureProfile
from solar.rocm.environment import Capability, RocmEnvironment


def test_packaged_profile_matches_repository_config():
    repository_config = (
        Path(__file__).parents[1] / "configs" / "arch" / "RX_9060_XT.yaml"
    )
    packaged_config = resources.files("solar.configs.arch").joinpath("RX_9060_XT.yaml")
    assert yaml.safe_load(
        repository_config.read_text(encoding="utf-8")
    ) == yaml.safe_load(packaged_config.read_text(encoding="utf-8"))


def test_only_amd_rocm_profiles_are_distributed():
    repository_dir = Path(__file__).parents[1] / "configs" / "arch"
    packaged_dir = resources.files("solar.configs.arch")
    expected = {"RX_9060_XT.yaml"}

    assert {path.name for path in repository_dir.glob("*.yaml")} == expected
    assert {
        path.name for path in packaged_dir.iterdir() if path.name.endswith(".yaml")
    } == expected


def test_rx_9060_xt_profile_and_roofline():
    profile = ArchitectureProfile.load("RX_9060_XT")
    assert profile.gfx_target == "gfx1200"
    assert profile.compute_units == 32
    assert profile.l2_bytes == 4 * 1024 * 1024
    assert profile.last_level_cache_bytes == 32 * 1024 * 1024
    assert profile.cache_flush_bytes == 32 * 1024 * 1024
    assert profile.peak_for("fp16") == 103e12
    assert profile.peak_for("float8_e4m3fnuz") == profile.peak_for("fp8")
    with pytest.raises(ValueError, match="not supported"):
        profile.peak_for("float8_e4m3fn")
    assert profile.theoretical_seconds(103e12, 0, "fp16") == pytest.approx(1.0)
    assert profile.theoretical_seconds(0, 320e9, "fp16") == pytest.approx(1.0)


def test_gfx1200_rejects_nvfp4():
    with pytest.raises(ValueError, match="not supported"):
        ArchitectureProfile.load("RX_9060_XT").peak_for("nvfp4")


def test_non_amd_architecture_profile_is_rejected():
    with pytest.raises(ValueError, match="AMD architecture profiles only"):
        ArchitectureProfile.load(
            {
                "name": "non-amd",
                "vendor": "other",
                "memory_bandwidth_bytes_per_second": 1,
                "peak_ops_per_second": {"fp16": 1},
            }
        )


def test_legacy_per_cycle_architecture_schema_is_rejected():
    with pytest.raises(ValueError, match="normalized"):
        ArchitectureProfile.load(
            {
                "name": "legacy",
                "freq_GHz": 2,
                "DRAM_byte_per_cycle": 100,
                "MAC_per_cycle_fp16_tc": 250,
            }
        )


def test_perf_model_defaults_to_rocm_profile(tmp_path: Path):
    analysis = {
        "metadata": {"precision": "fp16", "bytes_per_element": 2},
        "total": {
            "macs": 1_000_000,
            "flops": 2_000_000,
            "unfused_elements": 1_000_000,
            "fused_elements": 100_000,
            "fused_prefetched_elements": 100_000,
        },
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(analysis))
    result = EinsumGraphPerfModel().predict(path, tmp_path / "out", copy_analysis=False)
    assert result is not None
    assert result["arch"]["name"] == "Radeon_RX_9060_XT"
    assert result["model"]["rocm_native"] is True
    assert result["arch"]["throughput_precision"] == "fp16"
    assert result["arch"]["peak_ops_per_second"]["fp16"] == 103e12


def test_perf_model_normalizes_precision_aliases(tmp_path: Path):
    analysis = {
        "metadata": {"precision": "float16", "bytes_per_element": 2},
        "total": {"macs": 1, "flops": 2, "fused_elements": 1},
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(analysis))
    result = EinsumGraphPerfModel().predict(
        path, tmp_path / "out", precision="float16", copy_analysis=False
    )
    assert result is not None
    assert result["arch"]["throughput_precision"] == "fp16"
    assert result["workload"]["bytes_per_element"] == 2


def test_perf_model_normalizes_generic_float8_alias(tmp_path: Path):
    analysis = {
        "metadata": {"precision": "float8", "bytes_per_element": 1},
        "total": {"macs": 1024, "flops": 2048, "fused_elements": 512},
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(analysis))

    result = EinsumGraphPerfModel().predict(
        path, tmp_path / "out", precision="float8", copy_analysis=False
    )

    assert result is not None
    assert result["arch"]["throughput_precision"] == "fp8"
    assert result["workload"]["bytes_per_element"] == 1


def test_environment_accepts_discovered_gfx_target_and_serializes_capabilities():
    environment = RocmEnvironment(
        rocm_version="7.2",
        torch_version="2.11+rocm7.2",
        hip_version="7.2",
        device_name="RX 9060 XT",
        gfx_target="gfx1200",
        pytorch_compute_units=16,
        normalized_compute_units=32,
        total_memory_bytes=16,
        capabilities={"pytorch_rocm": Capability(True, "ok")},
    )
    assert environment.supported_target
    assert environment.to_dict()["capabilities"]["pytorch_rocm"]["available"] is True
