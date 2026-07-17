# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantization-aware performance prediction.

Verifies that:
1. When metadata.yaml with nvfp4 orig_dtypes exists, the perf model uses
   the matching matrix-throughput entry and 0.5 bytes_per_element.
2. Without metadata.yaml, the perf model uses the default precision.
3. The two produce different results.
4. The analysis metadata also reflects the quant override.
"""

import pytest
import yaml
from pathlib import Path
from textwrap import dedent

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.perf import EinsumGraphPerfModel


MATMUL_MODEL_SOURCE = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 128))

    def forward(self, x):
        return torch.matmul(x, self.weight)

def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(4, 32, 64)]

def get_init_inputs():
    return []
"""

NVFP4_METADATA = {
    "dtype_conversions": [
        {
            "function": "quantize",
            "operation": "dtype_cast",
            "orig_dtypes": "nvfp4 float4_e2m1fn_x2",
            "new_dtypes": "int8",
            "reason": "nvfp4 not supported on meta device",
        },
    ],
}

FP8_METADATA = {
    "dtype_conversions": [
        {
            "function": "forward",
            "operation": "source_dtype_replacement",
            "orig_dtypes": "fp8 float8_e4m3fn",
            "new_dtypes": "int8",
            "count": 2,
            "reason": "not supported on meta/cpu device",
        },
    ],
}


def _run_pipeline(tmp_path: Path) -> Path:
    """Run graph extraction + einsum conversion. Returns einsum dir."""
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(MATMUL_MODEL_SOURCE))

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()

    config = ProcessingConfig(
        save_graph=False, force_rerun=True, debug=False, safe_mode=False,
    )
    processor = PyTorchProcessor(config)
    ok = processor.process_model_file(str(model_file), str(graph_dir))
    assert ok, "Graph extraction failed"

    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()
    converter = PyTorchToEinsum()
    result = converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert result is not None, "Einsum conversion failed"
    assert (einsum_dir / "einsum_graph_renamed.yaml").exists()

    return einsum_dir


class TestPerfQuantNVFP4:
    """Test that nvfp4 metadata changes both analysis and perf results."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.einsum_dir = _run_pipeline(tmp_path)
        self.renamed_path = self.einsum_dir / "einsum_graph_renamed.yaml"

    def _run_analysis(self, metadata=None):
        """Run analysis, optionally writing metadata.yaml near einsum graph."""
        analysis_dir = self.tmp_path / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        if metadata is not None:
            meta_path = self.tmp_path / "metadata.yaml"
            with open(meta_path, "w") as f:
                yaml.dump(metadata, f)

        analyzer = EinsumGraphAnalyzer()
        analysis = analyzer.analyze_graph(
            str(self.renamed_path), str(analysis_dir),
            precision="fp16", copy_graph=False,
        )
        assert analysis is not None
        return analysis

    def _run_perf(self, analysis_path, metadata=None):
        """Run perf prediction on the gfx1200 profile."""
        perf_dir = self.tmp_path / "perf"
        perf_dir.mkdir(exist_ok=True)

        if metadata is not None:
            meta_path = analysis_path.parent.parent / "metadata.yaml"
            with open(meta_path, "w") as f:
                yaml.dump(metadata, f)

        model = EinsumGraphPerfModel()
        perf = model.predict(
            str(analysis_path), str(perf_dir),
            arch_config="RX_9060_XT", precision="fp16",
        )
        assert perf is not None
        return perf

    def test_analysis_metadata_without_quant(self):
        """Without metadata.yaml, analysis uses fp16 / 2 bytes."""
        analysis = self._run_analysis(metadata=None)
        meta = analysis["metadata"]
        assert meta["precision"] == "fp16"
        assert meta["bytes_per_element"] == 2

    def test_analysis_metadata_with_nvfp4(self):
        """With nvfp4 metadata.yaml, analysis uses nvfp4 / 0.5 bytes."""
        analysis = self._run_analysis(metadata=NVFP4_METADATA)
        meta = analysis["metadata"]
        assert meta["precision"] == "nvfp4"
        assert meta["bytes_per_element"] == 0.5

    def test_analysis_metadata_with_fp8(self):
        """With fp8 metadata.yaml, analysis uses fp8 / 1 byte."""
        analysis = self._run_analysis(metadata=FP8_METADATA)
        meta = analysis["metadata"]
        assert meta["precision"] == "fp8"
        assert meta["bytes_per_element"] == 1

    def test_perf_nvfp4_vs_fp16_different(self):
        """FP16 remains usable while NVFP4 metadata is rejected on gfx1200."""
        analysis_no_quant = self._run_analysis(metadata=None)
        analysis_path = self.tmp_path / "analysis" / "analysis.yaml"
        perf_no_quant = self._run_perf(analysis_path, metadata=None)
        assert perf_no_quant["workload"]["bytes_per_element"] == 2

        self._run_analysis(metadata=NVFP4_METADATA)
        with pytest.raises(ValueError, match="NVFP4"):
            self._run_perf(analysis_path, metadata=NVFP4_METADATA)

    def test_perf_nvfp4_has_quant_label(self):
        """NVFP4 metadata must never silently fall back to another precision."""
        self._run_analysis(metadata=NVFP4_METADATA)
        analysis_path = self.tmp_path / "analysis" / "analysis.yaml"
        with pytest.raises(ValueError, match="unsupported"):
            self._run_perf(analysis_path, metadata=NVFP4_METADATA)

    def test_perf_no_quant_no_label(self):
        """Perf output should NOT include quant_orig_dtype without metadata."""
        self._run_analysis(metadata=None)
        analysis_path = self.tmp_path / "analysis" / "analysis.yaml"
        perf = self._run_perf(analysis_path, metadata=None)

        assert "quant_orig_dtype" not in perf["workload"]


# Model with only elementwise ops (no MACs, only other_ops — like rmsnorm)
ELEMENTWISE_MODEL_SOURCE = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(128))
        self.eps = 1e-6

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(4, 32, 128)]

def get_init_inputs():
    return []
"""

# Model with both MACs (matmul) and other_ops (relu)
MATMUL_RELU_MODEL_SOURCE = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 128))

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        return F.relu(y)

def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(4, 32, 64)]

def get_init_inputs():
    return []
"""


def _run_pipeline_from_source(tmp_path: Path, source: str) -> Path:
    """Run graph extraction + einsum conversion for given source."""
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(source))

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()

    config = ProcessingConfig(
        save_graph=False, force_rerun=True, debug=False, safe_mode=False,
    )
    processor = PyTorchProcessor(config)
    ok = processor.process_model_file(str(model_file), str(graph_dir))
    assert ok, "Graph extraction failed"

    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()
    converter = PyTorchToEinsum()
    result = converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert result is not None, "Einsum conversion failed"
    assert (einsum_dir / "einsum_graph_renamed.yaml").exists()
    return einsum_dir


class TestPerfSMCycles:
    """Test that other_ops drive compute_scalar_cycles in perf model."""

    def _analyze_and_predict(self, tmp_path, source):
        einsum_dir = _run_pipeline_from_source(tmp_path, source)
        renamed_path = einsum_dir / "einsum_graph_renamed.yaml"

        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        analyzer = EinsumGraphAnalyzer()
        analysis = analyzer.analyze_graph(
            str(renamed_path), str(analysis_dir),
            precision="fp16", copy_graph=False,
        )
        assert analysis is not None

        perf_dir = tmp_path / "perf"
        perf_dir.mkdir(exist_ok=True)
        model = EinsumGraphPerfModel()
        perf = model.predict(
            str(analysis_dir / "analysis.yaml"), str(perf_dir),
            arch_config="RX_9060_XT", precision="fp16",
        )
        assert perf is not None
        return analysis, perf

    def test_elementwise_only_sm_cycles_nonzero(self, tmp_path):
        """Elementwise-only model: other_ops > 0, macs == 0.
        compute_scalar_cycles > 0 (informational), but compute_cycles = tc_cycles
        because elementwise ops are memory-bound and already captured by
        memory cycles in the roofline model."""
        analysis, perf = self._analyze_and_predict(tmp_path, ELEMENTWISE_MODEL_SOURCE)

        assert analysis["total"]["macs"] == 0
        assert analysis["total"]["other_ops"] > 0

        assert perf["unfused"]["compute_matrix_cycles"] == 0
        assert perf["unfused"]["compute_scalar_cycles"] > 0
        assert perf["unfused"]["compute_cycles"] == perf["unfused"]["compute_matrix_cycles"]

    def test_elementwise_only_total_other_ops_in_workload(self, tmp_path):
        """Perf output should include total_other_ops."""
        _, perf = self._analyze_and_predict(tmp_path, ELEMENTWISE_MODEL_SOURCE)
        assert perf["workload"]["total_other_ops"] > 0
        assert perf["workload"]["total_macs"] == 0

    def test_matmul_relu_both_cycles(self, tmp_path):
        """Matmul + relu: TC cycles > 0, SM cycles >= 0, other_ops tracked."""
        analysis, perf = self._analyze_and_predict(tmp_path, MATMUL_RELU_MODEL_SOURCE)

        assert analysis["total"]["macs"] > 0
        assert analysis["total"]["other_ops"] > 0

        assert perf["fused"]["compute_matrix_cycles"] > 0
        assert perf["workload"]["total_other_ops"] > 0
        assert perf["fused"]["compute_cycles"] == max(
            perf["fused"]["compute_matrix_cycles"],
            perf["fused"]["compute_scalar_cycles"],
        )

    def test_matmul_only_sm_cycles_zero(self, tmp_path):
        """Pure matmul model: other_ops == 0, compute_scalar_cycles == 0."""
        _, perf = self._analyze_and_predict(tmp_path, MATMUL_MODEL_SOURCE)

        assert perf["unfused"]["compute_matrix_cycles"] > 0
        assert perf["unfused"]["compute_scalar_cycles"] == 0
        assert perf["unfused"]["compute_cycles"] == perf["unfused"]["compute_matrix_cycles"]

    def test_sm_cycles_consistent_across_models(self, tmp_path):
        """compute_matrix_cycles and compute_scalar_cycles are the same in unfused/fused/prefetched."""
        _, perf = self._analyze_and_predict(tmp_path, MATMUL_RELU_MODEL_SOURCE)

        for model in ["unfused", "fused", "fused_prefetched"]:
            assert perf[model]["compute_matrix_cycles"] == perf["unfused"]["compute_matrix_cycles"]
            assert perf[model]["compute_scalar_cycles"] == perf["unfused"]["compute_scalar_cycles"]
            assert perf[model]["compute_cycles"] == perf["unfused"]["compute_cycles"]

    def test_arch_has_sm_throughput(self, tmp_path):
        """Perf output should include MAC_per_cycle_fp32_sm from arch."""
        _, perf = self._analyze_and_predict(tmp_path, ELEMENTWISE_MODEL_SOURCE)
        assert perf["arch"]["scalar_operations_per_cycle"] > 0


class TestPerfPrecisions:
    """Test that different FP precisions produce different perf results.

    RX 9060 XT has published FP32, FP16/BF16, and FP8 throughput.
    NVFP4 remains a readable metadata width but is not executable on gfx1200.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        einsum_dir = _run_pipeline_from_source(tmp_path, MATMUL_MODEL_SOURCE)
        self.renamed = einsum_dir / "einsum_graph_renamed.yaml"

        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        analyzer = EinsumGraphAnalyzer()
        analyzer.analyze_graph(
            str(self.renamed), str(analysis_dir),
            precision="fp32", copy_graph=False,
        )
        self.analysis_path = analysis_dir / "analysis.yaml"

    def _predict(self, precision):
        perf_dir = self.tmp_path / f"perf_{precision}"
        perf_dir.mkdir(exist_ok=True)
        model = EinsumGraphPerfModel()
        perf = model.predict(
            str(self.analysis_path), str(perf_dir),
            arch_config="RX_9060_XT", precision=precision,
        )
        assert perf is not None, f"Predict failed for precision={precision}"
        return perf

    def test_fp32_uses_sm_key(self):
        """fp32 should fall back to fp32_tc (or fp32_sm) key."""
        perf = self._predict("fp32")
        assert perf["arch"]["throughput_precision"] == "fp32"

    def test_fp16_uses_fp16_tc(self):
        perf = self._predict("fp16")
        assert perf["arch"]["throughput_precision"] == "fp16"

    def test_bf16_uses_bf16_tc(self):
        perf = self._predict("bf16")
        assert perf["arch"]["throughput_precision"] == "bf16"

    def test_fp16_and_bf16_same_throughput(self):
        """FP16 and BF16 have identical tensor core throughput on B200."""
        perf_fp16 = self._predict("fp16")
        perf_bf16 = self._predict("bf16")
        assert perf_fp16["arch"]["operations_per_cycle"] == perf_bf16["arch"]["operations_per_cycle"]

    def test_fp8_higher_throughput_than_fp16(self):
        perf_fp16 = self._predict("fp16")
        perf_fp8 = self._predict("fp8")
        assert perf_fp8["arch"]["operations_per_cycle"] > perf_fp16["arch"]["operations_per_cycle"]

    def test_fp8_fewer_bytes_than_fp16(self):
        perf_fp16 = self._predict("fp16")
        perf_fp8 = self._predict("fp8")
        assert perf_fp8["workload"]["bytes_per_element"] < perf_fp16["workload"]["bytes_per_element"]

    def test_nvfp4_highest_throughput(self):
        """NVFP4 is rejected instead of mapped to an unrelated AMD format."""
        with pytest.raises(ValueError, match="NVFP4"):
            self._predict("nvfp4")

    def test_nvfp4_half_byte(self):
        """Metadata width does not make NVFP4 executable on gfx1200."""
        with pytest.raises(ValueError, match="NVFP4"):
            self._predict("nvfp4")

    def test_fp32_four_bytes(self):
        perf = self._predict("fp32")
        assert perf["workload"]["bytes_per_element"] == 4

    def test_fp16_two_bytes(self):
        perf = self._predict("fp16")
        assert perf["workload"]["bytes_per_element"] == 2

    def test_fp8_one_byte(self):
        perf = self._predict("fp8")
        assert perf["workload"]["bytes_per_element"] == 1

    def test_higher_precision_slower_runtime(self):
        """fp32 should be slower than fp16 which should be slower than fp8."""
        perf_fp32 = self._predict("fp32")
        perf_fp16 = self._predict("fp16")
        perf_fp8 = self._predict("fp8")
        assert perf_fp32["fused"]["runtime_ms"] > perf_fp16["fused"]["runtime_ms"]
        assert perf_fp16["fused"]["runtime_ms"] > perf_fp8["fused"]["runtime_ms"]

    def test_memory_bytes_scale_with_precision(self):
        """Memory bytes should scale with bytes_per_element."""
        perf_fp32 = self._predict("fp32")
        perf_fp16 = self._predict("fp16")
        fp32_bytes = perf_fp32["fused"]["memory_bytes"]
        fp16_bytes = perf_fp16["fused"]["memory_bytes"]
        ratio = fp32_bytes / fp16_bytes
        assert 1.9 < ratio < 2.1, f"fp32/fp16 memory ratio should be ~2, got {ratio}"

    def test_all_precisions_produce_valid_output(self):
        """All supported precisions should produce valid perf dicts."""
        for prec in ["fp32", "fp16", "bf16", "fp8"]:
            perf = self._predict(prec)
            assert perf["workload"]["total_macs"] > 0
            assert perf["fused"]["total_cycles"] > 0
            assert perf["fused"]["runtime_ms"] > 0

