<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Solar: PyTorch Model Analysis Toolkit

Solar is a toolkit for analyzing PyTorch model graphs, converting them to einsum representations, and performing hardware-aware SOL performance predictions.

This ROCm port keeps the original five-stage analysis pipeline and adds an executable benchmarking path for AMD GPUs. The first validated target is the Radeon RX 9060 XT (`gfx1200`) on ROCm 7.2.

## Features

- **5-Stage Analysis Pipeline**: Seamless conversion from PyTorch models to performance predictions
- **Graph Extraction**: Extract structured computation graphs from PyTorch models (torchview-based)
- **Einsum Conversion**: Convert PyTorch operations to einsum notation with automatic rank renaming
- **Graph Visualization**: Generate PDF visualizations of einsum graphs
- **Hardware-Independent Analysis**: Compute MACs, FLOPs, and memory footprints
- **Performance Prediction**: AMD ROCm architecture-aware roofline modeling (RX 9060 XT)
- **Executable ROCm Evaluation**: Auditable PyTorch, Triton, HIP/C++, and AMD library kernel timing
- **Timeloop / Orojenesis Export**: Convert to Timeloop workload format for architectural exploration
- **Benchmark Support**: Native support for kernelbench benchmark suites
- **Human-Readable YAML**: All outputs use clean YAML without anchors/aliases

## Installation

The ROCm build supports Linux x86_64 and uses a pinned environment:

```bash
bash install_uv.sh
uv run solar-rocm-doctor
```

Install the system Graphviz package separately if PDF rendering is needed.

## The 5-Stage Pipeline

Solar processes models through five distinct stages:

```
Stage 1: PyTorch Graph Extraction
  └─> pytorch_graph.yaml

Stage 2: Einsum Conversion + Rank Renaming
  └─> einsum_graph.yaml
  └─> einsum_graph_renamed.yaml
  └─> einsum_graph.pdf (optional)

Stage 3: Hardware-Independent Analysis
  └─> analysis.yaml

Stage 4: Performance Prediction
  └─> perf_<arch>.yaml

Stage 5: Timeloop Export (optional)
  └─> timeloop_graph.yaml
```

## Examples

Solar includes several example models demonstrating different attention patterns:

### Available Examples

| Example | Description | Based On |
|---------|-------------|----------|
| `Attention/` | Multi-head self-attention | Standard Transformer |
| `BERT/` | BERT-like encoder model | BERT architecture |

### Running an Example

```bash
# Run the complete pipeline for any example
cd examples/Attention
bash run_solar.sh

# Outputs:
#   - output/graph/pytorch_graph.yaml           (Stage 1)
#   - output/einsum/einsum_graph.yaml           (Stage 2)
#   - output/einsum/einsum_graph_renamed.yaml   (Stage 2 - with BFS rank renaming)
#   - output/einsum/einsum_graph.pdf            (Stage 2 - visualization)
#   - output/analysis/analysis.yaml             (Stage 3)
#   - output/perf/perf_Radeon_RX_9060_XT.yaml           (Stage 4)
#   - output/timeloop/timeloop_graph.yaml       (Stage 5)
```

### Benchmark Suite (Kernelbench)

Process benchmark models:

```bash
# Process and analyze kernelbench models
solar-toeinsum --level level1 --kernel-ids 1 2 3

# Use different architecture
solar-toeinsum --level level1 --kernel-ids 1 --arch-config RX_9060_XT
```

## CLI Commands

### Single Model Processing

```bash
# Stage 1: Extract PyTorch graph
solar-process-model --model-file model.py --output-dir output/graph

# Stage 2: Convert to einsum (with optional PDF visualization)
solar-toeinsum-model --graph-path output/graph/pytorch_graph.yaml \
                     --output-dir output/einsum --no-copy-graph \
                     --save-graph

# Stage 3: Analyze (hardware-independent)
solar-analyze-model --einsum-graph-path output/einsum/einsum_graph_renamed.yaml \
                    --output-dir output/analysis

# Stage 4: Predict performance
solar-predict-perf-model --analysis-path output/analysis/analysis.yaml \
                         --output-dir output/perf --arch-config RX_9060_XT

# Stage 5: Convert to Timeloop format
solar-totimeloop --einsum-graph-path output/einsum/einsum_graph_renamed.yaml \
                 --output-dir output/timeloop
```

## Executable ROCm Evaluation

ROCm evaluation keeps benchmark, solution, and versioned baseline documents
separate. See [the ROCm benchmark guide](docs/ROCM_BENCHMARKING.md) for the
schema, timing profiles, clock policy, and publishability rules.

```bash
solar-evaluate \
  --benchmark examples/rocm_matmul/benchmark.yaml \
  --solution examples/rocm_matmul/solution.yaml \
  --timing-profile quick --no-lock-clocks \
  --output evaluation.yaml
```

The example package includes three verified solution manifests:

- `solution.yaml`: PyTorch/ROCm
- `triton_solution.yaml`: Triton/ROCm JIT kernel
- `hip_solution.yaml`: HIP C++ extension compiled for the detected gfx target

Replace the `--solution` value to exercise each backend. Native compilation
accepts `{python}`, `{staging}`, and `{gfx_target}` placeholders in the solution
manifest.

## Output File Formats

All output files use **human-readable YAML** without anchors/aliases:

- **pytorch_graph.yaml**: Structured graph with layers, shapes, weights, connections
- **einsum_graph.yaml**: Einsum equations + shapes for each layer
- **einsum_graph_renamed.yaml**: Einsum graph with consistent dimension labels (BFS-based)
- **einsum_graph.pdf**: Visual representation of the computation graph
- **analysis.yaml**: Hardware-independent metrics (MACs, FLOPs, bytes)
- **perf_<arch>.yaml**: Architecture-specific performance predictions
- **timeloop_graph.yaml**: Timeloop workload format for architectural exploration

## Testing

```bash
# Run all tests
bash scripts/run_tests.sh

# Quick smoke tests
bash scripts/run_tests.sh quick

# Run specific test categories
bash scripts/run_tests.sh graph      # Graph processing tests
bash scripts/run_tests.sh einsum     # Einsum analyzer tests
bash scripts/run_tests.sh unit       # All unit tests
bash scripts/run_tests.sh integration # Integration tests

# Test examples
bash scripts/run_tests.sh examples   # Run all example scripts

# Test benchmark compatibility
bash scripts/run_tests.sh kernelbench

# Verbose output
bash scripts/run_tests.sh all -v
```

See [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md) for detailed testing documentation.

## Python API

### Graph Processing

```python
from solar.graph import PyTorchProcessor

processor = PyTorchProcessor()
processor.process_model_file("model.py", output_dir="outputs/my_model")
```

### Einsum Conversion

```python
from solar.einsum import PyTorchToEinsum

converter = PyTorchToEinsum()
einsum_graph = converter.convert(
    "outputs/my_model/pytorch_graph.yaml",
    "outputs/my_model",
    copy_graph=False  # Don't duplicate input graph
)
# Produces both einsum_graph.yaml and einsum_graph_renamed.yaml
```

### Graph Visualization

```python
from solar.einsum import EinsumGraphVisualizer

visualizer = EinsumGraphVisualizer()
visualizer.save_graph_pdf(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model/einsum_graph.pdf"
)
```

### Analysis

```python
from solar.analysis import EinsumGraphAnalyzer

analyzer = EinsumGraphAnalyzer()
analysis = analyzer.analyze_graph(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model"
)
```

### Performance Prediction

```python
from solar.perf import EinsumGraphPerfModel

perf_model = EinsumGraphPerfModel()
perf = perf_model.predict(
    "outputs/my_model/analysis.yaml",
    "outputs/my_model",
    arch_config="RX_9060_XT"
)
```

### Timeloop Export

```python
from solar.einsum import EinsumToTimeloop

converter = EinsumToTimeloop()
result = converter.convert(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model/timeloop_graph.yaml"
)
```

## Architecture

```
SOLAR-ROCm/
├── solar/
│   ├── common/        # Shared types, constants, utilities (NoAliasDumper)
│   ├── graph/         # Stage 1: PyTorch graph extraction
│   ├── einsum/        # Stage 2: Einsum conversion, visualization, Timeloop export
│   ├── analysis/      # Stage 3: Hardware-independent analysis
│   ├── perf/          # Stage 4: Performance prediction
│   └── cli/           # Command-line interfaces
├── tests/             # Comprehensive test suite
├── examples/          # Maintained pipeline and ROCm kernel examples
└── configs/           # Normalized AMD ROCm architecture configs
```

## Key Components

- **NoAliasDumper**: Custom YAML dumper for human-readable output (no `&id001` references)
- **EinsumRankRenamer**: BFS-based dimension label renaming for consistent einsum equations
- **EinsumGraphVisualizer**: PDF visualization of computation graphs
- **EinsumToTimeloop**: Export to Timeloop workload format
- **Node Registry**: Extensible registry for operation handlers
- **LLM Agent** (optional): Dynamic handler generation for unknown operations
- **Benchmark Processors**: Specialized handling for kernelbench file structures

## Active Contributors

- [hqjennynv](https://github.com/hqjennynv)
- [sdamani-nvidia](https://github.com/sdamani-nvidia)
- [askiad](https://github.com/askiad)
- [LemonAndRabbit](https://github.com/LemonAndRabbit)

## Contributing

- Follow Google's Python Style Guide
- Add tests for new features
- Update documentation for API changes
- Run `bash scripts/run_tests.sh` before submitting PRs

## Documentation

- [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md): Comprehensive testing documentation
- [`docs/ROCM_BENCHMARKING.md`](docs/ROCM_BENCHMARKING.md): Executable ROCm benchmarking

## License

Apache License 2.0
