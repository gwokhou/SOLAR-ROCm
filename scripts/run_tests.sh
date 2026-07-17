#!/bin/bash
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

# Run tests for Solar package with support for kernelbench.
#
# Solar Pipeline Stages:
#   1. PyTorch graph extraction (pytorch_graph.yaml)
#   2. Einsum conversion + rank renaming (einsum_graph.yaml, einsum_graph_renamed.yaml, einsum_graph.pdf)
#   3. Hardware-independent analysis (analysis.yaml)
#   4. Performance prediction (perf_<arch>.yaml)
#   5. Timeloop export (timeloop_graph.yaml)

set -e

if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import torch; from solar._vendor import torchview" 2>/dev/null; then
    PYTHON=(".venv/bin/python")
else
    PYTHON=("python3")
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}       Solar Package Test Runner        ${NC}"
echo -e "${GREEN}========================================${NC}"

# Parse arguments
TEST_TYPE="${1:-all}"
VERBOSE="${2:-}"

# Get script directory and solar root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to solar root for running tests
cd "${SOLAR_ROOT}"

# Function to run tests
run_tests() {
    local test_module=$1
    local test_name=$2
    
    echo -e "\n${YELLOW}Running $test_name...${NC}"
    
    if [ "$VERBOSE" = "-v" ] || [ "$VERBOSE" = "--verbose" ]; then
        "${PYTHON[@]}" -m pytest tests/$test_module -v --tb=short
    else
        "${PYTHON[@]}" -m pytest tests/$test_module -q
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $test_name passed${NC}"
    else
        echo -e "${RED}❌ $test_name failed${NC}"
        exit 1
    fi
}

# Function to run example scripts
run_example() {
    local example_name=$1
    local example_dir="${SOLAR_ROOT}/examples/${example_name}"
    local output_dir="/tmp/solar_test_${example_name}"
    
    echo -e "\n${YELLOW}Running ${example_name} example...${NC}"
    
    if [ ! -f "${example_dir}/run_solar.sh" ]; then
        echo -e "${RED}❌ ${example_name}/run_solar.sh not found${NC}"
        return 1
    fi
    
    export "SOLAR_${example_name^^}_OUTPUT_DIR"="${output_dir}"
    
    # Run the example
    if SOLAR_PYTHON="${PYTHON[0]}" bash "${example_dir}/run_solar.sh" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${example_name} example passed${NC}"
        
        # Verify key output files exist
        if [ -f "${output_dir}/graph/pytorch_graph.yaml" ] && \
           [ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ] && \
           [ -f "${output_dir}/analysis/analysis.yaml" ] && \
           [ -f "${output_dir}/perf/perf_Radeon_RX_9060_XT.yaml" ]; then
            echo -e "   📁 All expected outputs generated"
        else
            echo -e "${YELLOW}   ⚠️  Some outputs missing${NC}"
        fi
        
        # Cleanup
        rm -rf "${output_dir}"
        return 0
    else
        echo -e "${RED}❌ ${example_name} example failed${NC}"
        rm -rf "${output_dir}"
        return 1
    fi
}

# Refuse implicit environment mutation; install_uv.sh creates the pinned ROCm env.
if ! "${PYTHON[@]}" -c "import solar" 2>/dev/null; then
    echo -e "${RED}SOLAR is not installed. Run: bash install_uv.sh${NC}"
    exit 2
fi

# Run tests based on type
case $TEST_TYPE in
    quick)
        echo -e "${BLUE}Running quick smoke tests...${NC}"
        "${PYTHON[@]}" -m pytest tests/test_graph_processing.py::TestTorchviewProcessor::test_process_graph -v
        "${PYTHON[@]}" -m pytest tests/test_einsum_analyzer.py::TestEinsumAnalyzer::test_matmul -v
        echo -e "${GREEN}✅ Quick smoke tests passed${NC}"
        ;;
    graph)
        run_tests "test_graph_processing.py" "Graph Processing Tests (Stage 1: pytorch_graph.yaml)"
        ;;
    einsum)
        run_tests "test_einsum_analyzer.py" "Einsum Analyzer Tests (Stage 2: einsum_graph.yaml)"
        ;;
    model)
        run_tests "test_model_analyzer.py" "Model Analyzer Tests (Stages 3-4: analysis + perf)"
        ;;
    llm)
        run_tests "test_llm_agent.py" "LLM Agent Tests"
        ;;
    bert)
        run_tests "test_standalone_bert.py" "Standalone BERT Example Tests"
        ;;
    integration)
        run_tests "test_integration.py" "Integration Tests"
        ;;
    kernelbench)
        echo -e "\n${YELLOW}Testing Kernelbench compatibility...${NC}"
        "${PYTHON[@]}" -m pytest tests/ -k "kernelbench or Kernelbench" -v
        ;;
    examples)
        echo -e "${BLUE}Running example scripts...${NC}"
        
        EXAMPLES_PASSED=0
        EXAMPLES_FAILED=0
        
        for example in Attention BERT Conv2d Matmul; do
            if run_example "$example"; then
                EXAMPLES_PASSED=$((EXAMPLES_PASSED + 1))
            else
                EXAMPLES_FAILED=$((EXAMPLES_FAILED + 1))
            fi
        done
        
        echo -e "\n${BLUE}Examples Summary:${NC}"
        echo -e "  Passed: ${GREEN}${EXAMPLES_PASSED}${NC}"
        echo -e "  Failed: ${RED}${EXAMPLES_FAILED}${NC}"
        
        if [ $EXAMPLES_FAILED -gt 0 ]; then
            exit 1
        fi
        ;;
    unit)
        echo -e "${BLUE}Running unit tests only...${NC}"
        run_tests "test_graph_processing.py" "Graph Processing Tests"
        run_tests "test_einsum_analyzer.py" "Einsum Analyzer Tests"
        run_tests "test_model_analyzer.py" "Model Analyzer Tests"
        run_tests "test_llm_agent.py" "LLM Agent Tests"
        ;;
    all)
        echo -e "${BLUE}Running complete test suite...${NC}"

        if [ "$VERBOSE" = "-v" ] || [ "$VERBOSE" = "--verbose" ]; then
            "${PYTHON[@]}" -m pytest tests/ -v --tb=short
        else
            "${PYTHON[@]}" -m pytest tests/ -q
        fi
        
        echo -e "\n${GREEN}All tests passed!${NC}"
        ;;
    *)
        echo "Usage: $0 [test_type] [options]"
        echo ""
        echo "Test types:"
        echo "  all         - Run all tests (default)"
        echo "  quick       - Quick smoke tests"
        echo "  unit        - All unit tests only"
        echo "  integration - Integration tests only"
        echo "  examples    - Run all example scripts"
        echo ""
        echo "Pipeline stage tests:"
        echo "  graph       - Stage 1: PyTorch graph extraction (pytorch_graph.yaml)"
        echo "  einsum      - Stage 2: Einsum conversion (einsum_graph.yaml, einsum_graph_renamed.yaml)"
        echo "  model       - Stage 3-4: Analysis and performance prediction"
        echo ""
        echo "Component tests:"
        echo "  llm         - LLM agent and node registry"
        echo "  bert        - Standalone BERT example"
        echo ""
        echo "Benchmark compatibility:"
        echo "  kernelbench - Test kernelbench models"
        echo ""
        echo "Options:"
        echo "  -v, --verbose - Verbose output with detailed test names"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run all tests"
        echo "  $0 quick              # Quick smoke tests"
        echo "  $0 examples           # Run all example scripts"
        echo "  $0 unit -v            # Unit tests with verbose output"
        echo "  $0 integration        # Integration tests only"
        exit 1
        ;;
esac

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}         All tests completed!           ${NC}"
echo -e "${GREEN}========================================${NC}"
