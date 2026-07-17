#!/usr/bin/env bash
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

set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: SOLAR ROCm supports Linux x86_64 only." >&2
    exit 2
fi

# Install SOLAR and its dependencies.
#
# This script:
#   1. Installs SOLAR dependencies
#   2. Installs SOLAR in editable mode (including its vendored torchview code)
#
# Usage:
#   bash install.sh              # Install everything
#   bash install.sh --skip-torch # Skip torch installation (if already installed)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_TORCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-torch)
            SKIP_TORCH=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash install.sh [--skip-torch]"
            echo ""
            echo "Options:"
            echo "  --skip-torch  Skip PyTorch installation (use if already installed)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Solar Installation ==="
echo ""

# Step 1: Install SOLAR dependencies
echo "==> Step 1: Installing SOLAR dependencies..."
cd "${SCRIPT_DIR}"

if [[ "$SKIP_TORCH" == "true" ]]; then
    echo "  Skipping torch (--skip-torch specified)."
    # Preserve every other pinned/runtime dependency, including Triton ROCm
    # and the official wheel indexes declared by requirements.txt.
    pip install -r <(sed '/^torch==/d' requirements.txt)
else
    pip install -r requirements.txt
fi

# Step 2: Install SOLAR and its vendored torchview package.
echo ""
echo "==> Step 2: Installing SOLAR in editable mode..."
pip install -e . --no-deps

echo ""
echo "=== Installation complete ==="
echo ""
echo "SOLAR: ${SCRIPT_DIR}"
echo ""
echo "To verify:"
echo "  python -c 'from solar._vendor import torchview; print(torchview.__file__)'"
echo "  python -c 'from solar.graph import PyTorchProcessor; print(\"OK\")'"
