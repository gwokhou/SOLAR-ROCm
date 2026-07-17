#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install Solar using uv (https://github.com/astral-sh/uv).
#
# This script:
#   1. Ensures uv is installed (installs to ~/.local/bin or uses existing)
#   2. Runs uv sync --frozen --python 3.12
#
# The pinned dependency graph uses the patched torchview source vendored in
# third_party/torchview; no network clone or post-sync replacement is needed.
#
# Prerequisites: Python 3.12 on PATH (or uv will download it).
#
# Usage:
#   bash install_uv.sh              # Full ROCm install
#   bash install_uv.sh --help

set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: SOLAR ROCm supports Linux x86_64 only." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: bash install_uv.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Solar installation (uv) ==="
echo ""

# Step 0: Ensure uv is installed
echo "==> Step 0: Ensuring uv is installed..."
if command -v uv >/dev/null 2>&1; then
    echo "  uv found: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "  ERROR: uv not found after install. Add ~/.local/bin to PATH and retry."
        exit 1
    fi
fi
echo ""

# Step 1: frozen ROCm environment
echo "==> Step 1: uv sync --frozen --python 3.12..."
cd "${SCRIPT_DIR}"
uv sync --frozen --python 3.12
echo "  Sync complete."
echo ""

echo "=== Installation complete ==="
echo ""
echo "Virtual env: ${SCRIPT_DIR}/.venv"
echo "Activate with: source ${SCRIPT_DIR}/.venv/bin/activate"
echo ""
echo "To verify:"
echo "  uv run python -c 'from solar._vendor import torchview; print(torchview.__file__)'"
echo "  uv run python -c 'from solar.graph import PyTorchProcessor; print(\"OK\")'"
echo ""
echo "The committed uv.lock pins the ROCm environment."
