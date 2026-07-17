# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete
FROM ${ROCM_IMAGE}

RUN test "$(dpkg --print-architecture)" = "amd64" || \
    (echo "SOLAR ROCm does not support Linux aarch64; use Linux x86_64." >&2; exit 2)

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    HSA_OVERRIDE_GFX_VERSION=12.0.0 \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/venv/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    UV_PROJECT_ENVIRONMENT=/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build python3 python3-dev sudo && \
    command -v hipcc && command -v rocprofv3 && command -v amd-smi

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --gid ${HOST_GID} solar || true && \
    useradd --uid ${HOST_UID} --gid ${HOST_GID} --create-home --shell /bin/bash solar || true && \
    mkdir -p /workspace /venv && chown -R ${HOST_UID}:${HOST_GID} /workspace /venv

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY solar ./solar
COPY configs ./configs
COPY patches ./patches
COPY third_party ./third_party
USER ${HOST_UID}:${HOST_GID}
RUN --mount=type=cache,target=/home/solar/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv sync --frozen --no-dev

ENTRYPOINT ["/venv/bin/solar-evaluate"]
