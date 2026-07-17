# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete
FROM ${ROCM_IMAGE}

RUN test "$(dpkg --print-architecture)" = "amd64" || \
    (echo "SOLAR ROCm does not support Linux aarch64; use Linux x86_64." >&2; exit 2)

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/venv/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    HOME=/home/solar \
    UV_PROJECT_ENVIRONMENT=/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    SOLAR_OROJENESIS_HOME=/opt/orojenesis

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build python3 python3-dev sudo \
      scons libconfig++-dev libboost-dev libboost-iostreams-dev \
      libboost-serialization-dev libyaml-cpp-dev libncurses-dev \
      libtinfo-dev libgpm-dev libgmp-dev && \
    command -v hipcc && command -v rocprofv3 && command -v amd-smi

ARG OROJENESIS_COMMIT=97d52178bf9a9c209bf79be96b87c164bcd35625
RUN git clone https://github.com/NVlabs/timeloop.git /opt/orojenesis && \
    git -C /opt/orojenesis checkout ${OROJENESIS_COMMIT} && \
    ln -s ../pat-public/src/pat /opt/orojenesis/src/pat && \
    scons -C /opt/orojenesis --static -j4 && \
    test "$(git -C /opt/orojenesis rev-parse HEAD)" = "${OROJENESIS_COMMIT}" && \
    test -x /opt/orojenesis/bin/timeloop-mapper

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --non-unique --gid ${HOST_GID} solar && \
    useradd --non-unique --uid ${HOST_UID} --gid solar --create-home --shell /bin/bash solar && \
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
