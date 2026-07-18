# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

ARG OROJENESIS_BUILDER_IMAGE=ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04@sha256:86e11093b4a7ec2a79b1b6701d10e840a6994f21c7e05929b51eb9be361c683a
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.18@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03

FROM ${OROJENESIS_BUILDER_IMAGE} AS orojenesis-builder
ARG OROJENESIS_BUILDER_IMAGE
ARG OROJENESIS_COMMIT=97d52178bf9a9c209bf79be96b87c164bcd35625
ENV DEBIAN_FRONTEND=noninteractive

RUN test "$(dpkg --print-architecture)" = "amd64" || \
    (echo "SOLAR Orojenesis supports Linux x86_64 only." >&2; exit 2)

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git ninja-build python3 scons \
      libconfig++-dev libboost-dev libboost-iostreams-dev \
      libboost-serialization-dev libyaml-cpp-dev libncurses-dev \
      libtinfo-dev libgpm-dev libgmp-dev

# GCC 14 no longer exposes std::uint64_t transitively through unrelated
# headers. Inject the standard header so the pinned source remains unmodified.
COPY --chmod=0755 docker/g++-cstdint /tmp/orojenesis-compiler/g++
COPY docker/write_orojenesis_provenance.py /tmp/write_orojenesis_provenance.py
RUN git clone https://github.com/NVlabs/timeloop.git /opt/orojenesis && \
    git -C /opt/orojenesis checkout ${OROJENESIS_COMMIT} && \
    ln -s ../pat-public/src/pat /opt/orojenesis/src/pat && \
    PATH="/tmp/orojenesis-compiler:${PATH}" \
      scons -C /opt/orojenesis --static -j4 && \
    test "$(git -C /opt/orojenesis rev-parse HEAD)" = "${OROJENESIS_COMMIT}" && \
    test -x /opt/orojenesis/bin/timeloop-mapper && \
    python3 /tmp/write_orojenesis_provenance.py \
      --home /opt/orojenesis \
      --expected-commit "${OROJENESIS_COMMIT}" \
      --builder-image "${OROJENESIS_BUILDER_IMAGE}" \
      --compiler-wrapper /tmp/orojenesis-compiler/g++ \
      --output /opt/orojenesis/orojenesis-provenance.json

FROM orojenesis-builder AS orojenesis-test
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      python3-pytest python3-yaml
WORKDIR /workspace
COPY pyproject.toml /workspace/pyproject.toml
COPY solar /workspace/solar
COPY tests/test_orojenesis_multi_integration.py /workspace/tests/test_orojenesis_multi_integration.py
ENV PYTHONPATH=/workspace \
    SOLAR_OROJENESIS_HOME=/opt/orojenesis \
    SOLAR_REQUIRE_OROJENESIS_INTEGRATION=1
RUN python3 -m pytest tests/test_orojenesis_multi_integration.py -v -rs

FROM ${UV_IMAGE} AS uv

FROM ${ROCM_IMAGE} AS final

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
    apt-get update && apt-get install -y --no-install-recommends python3 sudo && \
    command -v hipcc && command -v rocprofv3 && command -v amd-smi

COPY --from=orojenesis-builder --chmod=0755 \
  /opt/orojenesis/bin/timeloop-mapper /opt/orojenesis/bin/timeloop-mapper
COPY --from=orojenesis-builder \
  /opt/orojenesis/orojenesis-provenance.json /opt/orojenesis/orojenesis-provenance.json
COPY --from=uv /uv /usr/local/bin/uv

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
