# syntax=docker/dockerfile:1
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build rocjitsu -- the C++ emulator and the Rust CLI that drives it --
# inside the TheRock manylinux container, and assemble the result into a
# runtime image.
#
# WHY manylinux: a rocjitsu session bind-mounts its own CLI binary and the
# rocjitsu KMD interposer into the workload container. Built on a modern
# host they link a newer glibc than older images carry (e.g. vllm jammy =
# glibc 2.35) -> `version 'GLIBC_2.39' not found`. Building inside the
# manylinux image links against an old glibc with broad forward
# compatibility, so the resulting binaries run in (almost) any target
# container WITHOUT the `--hack`/derived-image workaround.
#
# The result has the standard prefix layout under /opt/rocjitsu:
#   /opt/rocjitsu/bin/rocjitsu                 (the CLI)
#   /opt/rocjitsu/bin/mirage                   (symlink, the CLI's old name)
#   /opt/rocjitsu/lib/librocjitsu.so           (combined rocjitsu library:
#                                               VM API + KMD interposer)
#   /opt/rocjitsu/lib/librocjitsu_hooks.so     (DBT HSA hooks)
#   /opt/rocjitsu/share/rocjitsu/configs/*.json
# The CLI searches `<prefix>/lib` beside its own binary before any rocjitsu
# build it can reach by walking up, so `/opt/rocjitsu/bin/rocjitsu` finds
# its sibling `/opt/rocjitsu/lib/*.so`.
#
# WHY one builder stage: this used to be two, because the CLI was a
# separate project that only shared configuration data with the emulator.
# It is now `rocjitsu/cli/`, wired into the same cmake build, and the
# cargo invocation is a custom target with no dependencies -- so ninja
# already runs it alongside the C++ compilation. One stage, one configure,
# one install, and the parallelism is the build tool's business.
#
# Caching: the cargo registry/git caches and target dir, and the cmake
# FetchContent deps and build tree, are all kept in BuildKit cache mounts
# so incremental rebuilds are fast and re-runs skip downloads.
#
# Build (context = this `emulation/` directory):
#   DOCKER_BUILDKIT=1 docker build -f emulation/Dockerfile -t rocjitsu emulation
#   # or with podman:  podman build -f emulation/Dockerfile -t rocjitsu emulation
#
# Build args:
#   BUILD_IMAGE    builder base (default ghcr.io/rocm/therock_build_manylinux_x86_64:main)
#   RUNTIME_IMAGE  final image base (default ubuntu:24.04)
#   CARGO_PROFILE  release or debug (default release)
#   RJ_LOG_GROUPS  rocjitsu compile-time log groups (default VM; see
#                  rocjitsu/cmake/rj_log.cmake: OFF, ALL, or names like VM,CP)
#
# Extract artifacts to the host without running the image, e.g.:
#   id=$(docker create rocjitsu); docker cp "$id:/opt/rocjitsu" ./out; docker rm "$id"

ARG BUILD_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64:main
ARG RUNTIME_IMAGE=ubuntu:24.04
ARG CARGO_PROFILE=release
ARG RJ_LOG_GROUPS=VM

# --------------------------------------------------------------------------
# Builder: cmake drives both the C++ emulator and the Rust CLI.
# --------------------------------------------------------------------------
FROM ${BUILD_IMAGE} AS builder
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG CARGO_PROFILE
ARG RJ_LOG_GROUPS

# Out of the source tree so it can be a cache mount. The CLI's cmake
# wrapper asks `cargo metadata` where the binary will land rather than
# assuming `<source>/target`, so moving it here needs no other change.
ENV CARGO_TARGET_DIR=/cargo-target

RUN set -e; \
    echo "== Toolchain =="; \
    command -v cc cmake cargo; \
    cc --version | head -1; \
    cmake --version | head -1; \
    cargo --version

# The whole project: `rocjitsu/cli` is part of it, and `builtin/build.rs`
# reads `rocjitsu/configs/*.json`, so there is nothing to copy separately.
COPY rocjitsu /src/rocjitsu

# CRITICAL: the manylinux image is Red Hat based, so GNUInstallDirs would
# default CMAKE_INSTALL_LIBDIR to lib64. Force `lib` so the rocjitsu shared
# libs land in /opt/rocjitsu/lib (matches the CLI's `../lib` search).
#
# Cache mounts:
#   /deps          -> FetchContent (googletest + flatbuffers) download/build tree
#   /rjbuild       -> cmake/ninja build tree (incremental rebuilds)
#   /cargo-target  -> incremental cargo build artifacts
#   /root/.cargo/… -> downloaded and compiled crate dependencies
RUN --mount=type=cache,target=/deps,sharing=locked \
    --mount=type=cache,target=/rjbuild,sharing=locked \
    --mount=type=cache,target=/cargo-target,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    echo "== Configuring rocjitsu (RJ_LOG_GROUPS=${RJ_LOG_GROUPS:-OFF}) ==" && \
    cmake -G Ninja -S /src/rocjitsu -B /rjbuild \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
        -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++" \
        -DCMAKE_INSTALL_PREFIX=/opt/rocjitsu \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DFETCHCONTENT_BASE_DIR=/deps \
        -DRJ_LOG_GROUPS="${RJ_LOG_GROUPS:-OFF}" \
        -DROCJITSU_CARGO_PROFILE="${CARGO_PROFILE}" \
        -DBUILD_TESTING=OFF && \
    echo "== Building rocjitsu ==" && \
    ninja -C /rjbuild -j"$(nproc)" && \
    echo "== Installing rocjitsu ==" && \
    cmake --install /rjbuild && \
    ls -lh /opt/rocjitsu/bin/rocjitsu /opt/rocjitsu/lib/librocjitsu*.so

# --------------------------------------------------------------------------
# Final image.
# --------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE} AS output
LABEL org.opencontainers.image.title="rocjitsu" \
      org.opencontainers.image.description="rocjitsu (manylinux-built, glibc-portable)"

COPY --from=builder /opt/rocjitsu /opt/rocjitsu

ENV PATH=/opt/rocjitsu/bin:${PATH}
WORKDIR /opt/rocjitsu
CMD ["/opt/rocjitsu/bin/rocjitsu", "--help"]
