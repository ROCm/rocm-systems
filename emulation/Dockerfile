# syntax=docker/dockerfile:1
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build mirage (Rust) AND rocjitsu (C++) inside the TheRock manylinux
# container, in PARALLEL, and assemble both into a single runtime image.
#
# WHY manylinux: mirage's per-session host bind-mounts its own `mirage`
# binary and the rocjitsu KMD interposer into the workload container.
# Built on a modern host they link a newer glibc than older images carry
# (e.g. vllm jammy = glibc 2.35) -> `version 'GLIBC_2.39' not found`.
# Building inside the manylinux image links against an old glibc with
# broad forward compatibility, so the resulting binaries run in (almost)
# any target container WITHOUT the `--hack`/derived-image workaround.
#
# The result has the standard prefix layout under /opt/mirage:
#   /opt/mirage/bin/mirage                  (Rust CLI/host)
#   /opt/mirage/bin/rocjitsu                (rocjitsu CLI)
#   /opt/mirage/lib/librocjitsu.so          (combined rocjitsu library:
#                                            VM API + KMD interposer)
#   /opt/mirage/lib/librocjitsu_hooks.so    (DBT HSA hooks)
#   /opt/mirage/share/rocjitsu/configs/*.json
# `mirage` searches `../lib` relative to its own binary, so
# `/opt/mirage/bin/mirage` finds its sibling `/opt/mirage/lib/*.so`.
#
# WHY two builder stages: rocjitsu (cmake/ninja C++) and mirage (cargo
# Rust) have NO build-time dependency on each other -- rocjitsu_sys loads
# the rocjitsu library at runtime via dlopen, it does not link or bindgen
# against it. BuildKit therefore builds `rocjitsu-builder` and
# `mirage-builder` concurrently, then the final stage merges both.
#
# Caching: cargo registry/git + the cargo target dir, and the cmake
# FetchContent deps + cmake build tree, are all kept in BuildKit cache
# mounts so incremental rebuilds are fast and re-runs skip downloads.
#
# Build (context = this `emulation/` directory):
#   DOCKER_BUILDKIT=1 docker build -f emulation/Dockerfile -t mirage emulation
#   # or with podman:  podman build -f emulation/Dockerfile -t mirage emulation
#
# Build args:
#   BUILD_IMAGE    builder base (default ghcr.io/rocm/therock_build_manylinux_x86_64:main)
#   RUNTIME_IMAGE  final image base (default ubuntu:24.04)
#   CARGO_PROFILE  release or debug (default release)
#   RJ_LOG_GROUPS  rocjitsu compile-time log groups (default VM; see
#                  rocjitsu/cmake/rj_log.cmake: OFF, ALL, or names like VM,CP)
#
# Extract artifacts to the host without running the image, e.g.:
#   id=$(docker create mirage); docker cp "$id:/opt/mirage" ./out; docker rm "$id"

ARG BUILD_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64:main
ARG RUNTIME_IMAGE=ubuntu:24.04
ARG CARGO_PROFILE=release
ARG RJ_LOG_GROUPS=VM

# --------------------------------------------------------------------------
# Common builder base: shared toolchain setup so the heavy image is pulled
# and prepared ONCE, then forked into the two parallel build stages.
# --------------------------------------------------------------------------
FROM ${BUILD_IMAGE} AS builder-base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Send cargo's registry/git caches and rustup to fixed, cache-mountable
# locations regardless of where the image already keeps them.
RUN set -e; \
    echo "== Toolchain =="; \
    command -v cc cmake cargo; \
    cc --version | head -1; \
    cmake --version | head -1; \
    cargo --version
# --------------------------------------------------------------------------
# rocjitsu (C++ / cmake + ninja) -- builds in parallel with mirage.
# --------------------------------------------------------------------------
FROM builder-base AS rocjitsu-builder
ARG RJ_LOG_GROUPS

# Only the rocjitsu subtree is needed; copying it alone keeps this stage's
# cache valid when only mirage sources change.
COPY rocjitsu /src/rocjitsu

# CRITICAL: the manylinux image is Red Hat based, so GNUInstallDirs would
# default CMAKE_INSTALL_LIBDIR to lib64. Force `lib` so the rocjitsu shared
# libs land in /opt/mirage/lib (matches mirage's `../lib` search).
#
# Cache mounts:
#   /deps    -> FetchContent (googletest + flatbuffers) download/build tree
#   /rjbuild -> cmake/ninja build tree (incremental rebuilds)
RUN --mount=type=cache,target=/deps,sharing=locked \
    --mount=type=cache,target=/rjbuild,sharing=locked \
    echo "== Configuring rocjitsu (RJ_LOG_GROUPS=${RJ_LOG_GROUPS:-OFF}) ==" && \
    cmake -G Ninja -S /src/rocjitsu -B /rjbuild \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
        -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++" \
        -DCMAKE_INSTALL_PREFIX=/opt/mirage \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DFETCHCONTENT_BASE_DIR=/deps \
        -DRJ_LOG_GROUPS="${RJ_LOG_GROUPS:-OFF}" \
        -DBUILD_TESTING=OFF && \
    echo "== Building rocjitsu ==" && \
    ninja -C /rjbuild -j"$(nproc)" && \
    echo "== Installing rocjitsu ==" && \
    cmake --install /rjbuild && \
    ls -lh /opt/mirage/lib/librocjitsu*.so

# --------------------------------------------------------------------------
# mirage (Rust / cargo) -- builds in parallel with rocjitsu.
# --------------------------------------------------------------------------
FROM builder-base AS mirage-builder
ARG CARGO_PROFILE
ENV CARGO_PROFILE=${CARGO_PROFILE}

# Whole mirage workspace (all crates live under mirage/).
COPY mirage /src/mirage
WORKDIR /src/mirage

# Cache mounts:
#   /opt/cargo/registry, /opt/cargo/git -> downloaded + compiled crate deps
#   target/                             -> incremental cargo build artifacts
# The built binary is copied OUT of the (ephemeral) target cache mount into
# /opt/mirage so it persists in the image layer.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/src/mirage/target,sharing=locked \
    echo "== Building mirage (cargo --$CARGO_PROFILE) ==" && \
    if [ "$CARGO_PROFILE" = "release" ]; then PROFILE_FLAG=--release; else PROFILE_FLAG=; fi && \
    cargo build $PROFILE_FLAG && \
    mkdir -p /opt/mirage/bin && \
    install -m 0755 "target/${CARGO_PROFILE}/mirage" /opt/mirage/bin/mirage && \
    ls -lh /opt/mirage/bin/mirage

# --------------------------------------------------------------------------
# Final image: merge both prefixes into a single /opt/mirage.
# --------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE} AS output
LABEL org.opencontainers.image.title="mirage" \
      org.opencontainers.image.description="mirage + rocjitsu (manylinux-built, glibc-portable)"

# rocjitsu contributes lib/*.so, bin/rocjitsu, share/rocjitsu/configs;
# mirage contributes bin/mirage. The two COPYs merge under /opt/mirage.
COPY --from=rocjitsu-builder /opt/mirage /opt/mirage
COPY --from=mirage-builder   /opt/mirage /opt/mirage

ENV PATH=/opt/mirage/bin:${PATH}
WORKDIR /opt/mirage
CMD ["/opt/mirage/bin/mirage", "--help"]
