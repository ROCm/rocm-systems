FROM ghcr.io/rocm/therock_build_manylinux_x86_64:main as builder
RUN set -e; \
    echo "== Toolchain =="; \
    command -v cc cmake cargo; \
    cc --version | head -1; \
    cmake --version | head -1; \
    cargo --version

FROM builder as mirage-builder


FROM builder as rocjitsu-builder

FROM ubuntu:24.04 as output
