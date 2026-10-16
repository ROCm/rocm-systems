# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Local HRR development image. The host only needs Docker.
# CMake, Ninja and Python 3 come from the TheRock manylinux build image
# that hrr-gpu.yml pins. This adds PyYAML for the HRR scripts, the GitHub
# CLI that hrr_capture_build.py uses to fetch a base, and GCC 13, the
# minimum emulation/rocjitsu/docs/building.md asks for.
FROM ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:cf4f6d9909056906e4a1c0bc4c245658a0af4c6a6174082a86adfa2d518b7aeb

USER root
RUN python3 -m pip install --no-cache-dir 'PyYAML==6.0.2' \
    && curl -fsSL https://cli.github.com/packages/rpm/gh-cli.repo -o /etc/yum.repos.d/gh-cli.repo \
    && dnf install -y gh gcc-toolset-13 \
    && dnf clean all \
    && groupadd -f video
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:${PATH}
WORKDIR /workspace
