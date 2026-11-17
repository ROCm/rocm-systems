# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# isadiff-ci: rj_isa_diff_pr on LLVM PRs. Build with `isadiff-ci build REV`; the context
# holds rocjitsu-src.tar (git archive of the tool) and ci/ (this directory).
FROM ubuntu:24.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git python3 \
      python3-pytest cmake ninja-build ccache g++-13 clang-18 lld-18 llvm-18 \
 && install -d -m 0755 /etc/apt/keyrings \
 && curl -fsSL -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
      https://cli.github.com/packages/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

FROM base AS tool
COPY rocjitsu-src.tar /tmp/
RUN mkdir /src && tar -xf /tmp/rocjitsu-src.tar -C /src \
 && cmake -G Ninja -S /src/emulation/rocjitsu -B /build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
 && ninja -C /build rj_isa_diff

FROM base
ARG TOOL_SHA=unknown
COPY --from=tool /build/tools/isa_diff/rj_isa_diff /build/tools/isa_diff/rj_isa_diff_pr \
     /build/tools/isa_diff/rj_isa_diff_mutants /opt/isadiff/bin/
COPY ci/ /opt/isadiff/ci/
# uid 1000 = nod, so files in the bind-mounted results dir belong to nod. The /work dirs
# exist with that owner, so fresh named volumes mounted there inherit it.
RUN (userdel -r ubuntu || true) \
 && useradd -m -u 1000 -U ci \
 && install -d -o ci -g ci /work /work/llvm-project /work/builds /work/ccache \
      /work/results /work/corpus
LABEL org.opencontainers.image.revision=$TOOL_SHA
ENV ISADIFF_TOOL_SHA=$TOOL_SHA \
    PATH=/opt/isadiff/bin:$PATH \
    RJ_ISA_DIFF_LLVM_BIN=/usr/lib/llvm-18/bin \
    CC=clang-18 CXX=clang++-18 LDFLAGS=-fuse-ld=lld \
    CCACHE_DIR=/work/ccache CCACHE_MAXSIZE=60G \
    GH_PROMPT_DISABLED=1 GH_NO_UPDATE_NOTIFIER=1 PYTHONDONTWRITEBYTECODE=1
USER ci
WORKDIR /work
ENTRYPOINT ["/opt/isadiff/ci/entrypoint.sh"]
