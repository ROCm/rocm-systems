# =============================================================================
# Multi-ROCm-version rocSHMEM smoketest
#
# Installs multiple ROCm versions side-by-side (6.3, 6.4, 7.0, 7.2 + 7.13 TheRock),
# builds rocshmem against a default subset, runs ping-pong smoketest + Python wheel.
#
# Build from local checkout:
#   docker build -f docker/smoketest-rocm-multiversions/Dockerfile \
#     --build-arg USE_SRC=. -t rocshmem-multiversion .
#
# Build from GitHub branch:
#   docker build -f docker/smoketest-rocm-multiversions/Dockerfile \
#     --build-arg ROCSHMEM_REF=develop -t rocshmem-multiversion .
#
# Run (compile-only, no GPU needed):
#   docker run --rm rocshmem-multiversion --build-only
#
# Run (with GPUs, full smoketest):
#   docker run --rm --device=/dev/kfd --device=/dev/dri \
#     --security-opt seccomp=unconfined rocshmem-multiversion
# =============================================================================

FROM ubuntu:24.04

ARG ROCM_LEGACY_VERSIONS="6.3.3 6.4.3 7.0.2 7.2.4"
ARG ROCM_THEROCK_VERSION=7.13

###############################################################################
# System dependencies
###############################################################################
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool flex bison m4 \
      git make cmake ninja-build pkg-config \
      libnuma-dev libpthread-stubs0-dev libzstd-dev \
      zip zlib1g-dev wget unzip gpg ca-certificates \
      libssl-dev curl \
      python3-pip python3-setuptools python3-venv python3-dev python3-yaml \
      libibverbs-dev ibverbs-utils libibumad-dev libpci-dev librdmacm-dev \
      rdma-core infiniband-diags \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

###############################################################################
# ROCm 6.3 + 7.2 (traditional repo.radeon.com)
###############################################################################
RUN mkdir -p /etc/apt/keyrings \
 && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
 && for ver in ${ROCM_LEGACY_VERSIONS}; do \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
        https://repo.radeon.com/rocm/apt/$ver noble main" \
        >> /etc/apt/sources.list.d/rocm.list; \
    done \
 && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
      > /etc/apt/preferences.d/rocm-pin-600 \
 && apt-get update \
 && for ver in ${ROCM_LEGACY_VERSIONS}; do \
      apt-get install -y rocm${ver}; \
    done \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

###############################################################################
# ROCm 7.13 (TheRock repo.amd.com)
###############################################################################
RUN wget -qO- https://repo.amd.com/rocm/packages/gpg/rocm.gpg \
    | gpg --dearmor > /etc/apt/keyrings/amdrocm.gpg \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] \
      https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
      > /etc/apt/sources.list.d/amdrocm.list \
 && apt-get update \
 && apt-get install -y \
      amdrocm-core-dev${ROCM_THEROCK_VERSION} \
      amdrocm-core${ROCM_THEROCK_VERSION}-gfx942 \
      amdrocm-core${ROCM_THEROCK_VERSION}-gfx950 \
      amdrocm-developer-tools${ROCM_THEROCK_VERSION} \
      amdrocm-runtime-dev \
 && ln -sf /opt/rocm/core/.info /opt/rocm/.info \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

###############################################################################
# UCX (shared across all ROCm versions, built against the newest)
###############################################################################
ARG UCX_COMMIT=18770fdc1c3b5de202d14a088a14b734d2c4bbf3
ARG UCX_INSTALL=/opt/ucx
RUN git clone https://github.com/ROCm/ucx.git /tmp/ucx \
 && cd /tmp/ucx && git checkout ${UCX_COMMIT} \
 && ./autogen.sh \
 && ./contrib/configure-release \
      --prefix=${UCX_INSTALL} \
      --with-rocm=/opt/rocm \
      --enable-mt \
      --without-go --without-java --without-cuda --without-knem \
 && make -j$(nproc) install \
 && rm -rf /tmp/ucx
ENV PATH="${UCX_INSTALL}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${UCX_INSTALL}/lib:${LD_LIBRARY_PATH}"

###############################################################################
# OpenMPI (shared across all ROCm versions, internal PMIx/prrte)
###############################################################################
ARG OMPI_COMMIT=697a596dde68815fe50db3c2a75a42ddb41b5ef4
ARG OMPI_INSTALL=/opt/ompi
RUN git clone --recursive https://github.com/ROCm/ompi.git /tmp/ompi \
 && cd /tmp/ompi && git checkout ${OMPI_COMMIT} \
 && git submodule update --init --recursive \
 && python3 -m venv /tmp/ompi-venv \
 && /tmp/ompi-venv/bin/pip install -r docs/requirements.txt \
 && ./autogen.pl \
 && ./configure \
      --prefix=${OMPI_INSTALL} \
      --with-rocm=/opt/rocm \
      --with-ucx=${UCX_INSTALL} \
      --disable-oshmem \
      --with-prrte=internal \
      --with-hwloc=internal \
      --with-libevent=internal \
      --without-cuda \
      --disable-mpi-fortran \
      --without-ofi \
 && make -j$(nproc) install \
 && rm -rf /tmp/ompi /tmp/ompi-venv
ENV PATH="${OMPI_INSTALL}/bin:${PATH}"
RUN echo "${OMPI_INSTALL}/lib" > /etc/ld.so.conf.d/ompi.conf && ldconfig

###############################################################################
# TheRock from-source build (optional, opt-in with --build-arg BUILD_THEROCK=1)
# Builds rocshmem as a TheRock subproject to validate the super-build recipe.
# This is expensive (builds compiler + HIP + HSA from source).
###############################################################################
ARG BUILD_THEROCK=
ARG THEROCK_REF=main
ARG THEROCK_GPU_FAMILIES="gfx94X-all;gfx110X-all"
RUN if [ -n "${BUILD_THEROCK}" ]; then \
      git clone https://github.com/ROCm/TheRock.git /app/therock \
      && cd /app/therock && git checkout ${THEROCK_REF} \
      && cmake -B build -GNinja . \
           -DTHEROCK_ENABLE_ALL=OFF \
           -DTHEROCK_ENABLE_ROCSHMEM=ON \
           -DTHEROCK_AMDGPU_FAMILIES="${THEROCK_GPU_FAMILIES}" \
      && cmake --build build \
      && echo "TheRock super-build: OK"; \
    fi

###############################################################################
# Python wheel dependencies
###############################################################################
RUN pip install --break-system-packages pybind11 pytest

###############################################################################
# Environment
###############################################################################
WORKDIR /app
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    ROCSHMEM_HEAP_SIZE=6442450944
ENV ROCSHMEM_SRC=/app/rocshmem

###############################################################################
# Driver script (changes rarely, cached separately from source)
###############################################################################
COPY docker/smoketest-rocm-multiversions/build_and_test.sh /app/build_and_test.sh
RUN chmod +x /app/build_and_test.sh

###############################################################################
# rocshmem source — LAST so that source changes don't rebuild expensive layers
###############################################################################
ARG USE_SRC=
ARG ROCSHMEM_REF=develop

COPY . /app/rocshmem-context/

RUN if [ -n "${USE_SRC}" ]; then \
      ln -s /app/rocshmem-context /app/rocshmem; \
    else \
      git clone --no-checkout --filter=blob:none \
        https://github.com/ROCm/rocm-systems.git /app/rocm-systems \
      && cd /app/rocm-systems \
      && git sparse-checkout set --cone projects/rocshmem \
      && git checkout ${ROCSHMEM_REF} \
      && ln -s /app/rocm-systems/projects/rocshmem /app/rocshmem; \
    fi

ENTRYPOINT ["/app/build_and_test.sh"]
