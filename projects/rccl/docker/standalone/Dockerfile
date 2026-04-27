# standalone/Dockerfile
#
# Self-contained image for multi-node RCCL / MPI workloads on AMD GPUs with
# Mellanox networking.  Includes UCX, OpenMPI, RCCL, RCCL-Tests, SSH server,
# and Mellanox RDMA libraries.  No external orchestration tooling required.
#
# Build:
#   docker build -t rccl-mn:standalone -f Dockerfile .
#
# With overrides (any subset):
#   docker build \
#     --build-arg ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.4-complete \
#     --build-arg GPU_TARGETS=gfx942 \
#     --build-arg UCX_VERSION=1.16.0 \
#     --build-arg OMPI_VERSION=4.1.6 \
#     --build-arg ROCM_SYSTEMS_BRANCH=develop \
#     -t rccl-mn:standalone -f Dockerfile .
#
# Run with launch.sh (recommended) or:
#   docker run -d --rm --name rccl-mn --network host \
#       --device /dev/kfd --device /dev/dri --device /dev/infiniband \
#       --shm-size 64g --ulimit memlock=-1 \
#       -v ~/.docker-ssh-keys:/opt/ssh-keys:ro \
#       rccl-mn:standalone

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.4-complete
FROM ${ROCM_IMAGE}

# ---------------------------------------------------------------------------
# Build-time configuration
# ---------------------------------------------------------------------------
ARG SSH_PORT=2224
ARG CONTAINER_USER=ubuntu
ARG GPU_TARGETS=gfx942
ARG UCX_VERSION=1.16.0
ARG OMPI_VERSION=4.1.6

# RCCL and RCCL-Tests are shipped from the rocm-systems monorepo
# (https://github.com/ROCm/rocm-systems).  We sparse-checkout only the three
# projects we need: rccl, rccl-tests, and rocshmem (RCCL build dependency).
ARG ROCM_SYSTEMS_REPO=https://github.com/ROCm/rocm-systems
ARG ROCM_SYSTEMS_BRANCH=develop

ENV ROCM_PATH=/opt/rocm
ENV WORKDIR=/workspace
ENV SSH_PORT=${SSH_PORT}
ENV CONTAINER_USER=${CONTAINER_USER}
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV UCX_WARN_UNUSED_ENV_VARS=n
ENV NCCL_DEBUG=VERSION

WORKDIR ${WORKDIR}

# ---------------------------------------------------------------------------
# System packages: build tools, SSH, networking, Mellanox RDMA, GPU diagnostics
# ---------------------------------------------------------------------------
RUN echo ">>> Installing system packages..." \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates git make rocm-cmake ninja-build gfortran \
       build-essential autoconf automake libtool pkg-config gettext-base \
       libomp5 libomp-dev libbfd-dev libboost-all-dev \
       libnuma1 libnuma-dev libpthread-stubs0-dev libzstd-dev \
       libssl-dev libdrm-dev libelf-dev libgl-dev libglx-dev libsimde-dev \
       libncursesw5-dev liblzma-dev xz-utils zlib1g-dev xxd \
       lcov zip unzip chrpath doxygen lshw vim less wget curl \
       python3-pip python3-setuptools python3-venv python3-dev \
       python3-tk python3-yaml \
       openssh-server openssh-client sudo \
       iproute2 net-tools iputils-ping pciutils \
       ibverbs-utils libibverbs-dev librdmacm-dev rdma-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Newer cmake than what the distro ships
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh \
    && chmod +x cmake-3.28.0-linux-x86_64.sh \
    && bash ./cmake-3.28.0-linux-x86_64.sh --prefix=/usr --exclude-subdir --skip-license \
    && rm cmake-3.28.0-linux-x86_64.sh

# ---------------------------------------------------------------------------
# UCX with ROCm + RDMA support
# ---------------------------------------------------------------------------
ENV UCX_INSTALL_PREFIX=/opt/ucx
RUN echo ">>> Building UCX ${UCX_VERSION}..." \
    && wget -q https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz \
    && mkdir -p ucx \
    && tar -zxf ucx-${UCX_VERSION}.tar.gz -C ucx --strip-components=1 \
    && cd ucx && mkdir build && cd build \
    && ../configure --prefix=${UCX_INSTALL_PREFIX} \
                    --with-rocm=${ROCM_PATH} \
                    --with-verbs --with-rdmacm --enable-mt \
                    --disable-examples --silent \
    && make -j$(nproc) install \
    && cd ../.. && rm -rf ucx ucx-${UCX_VERSION}.tar.gz

# ---------------------------------------------------------------------------
# OpenMPI built against UCX
# ---------------------------------------------------------------------------
ENV MPI_INSTALL_PREFIX=/opt/ompi
RUN echo ">>> Building OpenMPI ${OMPI_VERSION}..." \
    && OMPI_SERIES="${OMPI_VERSION%.*}" \
    && wget -q https://download.open-mpi.org/release/open-mpi/v${OMPI_SERIES}/openmpi-${OMPI_VERSION}.tar.gz \
    && mkdir -p ompi4 \
    && tar -zxf openmpi-${OMPI_VERSION}.tar.gz -C ompi4 --strip-components=1 \
    && cd ompi4 && mkdir build && cd build \
    && ../configure --prefix=${MPI_INSTALL_PREFIX} \
                    --with-ucx=${UCX_INSTALL_PREFIX} \
                    --disable-oshmem --disable-mpi-fortran --disable-mpi-cxx \
                    --enable-orterun-prefix-by-default --silent \
    && make -j$(nproc) install \
    && cd ../.. && rm -rf ompi4 openmpi-${OMPI_VERSION}.tar.gz

# ---------------------------------------------------------------------------
# Sparse-checkout rocm-systems with just the projects we need
# ---------------------------------------------------------------------------
ENV ROCM_SYSTEMS_DIR=${WORKDIR}/rocm-systems
RUN echo ">>> Sparse-cloning rocm-systems (${ROCM_SYSTEMS_BRANCH})..." \
    && git clone --no-checkout --filter=blob:none "${ROCM_SYSTEMS_REPO}" "${ROCM_SYSTEMS_DIR}" \
    && cd "${ROCM_SYSTEMS_DIR}" \
    && git sparse-checkout init --cone \
    && git sparse-checkout set projects/rccl projects/rccl-tests projects/rocshmem \
    && git checkout "${ROCM_SYSTEMS_BRANCH}" \
    && du -sh projects/rccl projects/rccl-tests projects/rocshmem 2>/dev/null

# ---------------------------------------------------------------------------
# RCCL (built from source)
# ---------------------------------------------------------------------------
ENV RCCL_INSTALL_PREFIX=${WORKDIR}/rccl/install
RUN echo ">>> Building RCCL for ${GPU_TARGETS}..." \
    && cd "${ROCM_SYSTEMS_DIR}/projects/rccl" \
    && ./install.sh --amdgpu_targets=${GPU_TARGETS} --prefix=${RCCL_INSTALL_PREFIX}

# ---------------------------------------------------------------------------
# RCCL-Tests with MPI support
# ---------------------------------------------------------------------------
RUN echo ">>> Building RCCL-Tests..." \
    && cd "${ROCM_SYSTEMS_DIR}/projects/rccl-tests" \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
             -DUSE_MPI=ON \
             -DCMAKE_PREFIX_PATH="${RCCL_INSTALL_PREFIX};${MPI_INSTALL_PREFIX}" \
             -DGPU_TARGETS=${GPU_TARGETS} .. \
    && make -j$(nproc)

# ---------------------------------------------------------------------------
# SSH daemon: host keys + sshd_config tuned for password-less inter-node
# ---------------------------------------------------------------------------
RUN echo ">>> Configuring sshd on port ${SSH_PORT}..." \
    && ssh-keygen -A \
    && mkdir -p /run/sshd \
    && sed -i "s/^#*Port .*/Port ${SSH_PORT}/"                            /etc/ssh/sshd_config \
    && sed -i 's/^#*PermitRootLogin .*/PermitRootLogin yes/'              /etc/ssh/sshd_config \
    && sed -i 's/^#*PubkeyAuthentication .*/PubkeyAuthentication yes/'    /etc/ssh/sshd_config \
    && sed -i 's/^#*PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config

# ---------------------------------------------------------------------------
# Non-root user (entrypoint remaps UID/GID at runtime to match the host)
# ---------------------------------------------------------------------------
RUN echo ">>> Creating user '${CONTAINER_USER}'..." \
    && (groupadd -g 109 render 2>/dev/null || true) \
    && (groupadd -g 1000 ${CONTAINER_USER} 2>/dev/null || true) \
    && (useradd -u 1000 -g 1000 -m -s /bin/bash ${CONTAINER_USER} 2>/dev/null || true) \
    && (usermod -aG video,render ${CONTAINER_USER} 2>/dev/null || true) \
    && echo "${CONTAINER_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ---------------------------------------------------------------------------
# Container-wide environment (PATH, LD_LIBRARY_PATH, NCCL/Mellanox tunables)
# ---------------------------------------------------------------------------
ENV PATH="${MPI_INSTALL_PREFIX}/bin:${UCX_INSTALL_PREFIX}/bin:${ROCM_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${RCCL_INSTALL_PREFIX}:${MPI_INSTALL_PREFIX}/lib:${UCX_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

# Mellanox / NCCL defaults.  Override at run time (-e NCCL_IB_HCA=...) when
# your cluster uses a different HCA naming scheme or GID index.
ENV NCCL_IB_HCA=mlx5
ENV NCCL_IB_GID_INDEX=3
ENV NCCL_IB_TIMEOUT=23
ENV NCCL_IB_RETRY_CNT=7

# Mount stub for SSH keys delivered by the host
RUN mkdir -p /opt/ssh-keys

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE ${SSH_PORT}
ENTRYPOINT ["/opt/entrypoint.sh"]
