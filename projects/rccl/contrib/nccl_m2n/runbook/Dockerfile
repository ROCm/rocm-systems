# Primus (TheRock HIP 7.14, OpenMPI in /opt/openmpi) + rdma-core v61 for
# Ruby's bnxt_re (kernel provider ABI rdmav59; image ships rdma-core 50 / rdmav34)
# + sshd on 2222 so mpirun can launch orted inside the container.
# The original base rocm/primus:v26.4-rccl-gfx950-run41014461271 is a local image, not on Docker Hub.
ARG BASE_IMAGE=rocm/primus:v26.4
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
      build-essential cmake pkg-config ninja-build git ca-certificates \
      libnl-3-dev libnl-route-3-dev libudev-dev libsystemd-dev libcap-dev \
      python3-docutils pandoc openssh-server openssh-client iproute2 \
 && rm -rf /var/lib/apt/lists/*

ARG RDMA_CORE_TAG=v61.0
RUN git clone --depth=1 --branch ${RDMA_CORE_TAG} https://github.com/linux-rdma/rdma-core.git /tmp/rdma-core \
 && cd /tmp/rdma-core \
 && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
        -DCMAKE_INSTALL_SYSCONFDIR=/etc -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 \
 && ninja -C build && ninja -C build install && ldconfig && rm -rf /tmp/rdma-core

ARG SSHD_PORT=2222
RUN mkdir -p /var/run/sshd /etc/ssh/keys && ssh-keygen -A \
 && sed -i "s|^#\?Port .*|Port ${SSHD_PORT}|" /etc/ssh/sshd_config \
 && sed -i 's|^#\?PermitRootLogin .*|PermitRootLogin yes|' /etc/ssh/sshd_config \
 && sed -i 's|^#\?PubkeyAuthentication .*|PubkeyAuthentication yes|' /etc/ssh/sshd_config \
 && sed -i 's|^#\?PasswordAuthentication .*|PasswordAuthentication no|' /etc/ssh/sshd_config \
 && sed -i 's|^#\?AuthorizedKeysFile.*|AuthorizedKeysFile /etc/ssh/keys/%u|' /etc/ssh/sshd_config \
 && sed -i 's|^#\?UsePAM .*|UsePAM no|' /etc/ssh/sshd_config

RUN R=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel \
 && printf '%s\n' \
      "export ROCM_PATH=$R" \
      "export PATH=/opt/openmpi/bin:$R/bin:$R/llvm/bin:/opt/venv/bin:\${PATH}" \
      "export LD_LIBRARY_PATH=/opt/openmpi/lib:$R/lib:$R/lib64:$R/llvm/lib:$R/lib/host-math/lib:$R/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-}" \
    > /etc/profile.d/rocm-env.sh \
 && sed -i '1r /etc/profile.d/rocm-env.sh' /etc/bash.bashrc

# nccl.m2n on PyTorch (RUNBOOK section 13). NVIDIA cuda-python goes: its cuda.core shadows the HIP shim of
# the RCCL nccl4py port and needs libcuda.so.1. setuptools < 82 for torch; hip-python dlopens an unversioned
# libamdhip64.so, which torch's _rocm_sdk_core lacks (../scripts/py_env.sh puts /opt/hipdev on LD_LIBRARY_PATH).
RUN /opt/venv/bin/python -m pip uninstall -y cuda-python cuda-bindings cuda-core cuda-pathfinder \
 && /opt/venv/bin/python -m pip install --no-cache-dir mpi4py hip-python==7.2.2.562.43 'setuptools>=80,<82' \
      'setuptools-scm>=10.2,<11' 'Cython>=3.1' \
 && mkdir -p /opt/hipdev \
 && ln -sf /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7 /opt/hipdev/libamdhip64.so

EXPOSE ${SSHD_PORT}
ENTRYPOINT []
CMD ["/usr/sbin/sshd", "-D", "-e"]
