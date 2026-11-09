#!/bin/bash
# Build libnccl_m2n.so inside manylinux_2_28 and stage it into bindings/nccl_m2n (cibuildwheel before-all).
# Compiler and HIP: TheRock rocm[devel] from M2N_ROCM_INDEX (default the stable index, latest GA release; nightly
# or release-candidate indexes work too), the latest one unless M2N_ROCM_DEVEL_VERSION pins a version.
# RCCL headers and librccl.so to link against: M2N_RCCL_HOME, an RCCL >= 2.30.5 install (nccl_m2n.h rejects older),
# not part of the wheel.
set -eo pipefail
D=$(dirname "$(readlink -f "$0")")
P=$(readlink -f "$D/../../..")   # projects/rccl
: "${M2N_RCCL_HOME:?set M2N_RCCL_HOME to an RCCL >= 2.30.5 install (include/nccl_device.h, lib/librccl.so)}"
M2N_ROCM_INDEX=${M2N_ROCM_INDEX:-https://repo.amd.com/rocm/whl/gfx94X-dcgpu}
M2N_GPU_ARCHS=${M2N_GPU_ARCHS:-gfx90a;gfx942;gfx950}
V=${M2N_ROCM_VENV:-/opt/rocm-venv}
B=${M2N_BUILDDIR:-/tmp/m2n-manylinux}

command -v perl > /dev/null || dnf install -y -q perl   # hipify-perl
/opt/python/cp312-cp312/bin/python -m venv $V
PRE=--pre; [[ $M2N_ROCM_INDEX == https://repo.amd.com/* ]] && PRE=   # GA index: skip its rc entries
$V/bin/pip install -q --index-url $M2N_ROCM_INDEX $PRE "rocm[devel]${M2N_ROCM_DEVEL_VERSION:+==$M2N_ROCM_DEVEL_VERSION}"
echo "rocm[devel]: $($V/bin/python -c 'import importlib.metadata as m; print(m.version("rocm-sdk-devel"))') from $M2N_ROCM_INDEX"
$V/bin/rocm-sdk init
ROCM=$($V/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/_rocm_sdk_devel
# TheRock clang selects the base GCC 8, which has no libstdc++ to link against; use the image's gcc-toolset
# (its libstdc++_nonshared keeps the GLIBCXX requirement at the manylinux_2_28 level).
GT=$(dirname "$(dirname "$(command -v gcc)")")
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-toolchain=$GT" HIPCC_LINK_FLAGS_APPEND="--gcc-toolchain=$GT"
make -C $P/contrib/nccl_m2n -f Makefile.rocm -j$(nproc) ROCM_PATH=$ROCM NCCL_HOME=$M2N_RCCL_HOME \
  RCCL_LIBDIR=$M2N_RCCL_HOME/lib "OFFLOAD_ARCH=$M2N_GPU_ARCHS" BUILDDIR=$B lib

L=$P/bindings/nccl_m2n/nccl/m2n/lib/rocm
mkdir -p $L $P/bindings/nccl_m2n/nccl/m2n/include
cp $B/lib/libnccl_m2n.so $B/lib/gpu_archs.txt $L/
patchelf --set-rpath '$ORIGIN' $L/libnccl_m2n.so
cp $B/include/nccl_m2n.h $P/bindings/nccl_m2n/nccl/m2n/include/
echo "libnccl_m2n.so: $(cat $L/gpu_archs.txt), max $(objdump -T $L/libnccl_m2n.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
