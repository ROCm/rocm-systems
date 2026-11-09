#!/bin/bash
# Release wheels of nccl.m2n and the nccl4py port: cibuildwheel in manylinux_2_28 builds libnccl_m2n.so
# (manylinux_lib.sh) and the bindings, then auditwheel tags them manylinux_2_28 ([tool.cibuildwheel] in
# bindings/nccl_m2n/pyproject.toml).
# Run on a host with Docker and cibuildwheel. Usage: [M2N_RCCL_HOME=...] bash manylinux_wheel.sh [output dir]
#   M2N_RCCL_HOME           RCCL >= 2.30.5 install to build against (default $WORK/rccl-install from build.sh rccl)
#   M2N_ROCM_INDEX          TheRock pip index for rocm[devel], the compiler (default https://repo.amd.com/rocm/whl/gfx94X-dcgpu, GA)
#   M2N_ROCM_DEVEL_VERSION  pin rocm[devel] (default: the latest in M2N_ROCM_INDEX)
#   M2N_GPU_ARCHS, CIBW_BUILD (default cp310-* ... cp314-*, as pyproject.toml)
set -eo pipefail
D=$(dirname "$(readlink -f "$0")")
WORK=${WORK:-$HOME/m2n-ruby}
export M2N_RCCL_HOME=$(readlink -m "${M2N_RCCL_HOME:-$WORK/rccl-install}")
[[ -f $M2N_RCCL_HOME/include/nccl_device.h ]] || { echo "M2N_RCCL_HOME=$M2N_RCCL_HOME is not an RCCL install" >&2; exit 1; }
OUT=$(readlink -m "${1:-$WORK/wheelhouse}")
export CIBW_BUILD=${CIBW_BUILD:-cp310-* cp311-* cp312-* cp313-* cp314-*}
# The RCCL install is mounted at the same path; host network because the default bridge may lack DNS.
export CIBW_CONTAINER_ENGINE="docker; create_args: --network=host -v $M2N_RCCL_HOME:$M2N_RCCL_HOME:ro"
# cibuildwheel copies its {project} into the container: stage only what the build needs, in the projects/rccl layout.
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
P=$D/../../..   # projects/rccl
tar -C "$P" --exclude=build --exclude=__pycache__ --exclude='*.egg-info' -cf - bindings/nccl4py bindings/nccl_m2n contrib/nccl_m2n |
  tar -C "$S" -xf -
cd "$S"
${CIBUILDWHEEL:-cibuildwheel} --platform linux --output-dir "$OUT" bindings/nccl_m2n
# The RCCL nccl4py port nccl.m2n depends on, for the same Python versions.
CIBW_ENVIRONMENT="SETUPTOOLS_SCM_PRETEND_VERSION=${NCCL4PY_VERSION:-0.2.0}" CIBW_SKIP="*-musllinux_*" CIBW_ARCHS=x86_64 \
  CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_28 ${CIBUILDWHEEL:-cibuildwheel} --platform linux --output-dir "$OUT" bindings/nccl4py
ls "$OUT"
