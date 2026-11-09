#!/bin/bash
# Build RCCL, nccl_m2n lib + tests, benches, and the nccl4py + nccl.m2n wheels from $SRC for M2N_GPU_ARCHS.
# Run inside the container as yourself. Usage: bash build.sh [rccl|m2n|bench|py|all]
set -eo pipefail
D=$(dirname "$(readlink -f "$0")")
WHAT=${1:-all}
want() { [[ $WHAT == $1 || $WHAT == all ]]; }
# nccl_m2n staging kernels need wave64 (GFX9): gfx1xxx is rejected at compile time. Not GPU_ARCHS:
# the Primus image sets that to "native".
M2N_GPU_ARCHS=${M2N_GPU_ARCHS:-gfx90a;gfx942;gfx950}

if [[ $WHAT != py ]]; then
  . $D/env.sh
  echo "SRC=$SRC HEAD=$(git -C $SRC rev-parse --short HEAD) ROCM_PATH=$ROCM_PATH"
fi

if want rccl; then
  B=${RCCL_BUILD:-/tmp/m2n-rccl-build-$(id -u)}
  cmake -S $SRC/projects/rccl -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/amdclang++ -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/amdclang \
    -DROCM_PATH=$ROCM_PATH -DCMAKE_PREFIX_PATH=$ROCM_PATH \
    "-DGPU_TARGETS=$M2N_GPU_ARCHS" "-DAMDGPU_TARGETS=$M2N_GPU_ARCHS" -DBUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=$RCCL_INSTALL > $WORK/logs/build_rccl_cmake.log 2>&1
  cmake --build $B -j ${JOBS:-128} > $WORK/logs/build_rccl.log 2>&1 || { tail -30 $WORK/logs/build_rccl.log; exit 1; }
  cmake --install $B > /dev/null
  ls -l $RCCL_INSTALL/lib/librccl.so
fi

if want m2n || want bench; then
  MK=(make -f Makefile.rocm -j32 ROCM_PATH=$ROCM_PATH NCCL_HOME=$RCCL_INSTALL RCCL_LIBDIR=$RCCL_INSTALL/lib
      GTEST_DIR=${GTEST_DIR:-$(dirname "$SRC")/googletest/googletest} MPI_HOME=/opt/openmpi "OFFLOAD_ARCH=$M2N_GPU_ARCHS"
      BUILDDIR=$WORK/mk)
  cd $SRC/projects/rccl/contrib/nccl_m2n
  if want m2n; then
    for t in lib tests; do
      "${MK[@]}" $t > $WORK/logs/build_m2n_$t.log 2>&1 || { tail -30 $WORK/logs/build_m2n_$t.log; exit 1; }
    done
    ls -l $BIN/basic_api_test_local $BIN/basic_api_test_mpi
  fi
  if want bench; then
    "${MK[@]}" reshard reshard_model > $WORK/logs/build_bench.log 2>&1 || { tail -30 $WORK/logs/build_bench.log; exit 1; }
    ls -l $BIN/reshard_bench $BIN/reshard_model_bench
  fi
fi

if want py; then
  # Wheels into $WORK/whl; install them as root afterwards (README).
  . $D/py_env.sh
  W=$WORK/whl; T=$WORK/py-build   # staged sources
  rm -rf $T $W && mkdir -p $T $W
  cp -r --no-preserve=all $SRC/projects/rccl/bindings/nccl4py $T/nccl4py
  cp -r --no-preserve=all $M2N_PY $T/nccl_m2n
  (cd $T/nccl4py && SETUPTOOLS_SCM_PRETEND_VERSION=${NCCL4PY_VERSION:-0.2.0} \
    python -m pip wheel -q --no-build-isolation --no-deps -w $W .)
  # The nccl.m2n build needs the port importable (setup.py checks nccl._hip_compat).
  python -m pip install -q --no-deps --target $T/site $W/nccl4py-*.whl
  # Ship libnccl_m2n.so (from the m2n step), its GPU arch list and header in the wheel; drop the
  # build-tree RUNPATH (librccl comes from the user's ROCm).
  L=$T/nccl_m2n/nccl/m2n/lib/rocm
  mkdir -p $L $T/nccl_m2n/nccl/m2n/include
  cp $WORK/mk/lib/libnccl_m2n.so $WORK/mk/lib/gpu_archs.txt $L/
  patchelf --set-rpath '$ORIGIN' $L/libnccl_m2n.so
  cp $WORK/mk/include/nccl_m2n.h $T/nccl_m2n/nccl/m2n/include/
  (cd $T/nccl_m2n && PYTHONPATH=$T/site NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1 \
    python -m pip wheel -q --no-build-isolation --no-deps -w $W .)
  python -m zipfile -l $W/nccl_extensions-*.whl | grep -q 'nccl/m2n/lib/rocm/libnccl_m2n.so' \
    || { echo "libnccl_m2n.so missing from the wheel"; exit 1; }
  # Local wheel, ROCm runtime excluded; the tag follows the image's glibc. Release wheel: manylinux_wheel.sh.
  if [[ ${REPAIR:-1} == 1 ]]; then
    auditwheel repair -w $W/repaired $W/nccl_extensions-*.whl --plat manylinux_2_38_x86_64 --exclude librccl.so.1 --exclude libamdhip64.so.7 \
      --exclude libhsa-runtime64.so.1 --exclude libamd_comgr.so.3 --exclude librocprofiler-register.so.0
    ls $W/repaired
  fi
  ls $W
fi
