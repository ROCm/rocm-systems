#!/bin/bash
# Build RCCL (gfx950), nccl_m2n lib + tests, benches, rccl-tests and the AUTO shim from $SRC.
# Run inside the container.
# Usage: bash build.sh [rccl|m2n|bench|extras|all]    (all = rccl + m2n + bench + extras)
set -eo pipefail
D=$(dirname "$(readlink -f "$0")")
. $D/env.sh
WHAT=${1:-all}
B=${RCCL_BUILD:-/tmp/m2n-rccl-build-$(id -u)}
echo "SRC=$SRC HEAD=$(git -C $SRC rev-parse --short HEAD) ROCM_PATH=$ROCM_PATH"
want() { [[ $WHAT == $1 || $WHAT == all ]]; }

if want rccl; then
  cmake -S $SRC/projects/rccl -B $B -G "$CMAKE_GENERATOR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/amdclang++ -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/amdclang \
    -DROCM_PATH=$ROCM_PATH -DCMAKE_PREFIX_PATH=$ROCM_PATH \
    -DGPU_TARGETS=gfx950 -DAMDGPU_TARGETS=gfx950 -DBUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=$RCCL_INSTALL > $WORK/logs/build_rccl_cmake.log 2>&1
  cmake --build $B -j ${JOBS:-128} > $WORK/logs/build_rccl.log 2>&1 || { tail -30 $WORK/logs/build_rccl.log; exit 1; }
  cmake --install $B > /dev/null
  ls -l $RCCL_INSTALL/lib/librccl.so
fi

MK=(make -f Makefile.rocm -j32 ROCM_PATH=$ROCM_PATH NCCL_HOME=$RCCL_INSTALL RCCL_LIBDIR=$RCCL_INSTALL/lib
    GTEST_DIR=${GTEST_DIR:-$(dirname "$SRC")/googletest/googletest} MPI_HOME=$MPI_HOME OFFLOAD_ARCH=gfx950 BUILDDIR=$WORK/mk)
M2N_DIR=$SRC/projects/rccl/contrib/nccl_m2n

if want m2n; then
  cd $M2N_DIR
  for t in lib tests; do
    "${MK[@]}" $t > $WORK/logs/build_m2n_$t.log 2>&1 || { tail -30 $WORK/logs/build_m2n_$t.log; exit 1; }
  done
  ls -l $BIN/basic_api_test_local $BIN/basic_api_test_mpi
fi

if want bench; then
  cd $M2N_DIR
  "${MK[@]}" reshard reshard_batch_user_window reshard_model > $WORK/logs/build_bench.log 2>&1 || { tail -30 $WORK/logs/build_bench.log; exit 1; }
  ls -l $BIN/reshard_bench $BIN/reshard_batch_bench_user_window $BIN/reshard_model_bench
  ldd $BIN/reshard_bench | grep librccl
fi

if want extras; then
  # rccl-tests for the alltoall reference (links the base-image librccl, runs ours via LD_LIBRARY_PATH).
  [[ -d $WORK/rccl-tests ]] || git clone -q --depth 1 https://github.com/ROCm/rccl-tests.git $WORK/rccl-tests
  make -C $WORK/rccl-tests -j64 MPI=1 MPI_HOME=$MPI_HOME HIP_HOME=$ROCM_PATH NCCL_HOME=$RCCL_INSTALL \
    GPU_TARGETS=gfx950 > $WORK/logs/build_rccl_tests.log 2>&1 || { tail -30 $WORK/logs/build_rccl_tests.log; exit 1; }
  ls -l $WORK/rccl-tests/build/alltoall_perf
fi
