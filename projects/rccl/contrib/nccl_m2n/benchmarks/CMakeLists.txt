# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# benchmarks/CMakeLists.txt
#
# Picked up by the root CMakeLists.txt when -DNCCL_M2N_BUILD_BENCH=ON.
# Builds three MPI-linked bench binaries:
#   reshard_bench                       (mpi + MPI bootstrap)
#   reshard_batch_bench_user_window     (mpi)
#   reshard_model_bench                 (mpi)
#
# Mirrors benchmarks/Makefile rules. All three link the in-tree
# nccl_m2n_shared (which itself pulls in nccl + cudart via INTERFACE).

find_package(MPI REQUIRED)

set(BENCH_KERNELS_SRC bench_common_kernels.cu)

# reshard_bench  ─ uses bench_common_kernels for byte-pattern init/validate
add_executable(reshard_bench
    reshard_bench.cc
    ${BENCH_KERNELS_SRC}
)

# reshard_batch_bench_user_window ─ self-contained kernel defs in the .cu
add_executable(reshard_batch_bench_user_window
    reshard_batch_bench_user_window.cu
)

# reshard_model_bench ─ uses bench_common_kernels too
add_executable(reshard_model_bench
    reshard_model_bench.cu
    ${BENCH_KERNELS_SRC}
)

set(BENCH_TARGETS reshard_bench reshard_batch_bench_user_window reshard_model_bench)

set_target_properties(${BENCH_TARGETS} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}"
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    # Find libnccl_m2n.so relative to build/bin → build/lib at runtime so
    # in-tree runs don't need LD_LIBRARY_PATH (matches Makefile's
    # M2N_CLIENT_LDFLAGS '$ORIGIN/../lib').
    BUILD_RPATH "${NCCL_M2N_LIBRARY_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib"
)

foreach(t IN LISTS BENCH_TARGETS)
  target_include_directories(${t} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${NCCL_M2N_SOURCE_DIR}/src
      ${NCCL_M2N_INCLUDE_DIR}
  )
  target_include_directories(${t} SYSTEM PRIVATE
      ${NCCL_M2N_SOURCE_DIR}/third_party
      ${_NCCL_M2N_NCCL_INCLUDES}
      ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_compile_options(${t} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  target_link_libraries(${t} PRIVATE
      NCCL::m2n
      MPI::MPI_CXX
  )
endforeach()
