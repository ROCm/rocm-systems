# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# tests/CMakeLists.txt
#
# Picked up by the root CMakeLists.txt when -DNCCL_M2N_BUILD_TESTS=ON.
# Builds the basic_api correctness suite:
#   basic_api_test_mpi    — MPI bootstrap, links libnccl_m2n.a
#   basic_api_test_local  — pthread (no MPI), compiles test-enabled library sources
#
# An optional extra test target in a tests/ci/ subdirectory is included via
# add_subdirectory(ci) when that directory is present.
#
# GTest discovery, in order:
#   1) Existing GTest::gtest or gtest target from the parent build.
#   2) Installed GTest package.
#   3) GTEST_DIR CMake/environment override, the repository-root third_party/googletest
#      submodule, or a vendored tests/googletest source.
#   4) Skip the test build with a clear message.
#
# MPI is required for basic_api_test_mpi; basic_api_test_local builds even
# when MPI is absent.

find_package(Threads REQUIRED)

# Multi-dimensional coverage runs without GTest. The geometry executable
# exercises host planning only; launch the GPU executable under MPI.
add_executable(multi_dim_geometry_test multi_dim_geometry_test.cc)
target_include_directories(multi_dim_geometry_test PRIVATE "${NCCL_M2N_SOURCE_DIR}/src")
target_link_libraries(multi_dim_geometry_test PRIVATE NCCL::m2n_static Threads::Threads)
set_target_properties(multi_dim_geometry_test PROPERTIES CXX_STANDARD 17
    RUNTIME_OUTPUT_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}")
add_test(NAME multi_dim_geometry_test COMMAND multi_dim_geometry_test)
find_package(MPI QUIET)
if(MPI_FOUND)
  add_executable(multi_dim_test_mpi multi_dim_test.cc)
  target_compile_definitions(multi_dim_test_mpi PRIVATE OMPI_SKIP_MPICXX=1)
  target_link_libraries(multi_dim_test_mpi PRIVATE NCCL::m2n_static MPI::MPI_CXX Threads::Threads)
  set_target_properties(multi_dim_test_mpi PROPERTIES CXX_STANDARD 17
      RUNTIME_OUTPUT_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}")
endif()
# Always available, including when GTest is absent. This target tests the
# host-side namespace/resource contract without CUDA/NCCL runtime calls.
add_executable(pipe_commb_resources_test pipe_commb_resources_test.cc)
target_include_directories(pipe_commb_resources_test PRIVATE "${NCCL_M2N_SOURCE_DIR}/src")
set_target_properties(pipe_commb_resources_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME pipe_commb_resources_test COMMAND pipe_commb_resources_test)

# ─── GTest ────────────────────────────────────────────────────────────────────
# Reuse a parent/package target when possible. The test binaries define their
# own main(), so only gtest is required. The source fallback compiles
# gtest-all.cc exactly once and uses an M2N-local target name.

set(_NCCL_M2N_GTEST_TARGET "")
if(TARGET GTest::gtest)
  set(_NCCL_M2N_GTEST_TARGET GTest::gtest)
elseif(TARGET gtest)
  set(_NCCL_M2N_GTEST_TARGET gtest)
else()
  find_package(GTest QUIET)
  if(TARGET GTest::gtest)
    set(_NCCL_M2N_GTEST_TARGET GTest::gtest)
  endif()
endif()

set(SRCS ${NCCL_M2N_SRC_FULL_PATHS})

# Match Make's environment-variable support without overriding an explicit
# -DGTEST_DIR value, including an intentionally empty one.
if(NOT DEFINED GTEST_DIR AND DEFINED ENV{GTEST_DIR})
  set(GTEST_DIR "$ENV{GTEST_DIR}")
endif()

# The upstream googletest repo nests the framework one level down, so the
# usable source root of the repository-root submodule is
# third_party/googletest/googletest.
get_filename_component(_NCCL_M2N_GTEST_SUBMODULE
    "${NCCL_M2N_SOURCE_DIR}/../third_party/googletest/googletest" ABSOLUTE)

set(_NCCL_M2N_GTEST_DIR "")
if(NOT _NCCL_M2N_GTEST_TARGET AND GTEST_DIR AND
   EXISTS "${GTEST_DIR}/include/gtest/gtest.h")
  set(_NCCL_M2N_GTEST_DIR "${GTEST_DIR}")
elseif(NOT _NCCL_M2N_GTEST_TARGET AND
       EXISTS "${_NCCL_M2N_GTEST_SUBMODULE}/include/gtest/gtest.h")
  set(_NCCL_M2N_GTEST_DIR "${_NCCL_M2N_GTEST_SUBMODULE}")
elseif(NOT _NCCL_M2N_GTEST_TARGET AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/include/gtest/gtest.h")
  set(_NCCL_M2N_GTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
endif()

if(NOT _NCCL_M2N_GTEST_TARGET AND NOT _NCCL_M2N_GTEST_DIR)
  message(STATUS
    "nccl_m2n tests/: googletest not found — skipping basic_api. "
    "Provide GTest::gtest, install GTest, export GTEST_DIR, run "
    "'git submodule update --init third_party/googletest', or pass "
    "-DGTEST_DIR=/path/to/googletest.")
  return()
endif()

if(NOT _NCCL_M2N_GTEST_TARGET)
  message(STATUS "nccl_m2n tests/: using googletest source ${_NCCL_M2N_GTEST_DIR}")
  add_library(nccl_m2n_gtest STATIC
      "${_NCCL_M2N_GTEST_DIR}/src/gtest-all.cc")
  target_include_directories(nccl_m2n_gtest
      PUBLIC "${_NCCL_M2N_GTEST_DIR}/include"
      PRIVATE "${_NCCL_M2N_GTEST_DIR}")
  target_compile_options(nccl_m2n_gtest PRIVATE -Wno-uninitialized)
  set_target_properties(nccl_m2n_gtest PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON)
  set(_NCCL_M2N_GTEST_TARGET nccl_m2n_gtest)
else()
  message(STATUS "nccl_m2n tests/: using googletest target ${_NCCL_M2N_GTEST_TARGET}")
endif()

# ─── MPI (optional) ───────────────────────────────────────────────────────────
find_package(MPI)

# ─── basic_api_test_local (no MPI) ────────────────────────────────────────────
add_executable(basic_api_test_local
    nonblocking_comm_test.cc
    basic_api_test_local.cc
    pack_staging_pool_test.cc
    devcomm_cache_test.cc
    test_helpers.cu
    ${SRCS}
)
target_include_directories(basic_api_test_local PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NCCL_M2N_SOURCE_DIR}/src
    ${NCCL_M2N_INCLUDE_DIR}
)
target_include_directories(basic_api_test_local SYSTEM PRIVATE
    ${NCCL_M2N_SOURCE_DIR}/third_party
    ${_NCCL_M2N_NCCL_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(basic_api_test_local PRIVATE
    NCCL_OS_LINUX
    NCCL_M2N_TESTING
)
target_compile_options(basic_api_test_local PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(basic_api_test_local PRIVATE
    ${_NCCL_M2N_NCCL_TARGET}
    CUDA::cudart
    CUDA::cuda_driver
    ${_NCCL_M2N_GTEST_TARGET}
    Threads::Threads
)
if(_NCCL_M2N_NCCL_HEADER_TARGET)
  add_dependencies(basic_api_test_local ${_NCCL_M2N_NCCL_HEADER_TARGET})
endif()

# ─── basic_api_test_mpi (only when MPI present) ───────────────────────────────
if(MPI_FOUND)
  add_executable(basic_api_test_mpi
      basic_api_test_mpi.cc
      pipe_graph_path_test.cc
      test_helpers.cu
  )
  target_include_directories(basic_api_test_mpi PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${NCCL_M2N_SOURCE_DIR}/src
      ${NCCL_M2N_INCLUDE_DIR}
  )
  target_include_directories(basic_api_test_mpi SYSTEM PRIVATE
      ${NCCL_M2N_SOURCE_DIR}/third_party
      ${_NCCL_M2N_NCCL_INCLUDES}
      ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(basic_api_test_mpi PRIVATE
      NCCL::m2n_static
      ${_NCCL_M2N_GTEST_TARGET}
      MPI::MPI_CXX
      Threads::Threads
  )
else()
  message(STATUS "nccl_m2n tests/: MPI not found — skipping basic_api_test_mpi.")
endif()

set_target_properties(basic_api_test_local PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}"
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    BUILD_RPATH "${NCCL_M2N_LIBRARY_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib"
)

if(MPI_FOUND)
  set_target_properties(basic_api_test_mpi PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}"
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      BUILD_RPATH "${NCCL_M2N_LIBRARY_DIR}"
      INSTALL_RPATH "$ORIGIN/../lib"
  )
endif()

# CTest registration (`cmake --build build && ctest --test-dir build`).
add_test(NAME basic_api_test_local
         COMMAND basic_api_test_local
         WORKING_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}")
add_test(NAME basic_api_pack_local
         COMMAND basic_api_test_local --api default --copy-algorithm pack
         WORKING_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}")
# basic_api_test_mpi is launched under mpirun, not registered as a plain
# ctest entry.

# ─── Optional extra tests ─────────────────────────────────────────────────────
# Built when a tests/ci/ subdirectory is present; harmless when it is absent.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ci/CMakeLists.txt")
  add_subdirectory(ci)
endif()
