cmake_minimum_required(VERSION 3.16)
project(rccl-PureUnitTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use system GTest
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Fetch fmt (header-only) for VersionInfoTests
include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.2.1
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(fmt)

set(RCCL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
set(RCCL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# --- Real RCCL source wrappers (compiled with -isystem stubs/) ---
# These compile real RCCL .cc files (paths.cc, search.cc) against real RCCL
# headers, with only HIP/CUDA/HSA system headers stubbed out.
add_library(rccl-real-source-wrappers STATIC
  paths_wrapper.cc
  search_wrapper.cc
  NullParentTests_wrapper.cc
  wrapper_link_stubs.cpp
)

# System-include stubs FIRST — intercepts <hip/hip_runtime.h>, <cuda.h>, etc.
# but NOT quoted RCCL includes like "comm.h", "core.h", "topo.h".
target_include_directories(rccl-real-source-wrappers SYSTEM BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/stubs
)
target_include_directories(rccl-real-source-wrappers PRIVATE
  ${RCCL_SRC_DIR}/include
  ${RCCL_SRC_DIR}/include/plugin
  ${RCCL_SRC_DIR}/include/plugin/tuner
  ${RCCL_SRC_DIR}/../build/include
  ${RCCL_SRC_DIR}
  ${RCCL_SRC_DIR}/graph
  ${RCCL_TEST_DIR}
)
target_compile_definitions(rccl-real-source-wrappers PRIVATE
  ROCM_VERSION=0
  __HIP_PLATFORM_AMD__
  RCCL_EXPOSE_STATIC
  NCCL_OS_LINUX
)
target_compile_options(rccl-real-source-wrappers PRIVATE -include cstdint)
target_link_libraries(rccl-real-source-wrappers PRIVATE
  GTest::GTest
  fmt::fmt-header-only
)

# --- Main test executable ---
add_executable(rccl-PureUnitTests
  main.cpp
  rccl_stubs.cpp
  ${RCCL_TEST_DIR}/BitOpsTests.cpp
  ${RCCL_TEST_DIR}/IommuPassthrough_test.cpp
  ${RCCL_TEST_DIR}/VersionInfoTests.cpp
  ${RCCL_TEST_DIR}/DdaCollCommonTests.cpp
  ${RCCL_TEST_DIR}/RomeTopoConsensusTests.cpp
  ${RCCL_TEST_DIR}/MiscTests.cpp
  ${RCCL_TEST_DIR}/TimeoutTests.cpp
  ${RCCL_TEST_DIR}/EnqueueCountTests.cpp
  ${RCCL_TEST_DIR}/BootstrapBidirTests.cpp
  ${RCCL_TEST_DIR}/AltRsmiTests.cpp
  ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
  bootstrap_bidir_stub.cpp
  ${RCCL_SRC_DIR}/misc/kernel_config.cc
  ${RCCL_SRC_DIR}/misc/alt_rsmi.cc
  ${RCCL_SRC_DIR}/graph/rome_topo_consensus.cc
)

# -iquote stubs/ is searched BEFORE the including file's directory for
# quoted includes, so our stubs shadow real headers even when a real
# header (e.g. topo.h) does #include "comm.h".
target_compile_options(rccl-PureUnitTests PRIVATE
  -iquote ${CMAKE_CURRENT_SOURCE_DIR}/stubs
)
target_include_directories(rccl-PureUnitTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/stubs
  ${RCCL_SRC_DIR}/include
  ${RCCL_SRC_DIR}/include/plugin
  ${RCCL_SRC_DIR}/include/plugin/tuner
  ${RCCL_SRC_DIR}/../build/include
  ${RCCL_SRC_DIR}
  ${RCCL_SRC_DIR}/graph
  ${RCCL_TEST_DIR}
)

target_compile_definitions(rccl-PureUnitTests PRIVATE
  ROCM_VERSION=0
  __HIP_PLATFORM_AMD__
  RCCL_EXPOSE_STATIC
  ARSMI_TEST_BUILD
  NCCL_OS_LINUX
)

# Force-include <cstdint> so headers like alt_rsmi.h that use uint32_t
# without including <stdint.h>/<cstdint> compile correctly.
target_compile_options(rccl-PureUnitTests PRIVATE -include cstdint)

target_link_libraries(rccl-PureUnitTests PRIVATE
  GTest::GTest
  Threads::Threads
  fmt::fmt-header-only
  -Wl,--whole-archive
  rccl-real-source-wrappers
  -Wl,--no-whole-archive
)
