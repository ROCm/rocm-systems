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
set(RCCL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build")
set(RCCL_HIPIFY_DIR "${RCCL_BUILD_DIR}/hipify")

# --- Detect hipcc for compiling real RCCL source wrappers ---
# The RCCL build hipifies sources (cuda.h -> hip/hip_runtime.h) into build/hipify/.
# We compile those with hipcc --offload-host-only: no stubs needed, resilient to NCCL syncs.
if(NOT ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()
find_program(HIPCC_EXECUTABLE hipcc
  HINTS ${ROCM_PATH}
  PATH_SUFFIXES bin
)

if(HIPCC_EXECUTABLE AND EXISTS "${RCCL_HIPIFY_DIR}/src/graph/search.cc")
  message(STATUS "hipcc found: ${HIPCC_EXECUTABLE}")
  message(STATUS "Hipified RCCL sources: ${RCCL_HIPIFY_DIR}")
  set(HAVE_HIPCC_WRAPPERS ON)
else()
  if(NOT HIPCC_EXECUTABLE)
    message(WARNING "hipcc not found — real-source wrapper tests disabled")
  else()
    message(WARNING "Hipified sources not found at ${RCCL_HIPIFY_DIR} — run the RCCL build first")
  endif()
  set(HAVE_HIPCC_WRAPPERS OFF)
endif()

if(HAVE_HIPCC_WRAPPERS)
  # Detect GCC install dir so hipcc can find C++ standard headers
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.a
    OUTPUT_VARIABLE _gcc_libstdcpp
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  get_filename_component(GCC_INSTALL_DIR "${_gcc_libstdcpp}" DIRECTORY)

  set(HIPCC_COMPILE_FLAGS
    --offload-host-only
    -std=c++17
    --gcc-install-dir=${GCC_INSTALL_DIR}
    -D__HIP_PLATFORM_AMD__
    -DNCCL_OS_LINUX
    -Wno-tautological-constant-out-of-range-compare
  )
  set(HIPCC_INCLUDE_FLAGS
    -I${RCCL_HIPIFY_DIR}/src/include
    -I${RCCL_HIPIFY_DIR}/src/include/plugin
    -I${RCCL_HIPIFY_DIR}/src/include/plugin/tuner
    -I${RCCL_BUILD_DIR}/include
    -I${RCCL_HIPIFY_DIR}/src
    -I${RCCL_HIPIFY_DIR}/src/graph
    -I${RCCL_TEST_DIR}
    -I${RCCL_BUILD_DIR}/_deps/fmt-src/include
  )

  # Compile each wrapper file with hipcc (custom commands)
  foreach(wrapper paths_wrapper search_wrapper NullParentTests_wrapper)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${wrapper}.o
      COMMAND ${HIPCC_EXECUTABLE} ${HIPCC_COMPILE_FLAGS} ${HIPCC_INCLUDE_FLAGS}
              -c ${CMAKE_CURRENT_SOURCE_DIR}/${wrapper}.cc
              -o ${CMAKE_CURRENT_BINARY_DIR}/${wrapper}.o
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${wrapper}.cc
      COMMENT "hipcc (host-only): ${wrapper}.cc"
    )
  endforeach()

  # wrapper_link_stubs.cpp is pure C++ — compiled with the regular g++ compiler
  set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/paths_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/search_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/NullParentTests_wrapper.o
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE
  )
  add_library(rccl-real-source-wrappers STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/paths_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/search_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/NullParentTests_wrapper.o
    wrapper_link_stubs.cpp
  )
  target_compile_definitions(rccl-real-source-wrappers PRIVATE
    __HIP_PLATFORM_AMD__
    NCCL_OS_LINUX
  )
  target_link_libraries(rccl-real-source-wrappers PRIVATE
    GTest::GTest
    fmt::fmt-header-only
  )
endif()

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
)
if(HAVE_HIPCC_WRAPPERS)
  target_link_libraries(rccl-PureUnitTests PRIVATE
    -Wl,--whole-archive
    rccl-real-source-wrappers
    -Wl,--no-whole-archive
  )
endif()
