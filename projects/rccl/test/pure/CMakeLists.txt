cmake_minimum_required(VERSION 3.16)
project(rccl-PureUnitTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
include(FetchContent)

# --- Resolve GTest: system → RCCL-vendored → FetchContent fallback ---
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  # Check RCCL's vendored GTest (built by the main RCCL CMake into build/gtest/)
  set(_rccl_gtest_root "${CMAKE_CURRENT_SOURCE_DIR}/../../build/gtest")
  if(EXISTS "${_rccl_gtest_root}/lib/libgtest.a" OR EXISTS "${_rccl_gtest_root}/lib64/libgtest.a")
    message(STATUS "Using RCCL-vendored GTest: ${_rccl_gtest_root}")
    find_package(GTest REQUIRED PATHS "${_rccl_gtest_root}" NO_DEFAULT_PATH)
  else()
    message(STATUS "System GTest not found, RCCL-vendored GTest not found — downloading via FetchContent")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
  endif()
endif()

# Fetch fmt (header-only) for VersionInfoTests
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.2.1
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(fmt)

set(RCCL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
set(RCCL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# --- Resolve RCCL build directory ---
# The RCCL build may use a flat layout (build/hipify/) or a typed layout
# (build/release/hipify/, build/debug/hipify/) depending on build system.
# Accept -DRCCL_BUILD_DIR=... to override; otherwise auto-detect.
set(RCCL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build" CACHE PATH
    "RCCL build directory (auto-detects release/debug subdirs)")

if(NOT EXISTS "${RCCL_BUILD_DIR}/hipify")
  foreach(_subdir release debug Release Debug RelWithDebInfo)
    if(EXISTS "${RCCL_BUILD_DIR}/${_subdir}/hipify")
      set(RCCL_BUILD_DIR "${RCCL_BUILD_DIR}/${_subdir}")
      message(STATUS "Detected build subdir layout: ${RCCL_BUILD_DIR}")
      break()
    endif()
  endforeach()
endif()

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

  # Resolve GTest include dir for hipcc custom commands.
  # A target's INTERFACE_INCLUDE_DIRECTORIES may contain $<BUILD_INTERFACE:...>
  # genexes and ;-separated lists; splicing that into the hipcc command line as
  # -I${var} corrupts the command. Resolve to a plain filesystem path instead.
  set(_gtest_inc "")
  if(DEFINED googletest_SOURCE_DIR AND EXISTS "${googletest_SOURCE_DIR}/googletest/include")
    set(_gtest_inc "${googletest_SOURCE_DIR}/googletest/include")
  elseif(GTEST_INCLUDE_DIRS AND EXISTS "${GTEST_INCLUDE_DIRS}")
    set(_gtest_inc "${GTEST_INCLUDE_DIRS}")
  elseif(TARGET GTest::gtest)
    get_target_property(_gtest_inc GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
    string(GENEX_STRIP "${_gtest_inc}" _gtest_inc)
    if(_gtest_inc)
      list(GET _gtest_inc 0 _gtest_inc)
    endif()
  endif()
  if(NOT _gtest_inc OR NOT EXISTS "${_gtest_inc}")
    set(_gtest_inc "")
  endif()

  # Resolve fmt include dir for hipcc (may be FetchContent or RCCL-vendored)
  set(_fmt_inc "")
  if(EXISTS "${RCCL_BUILD_DIR}/_deps/fmt-src/include")
    set(_fmt_inc "${RCCL_BUILD_DIR}/_deps/fmt-src/include")
  elseif(DEFINED fmt_SOURCE_DIR AND EXISTS "${fmt_SOURCE_DIR}/include")
    set(_fmt_inc "${fmt_SOURCE_DIR}/include")
  else()
    get_target_property(_fmt_inc fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
    string(GENEX_STRIP "${_fmt_inc}" _fmt_inc)
    if(_fmt_inc)
      list(GET _fmt_inc 0 _fmt_inc)
    endif()
    if(NOT _fmt_inc OR NOT EXISTS "${_fmt_inc}")
      set(_fmt_inc "")
    endif()
  endif()

  set(HIPCC_COMPILE_FLAGS
    --offload-host-only
    -std=c++17
    -fPIC
    --gcc-install-dir=${GCC_INSTALL_DIR}
    -D__HIP_PLATFORM_AMD__
    -DNCCL_OS_LINUX
    -DROCM_VERSION=70000
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
    -I${RCCL_TEST_DIR}/common
  )
  if(_fmt_inc)
    list(APPEND HIPCC_INCLUDE_FLAGS -I${_fmt_inc})
  endif()
  if(_gtest_inc)
    list(APPEND HIPCC_INCLUDE_FLAGS -I${_gtest_inc})
  endif()

  # Compile each wrapper file with hipcc (custom commands)
  foreach(wrapper paths_wrapper search_wrapper NullParentTests_wrapper
                  mem_manager_src_wrapper mem_manager_test_wrapper)
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
    ${CMAKE_CURRENT_BINARY_DIR}/mem_manager_src_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/mem_manager_test_wrapper.o
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE
  )
  add_library(rccl-real-source-wrappers STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/paths_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/search_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/NullParentTests_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/mem_manager_src_wrapper.o
    ${CMAKE_CURRENT_BINARY_DIR}/mem_manager_test_wrapper.o
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
  ${RCCL_BUILD_DIR}/include
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
