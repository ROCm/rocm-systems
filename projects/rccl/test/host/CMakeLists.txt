cmake_minimum_required(VERSION 3.16)

# --- hipcc is mandatory ---
# Auto-detect hipcc via ROCM_PATH, or pass -DCMAKE_CXX_COMPILER=hipcc.
if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
  if(NOT ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
      set(ROCM_PATH "/opt/rocm")
    endif()
  endif()
  find_program(_hipcc hipcc HINTS ${ROCM_PATH} PATH_SUFFIXES bin NO_CMAKE_FIND_ROOT_PATH)
  if(NOT _hipcc)
    find_program(_hipcc hipcc)
  endif()
  if(_hipcc)
    set(CMAKE_CXX_COMPILER "${_hipcc}")
  else()
    message(FATAL_ERROR
      "hipcc not found. Either:\n"
      "  cmake -B build -DCMAKE_CXX_COMPILER=/path/to/hipcc\n"
      "  cmake -B build -DROCM_PATH=/opt/rocm-6.x")
  endif()
endif()

# hipcc needs --gcc-install-dir to find C++ standard headers.
find_program(_system_gxx g++ REQUIRED)
execute_process(
  COMMAND ${_system_gxx} -print-file-name=libstdc++.a
  OUTPUT_VARIABLE _gcc_libstdcpp
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
get_filename_component(_gcc_install_dir "${_gcc_libstdcpp}" DIRECTORY)
set(CMAKE_CXX_FLAGS_INIT "--gcc-install-dir=${_gcc_install_dir} --offload-host-only")

project(rccl-HostUnitTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wno-tautological-constant-out-of-range-compare)

find_package(Threads REQUIRED)
include(FetchContent)

# --- Resolve GTest: system -> RCCL-vendored -> FetchContent fallback ---
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  set(_rccl_gtest_root "${CMAKE_CURRENT_SOURCE_DIR}/../../build/gtest")
  if(EXISTS "${_rccl_gtest_root}/lib/libgtest.a" OR EXISTS "${_rccl_gtest_root}/lib64/libgtest.a")
    message(STATUS "Using RCCL-vendored GTest: ${_rccl_gtest_root}")
    find_package(GTest REQUIRED PATHS "${_rccl_gtest_root}" NO_DEFAULT_PATH)
  else()
    message(STATUS "Downloading GTest via FetchContent")
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
  endif()
endif()

# --- Resolve fmt: system -> FetchContent fallback ---
find_package(fmt QUIET)
if(NOT fmt_FOUND)
  message(STATUS "Downloading fmt via FetchContent")
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(fmt)
endif()

# --- Resolve RCCL build directory ---
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
set(RCCL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

if(NOT EXISTS "${RCCL_HIPIFY_DIR}/src/graph/search.cc")
  message(FATAL_ERROR "Hipified sources not found at ${RCCL_HIPIFY_DIR}. Run the RCCL build first.")
endif()
message(STATUS "hipcc: ${CMAKE_CXX_COMPILER}")
message(STATUS "Hipified sources: ${RCCL_HIPIFY_DIR}")

# --- Detect whether the system amdsmi.h exposes the UALoE fabric struct ---
# Compile-only (STATIC_LIBRARY) avoids link failures from amdsmi symbols.
include(CheckCXXSourceCompiles)
unset(AMDSMI_FABRIC_API CACHE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles("
  #include <amd_smi/amdsmi.h>
  int main() {
    amdsmi_fabric_info_t info{};
    (void)info;
    return 0;
  }
" AMDSMI_FABRIC_API)
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
if(AMDSMI_FABRIC_API)
  message(STATUS "amdsmi fabric API found - enabling AMDSMI_FABRIC_DIRECT")
else()
  message(STATUS "amdsmi fabric API not found (AMDSMI_FABRIC_DIRECT disabled)")
endif()

# --- Common include paths for hipified RCCL headers ---
set(RCCL_INCLUDE_DIRS
  ${RCCL_HIPIFY_DIR}/src/include
  ${RCCL_HIPIFY_DIR}/src/include/plugin
  ${RCCL_HIPIFY_DIR}/src/include/plugin/tuner
  ${RCCL_BUILD_DIR}/include
  ${RCCL_HIPIFY_DIR}/src
  ${RCCL_HIPIFY_DIR}/src/graph
  ${RCCL_TEST_DIR}
  ${RCCL_TEST_DIR}/common
)

# --- Wrapper library for production .cc files that need link stubs ---
add_library(rccl-source-wrappers STATIC
  mem_manager_src_wrapper.cc
  mem_manager_wrapper_test.cc
  bootstrap_wrapper.cc
  init_stubs.cpp
  wrapper_link_stubs.cpp
)
target_include_directories(rccl-source-wrappers PRIVATE ${RCCL_INCLUDE_DIRS})
target_compile_definitions(rccl-source-wrappers PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
)
target_link_libraries(rccl-source-wrappers PRIVATE
  GTest::GTest
  fmt::fmt-header-only
)

# --- Main test executable ---
add_executable(rccl-HostUnitTests
  main.cpp
  ${RCCL_TEST_DIR}/BitOpsTests.cpp
  ${RCCL_TEST_DIR}/IommuPassthrough_test.cpp
  ${RCCL_TEST_DIR}/VersionInfoTests.cpp
  ${RCCL_TEST_DIR}/DdaCollCommonTests.cpp
  ${RCCL_TEST_DIR}/RomeTopoConsensusTests.cpp
  ${RCCL_TEST_DIR}/MiscTests.cpp
  ${RCCL_TEST_DIR}/VersionGateTests.cpp
  ${RCCL_TEST_DIR}/TimeoutTests.cpp
  ${RCCL_TEST_DIR}/EnqueueCountTests.cpp
  ${RCCL_TEST_DIR}/BootstrapBidirTests.cpp
  ${RCCL_TEST_DIR}/AltRsmiTests.cpp
  ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
  ${RCCL_HIPIFY_DIR}/src/misc/kernel_config.cc
  ${RCCL_HIPIFY_DIR}/src/misc/alt_rsmi.cc
  ${RCCL_HIPIFY_DIR}/src/graph/rome_topo_consensus.cc
)

target_include_directories(rccl-HostUnitTests PRIVATE ${RCCL_INCLUDE_DIRS})

target_compile_definitions(rccl-HostUnitTests PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  RCCL_EXPOSE_STATIC
  ARSMI_TEST_BUILD
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
)

target_link_options(rccl-HostUnitTests PRIVATE -no-hip-rt)

target_link_libraries(rccl-HostUnitTests PRIVATE
  GTest::GTest
  Threads::Threads
  fmt::fmt-header-only
  -Wl,--whole-archive
  rccl-source-wrappers
  -Wl,--no-whole-archive
)
