cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Standalone build of rccl-UnitTestsMicro.
#
# ROCm is a HARD prerequisite (epic AICOMRCCL-1661: "ROCm toolchain is
# available"). This project compiles the hipified RCCL sources + the fakes
# layer HOST-ONLY with hipcc's --offload-host-only, using the REAL ROCm headers.
# There is no CPU-only / g++ path and no stubbed <hip/*>,<hsa/*>,<cuda*> headers.
# It still links neither librccl.so nor the HIP runtime (-no-hip-rt); every HIP
# symbol the tests reach is satisfied by fakes/ + micro_link_stubs.cc.
#
# The only thing it needs beyond ROCm is a hipified source snapshot from a prior
# RCCL build (build/<release|debug>/hipify) -- point at it with -DRCCL_BUILD_DIR.
# ---------------------------------------------------------------------------

# --- hipcc is mandatory: auto-detect via ROCM_PATH, or pass -DCMAKE_CXX_COMPILER=hipcc ---
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
      "hipcc not found. ROCm is a prerequisite for the microtests. Either:\n"
      "  cmake -B build -DCMAKE_CXX_COMPILER=/path/to/hipcc\n"
      "  cmake -B build -DROCM_PATH=/opt/rocm-7.x")
  endif()
endif()

# Compile the hipified sources for the HOST pass only (no amdgcn device pass).
# --offload-host-only keeps this a normal host compile while still resolving the
# real ROCm headers. If g++ is present, point hipcc at its libstdc++ so the C++
# standard headers are found; otherwise rely on hipcc's own defaults.
find_program(_system_gxx g++)
if(_system_gxx)
  execute_process(
    COMMAND ${_system_gxx} -print-file-name=libstdc++.a
    OUTPUT_VARIABLE _gcc_libstdcpp
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(_gcc_install_dir "${_gcc_libstdcpp}" DIRECTORY)
  set(CMAKE_CXX_FLAGS_INIT "--gcc-install-dir=${_gcc_install_dir} --offload-host-only")
else()
  set(CMAKE_CXX_FLAGS_INIT "--offload-host-only")
endif()

project(rccl-UnitTestsMicro-standalone LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(fmt)

set(SA      ${CMAKE_CURRENT_SOURCE_DIR})
set(MICRO   ${SA}/..)
set(TESTDIR ${SA}/../..)
set(RCCL    ${SA}/../../..)
# Hipified RCCL sources + generated headers from a prior ROCm build.
# Override with -DRCCL_BUILD_DIR=/path/to/projects/rccl/build/<release|debug>.
set(RCCL_BUILD_DIR "${RCCL}/build/release" CACHE PATH "RCCL build dir containing hipify/ and include/")
set(HIPIFY  ${RCCL_BUILD_DIR}/hipify)
set(GENINC  ${RCCL_BUILD_DIR}/include)

if(NOT EXISTS "${HIPIFY}/src/transport/p2p_tmp.cc")
  message(FATAL_ERROR
    "Hipified sources not found at ${HIPIFY}. Run the RCCL build first, then\n"
    "  cmake -B build -DRCCL_BUILD_DIR=/path/to/projects/rccl/build/<release|debug>")
endif()

add_executable(rccl-UnitTestsMicro
  ${MICRO}/p2p-test.cc
  ${MICRO}/fakes/nccl_fakes.cc
  ${MICRO}/fakes/p2p_fakes.cc
  ${MICRO}/fakes/hip_fakes.cc
  ${SA}/micro_link_stubs.cc
  ${TESTDIR}/common/main_altrsmi.cpp
  ${TESTDIR}/common/ProcessIsolatedTestRunner.cpp
)

target_compile_options(rccl-UnitTestsMicro PRIVATE
  -Wno-unused-parameter -Wno-unused-variable)

target_include_directories(rccl-UnitTestsMicro PRIVATE
  ${MICRO}
  ${HIPIFY}/src/include
  ${HIPIFY}/src/include/plugin
  ${HIPIFY}/src/include/plugin/tuner
  ${GENINC}
  ${HIPIFY}/src
  ${HIPIFY}/src/graph
  ${TESTDIR}
  ${TESTDIR}/common
  ${fmt_SOURCE_DIR}/include
)

target_compile_definitions(rccl-UnitTestsMicro PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70002
  P2P_CC_PATH="${HIPIFY}/src/transport/p2p_tmp.cc"
)

# Do not link the HIP runtime: the fakes provide every HIP symbol the tests reach.
target_link_options(rccl-UnitTestsMicro PRIVATE -no-hip-rt)
target_link_libraries(rccl-UnitTestsMicro PRIVATE
  GTest::gtest Threads::Threads fmt::fmt)

# llvm source-based coverage (matches the ROCm toolchain's llvm-cov/llvm-profdata).
# On by default; disable with -DMICRO_COVERAGE=OFF.
option(MICRO_COVERAGE "Build rccl-UnitTestsMicro with llvm source-based coverage" ON)
if(MICRO_COVERAGE)
  target_compile_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()
