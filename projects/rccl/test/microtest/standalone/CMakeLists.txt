cmake_minimum_required(VERSION 3.16)
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
  -isystem ${SA}/stubs
  -include cstdint
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

target_link_libraries(rccl-UnitTestsMicro PRIVATE
  GTest::gtest Threads::Threads fmt::fmt)

# llvm source-based coverage (matches the ROCm toolchain's llvm-cov/llvm-profdata).
# On by default; disable with -DMICRO_COVERAGE=OFF.
option(MICRO_COVERAGE "Build rccl-UnitTestsMicro with llvm source-based coverage" ON)
if(MICRO_COVERAGE)
  target_compile_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()
