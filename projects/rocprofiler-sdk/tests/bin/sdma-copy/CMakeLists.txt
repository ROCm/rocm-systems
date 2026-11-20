#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-sdma-copy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sdma-copy)
target_sources(sdma-copy PRIVATE sdma-copy.cpp)
target_compile_options(sdma-copy PRIVATE -Wpedantic)

find_package(Threads REQUIRED)
find_package(
    hsa-runtime64
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})

target_link_libraries(sdma-copy PRIVATE Threads::Threads hsa-runtime64::hsa-runtime64)
