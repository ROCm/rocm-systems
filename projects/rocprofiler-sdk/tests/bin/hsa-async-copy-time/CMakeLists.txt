#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-hsa-async-copy-time LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hsa-async-copy-time)
target_sources(hsa-async-copy-time PRIVATE hsa-async-copy-time.cpp)
target_compile_options(hsa-async-copy-time PRIVATE -Wpedantic)

find_package(Threads REQUIRED)
target_link_libraries(hsa-async-copy-time PRIVATE Threads::Threads
                                                  rocprofiler-sdk::tests-common-library)

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

target_link_libraries(hsa-async-copy-time PRIVATE hsa-runtime64::hsa-runtime64)
