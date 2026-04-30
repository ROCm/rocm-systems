cmake_minimum_required(VERSION 3.21)
project(rocprofiler-compute-tool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set cmake_prefix_path for searching, ROCM_PATH if avail otherwise default to general rocm install path
set(CMAKE_PREFIX_PATH $ENV{ROCM_PATH} "/opt/rocm/")

find_package(
    rocprofiler-sdk
    HINTS ${CMAKE_PREFIX_PATH}/opt/rocm/lib/cmake
    CONFIG
    REQUIRED
)
find_package(libdw REQUIRED)
find_package(amd_comgr REQUIRED CONFIG
    HINTS  ${CMAKE_PREFIX_PATH} 
    PATHS  ${CMAKE_PREFIX_PATH} 
    PATH_SUFFIXES lib/cmake/amd_comgr)


add_subdirectory(external)
add_subdirectory(rocprofiler_compute_tool)
add_subdirectory(pc_sampling_collector)
add_subdirectory(utils)
