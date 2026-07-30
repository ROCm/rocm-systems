cmake_minimum_required(VERSION 3.21)
project(rocprofiler-compute-tool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} "/opt/rocm/")

# libdw has no upstream CMake config and the SDK's bundled Findlibdw.cmake is
# brittle across pkg-config / multiarch layouts. Define libdw::libdw via
# pkg-config up front so the SDK's internal find_package(libdw) check finds
# the target already present.
find_package(PkgConfig REQUIRED)
pkg_check_modules(libdw REQUIRED IMPORTED_TARGET libdw)
if(NOT TARGET libdw::libdw)
    add_library(libdw::libdw ALIAS PkgConfig::libdw)
endif()

find_package(rocprofiler-sdk REQUIRED CONFIG)
find_package(amd_comgr REQUIRED CONFIG)

add_subdirectory(external)
add_subdirectory(utils)
add_subdirectory(pc_sampling_collector)
add_subdirectory(rocprofiler_compute_tool)
