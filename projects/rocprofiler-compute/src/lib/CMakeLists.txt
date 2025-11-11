# Minimum required c++ standard is 17 for compilation
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set cmake_prefix_path for searching, ROCM_PATH if avail otherwise default to general rocm install path
set(CMAKE_PREFIX_PATH $ENV{ROCM_PATH} "/opt/rocm/")

set(CMAKE_HIP_COMPILER "amdclang++" CACHE STRING "desired c++ compiler" FORCE)
if(
    (CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
    OR (CMAKE_HIP_COMPILER_ID STREQUAL "ROCMClang")
)
    message(STATUS "Using ${CMAKE_HIP_COMPILER} to build for amdgpu backend")
else()
    message(
        FATAL_ERROR
        "'amdclang++' compiler required to compile test binaries for ROCm platform."
    )
endif()

find_package(rocprofiler-sdk REQUIRED)

add_library(rocprofiler-compute-tool SHARED)
target_sources(rocprofiler-compute-tool PRIVATE rocprofiler_compute_tool.cpp helper.cpp)
target_link_libraries(rocprofiler-compute-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk)
