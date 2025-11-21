# Minimum required c++ standard is 17 for compilation
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set cmake_prefix_path for searching, ROCM_PATH if avail otherwise default to general rocm install path
set(CMAKE_PREFIX_PATH $ENV{ROCM_PATH} "/opt/rocm/")

find_package(
    rocprofiler-sdk
    HINTS $ENV{ROCM_PATH}/lib/cmake /opt/rocm/lib/cmake
    CONFIG
    REQUIRED
)

add_library(rocprofiler-compute-tool SHARED)
target_sources(rocprofiler-compute-tool PRIVATE rocprofiler_compute_tool.cpp helper.cpp)
target_link_libraries(rocprofiler-compute-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk)
