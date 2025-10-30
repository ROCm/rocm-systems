set(CMAKE_HIP_COMPILER "amdclang++" CACHE STRING "desired c++ compiler" FORCE)
if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
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
