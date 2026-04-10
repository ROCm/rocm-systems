#
# OpenCL test application for ROCm profiling
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-opencl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Create executable
add_executable(opencl-vector-add)
target_sources(opencl-vector-add PRIVATE opencl-vector-add.cpp)

# Compile options
target_compile_options(opencl-vector-add PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow
                                                 -Werror)

# compile definitions
math(EXPR CL_TARGET_OPENCL_VERSION
     "(${OpenCL_VERSION_MAJOR} * 100) + (${OpenCL_VERSION_MINOR} * 10)")
target_compile_definitions(opencl-vector-add
                           PRIVATE CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})

# Link OpenCL
target_link_libraries(opencl-vector-add PRIVATE OpenCL::OpenCL)

# Find and link threads
find_package(Threads REQUIRED)
target_link_libraries(opencl-vector-add PRIVATE Threads::Threads)

# Find and link rocprofiler-sdk-roctx
find_package(rocprofiler-sdk-roctx REQUIRED)
target_link_libraries(opencl-vector-add
                      PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
