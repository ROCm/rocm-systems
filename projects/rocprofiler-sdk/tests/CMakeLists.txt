#
# Integration tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(COMMAND rocprofiler_deactivate_clang_tidy)
    rocprofiler_deactivate_clang_tidy()
endif()

project(rocprofiler-tests LANGUAGES C CXX)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tests")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

enable_testing()
include(CTest)

# needed for validation
find_package(Python3 REQUIRED)

# common utilities
add_subdirectory(common)

# applications used by integration tests
add_subdirectory(apps)

# tool libraries
add_subdirectory(kernel-tracing)
