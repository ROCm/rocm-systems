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

include(GNUInstallDirs)
# always use lib instead of lib64
set(CMAKE_INSTALL_LIBDIR "lib")

# needed for validation
find_package(Python3 REQUIRED)

# common utilities
add_subdirectory(common)

# tool libraries used for data collection during integration tests
add_subdirectory(tools)

# applications used by integration tests
add_subdirectory(apps)

# validation tests
add_subdirectory(kernel-tracing)
add_subdirectory(async-copy-tracing)

# rocprofv3 validation tests
add_subdirectory(rocprofv3)
