#
# Integration tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(COMMAND rocprofiler_deactivate_clang_tidy)
    rocprofiler_deactivate_clang_tidy()
endif()

project(rocprofiler-sdk-tests LANGUAGES C CXX)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tests")
set(ROCPROFILER_SDK_TESTS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(ROCPROFILER_SDK_TESTS_BINARY_DIR "${PROJECT_BINARY_DIR}")

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

# define the library output directory
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

# needed for validation
find_package(Python3 REQUIRED)

# generally needed
find_package(rocprofiler-sdk REQUIRED)

# get the gfx architectures that are present on the system
rocprofiler_sdk_get_gfx_architectures(rocprofiler-sdk-tests-gfx-info ECHO)

# configure python module <BINARY_DIR>/rocprofiler_sdk/pytest_utils
add_subdirectory(pytest-packages)

# common utilities
add_subdirectory(common)

# tool libraries used for data collection during integration tests
add_subdirectory(tools)

# libraries used by integration test applications
add_subdirectory(lib)

# applications used by integration tests
add_subdirectory(bin)

# validation tests
add_subdirectory(kernel-tracing)
add_subdirectory(async-copy-tracing)
add_subdirectory(hsa-memory-allocation)
add_subdirectory(scratch-memory-tracing)
add_subdirectory(c-tool)
add_subdirectory(page-migration)
add_subdirectory(thread-trace)
add_subdirectory(pc_sampling)
add_subdirectory(hip-graph-tracing)
add_subdirectory(counter-collection)
add_subdirectory(openmp-tools)
add_subdirectory(rocdecode)
add_subdirectory(rocjpeg)

# rocpd validation tests
add_subdirectory(rocpd)

# rocprofv3 validation tests
add_subdirectory(rocprofv3)

# python bindings
add_subdirectory(python-bindings)
add_subdirectory(rocprofv3-avail)
