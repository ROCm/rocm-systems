#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-samples LANGUAGES C CXX)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "samples")
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE
        "RelWithDebInfo"
        CACHE STRING "Build type" FORCE)
endif()

if(NOT PROJECT_IS_TOP_LEVEL)
    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}] ")
endif()

include(GNUInstallDirs)

# always use lib instead of lib64
set(CMAKE_INSTALL_LIBDIR "lib")

# this should be defaulted to OFF by ROCm 7.0.1 or 7.1 this should only used to disable
# sample tests in extreme circumstances
option(ROCPROFILER_DISABLE_UNSTABLE_CTESTS "Disable unstable tests" ON)

enable_testing()
include(CTest)

# generally needed
find_package(rocprofiler-sdk REQUIRED)

# get the gfx architectures that are present on the system
rocprofiler_sdk_get_gfx_architectures(rocprofiler-sdk-samples-gfx-info ECHO)

# common utilities for samples
add_subdirectory(common)

# actual samples
add_subdirectory(api_callback_tracing)
add_subdirectory(api_buffered_tracing)
add_subdirectory(code_object_tracing)
add_subdirectory(counter_collection)
add_subdirectory(intercept_table)
add_subdirectory(code_object_isa_decode)
add_subdirectory(external_correlation_id_request)
add_subdirectory(pc_sampling)
add_subdirectory(openmp_target)
