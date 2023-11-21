#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-samples LANGUAGES C CXX)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE
        "RelWithDebInfo"
        CACHE STRING "Build type" FORCE)
endif()

enable_testing()
include(CTest)

add_library(rocprofiler-samples-build-flags INTERFACE)
add_library(rocprofiler::samples-build-flags ALIAS rocprofiler-samples-build-flags)
target_compile_options(rocprofiler-samples-build-flags INTERFACE -W -Wall -Wextra
                                                                 -Wshadow)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-samples-build-flags INTERFACE -Werror)
endif()

add_subdirectory(api_callback_tracing)
add_subdirectory(api_buffered_tracing)
add_subdirectory(code_object_tracing)
add_subdirectory(counter_collection)
add_subdirectory(intercept_table)
add_subdirectory(pc_sampling)
