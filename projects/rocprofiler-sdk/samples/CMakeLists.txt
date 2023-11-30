#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-samples LANGUAGES C CXX)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE
        "RelWithDebInfo"
        CACHE STRING "Build type" FORCE)
endif()

enable_testing()
include(CTest)

# common utilities for samples
add_subdirectory(common)

# actual samples
add_subdirectory(api_callback_tracing)
add_subdirectory(api_buffered_tracing)
add_subdirectory(code_object_tracing)
add_subdirectory(counter_collection)
add_subdirectory(intercept_table)
