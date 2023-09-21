#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-samples LANGUAGES C CXX)

add_library(rocprofiler-samples-build-flags INTERFACE)
add_library(rocprofiler::samples-build-flags ALIAS rocprofiler-samples-build-flags)
target_compile_options(rocprofiler-samples-build-flags INTERFACE -W -Wall -Wextra
                                                                 -Wshadow)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-samples-build-flags INTERFACE -Werror)
endif()

# add_subdirectory(api_tracing)
add_subdirectory(pc_sampling)
add_subdirectory(api_callback_tracing)
add_subdirectory(api_buffered_tracing)
