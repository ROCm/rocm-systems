## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target rocprofiler-compute-tool)
add_library(
    ${target}
    SHARED
    rocprofiler_compute_tool.h
    rocprofiler_compute_tool.cpp
    counters_writer.h
    counters_writer.cpp
    sdk_callbacks.h
    sdk_callbacks.cpp
    input_parameters.h
    input_parameters.cpp
    sdk_wrapper.h
    sdk_wrapper.cpp
    tool_setup.h
    tool_setup.cpp
)

target_include_directories(${target} PUBLIC .)
target_link_libraries(${target} PUBLIC rocprofiler-sdk::rocprofiler-sdk gsl_assert)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
