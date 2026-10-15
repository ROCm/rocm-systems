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
    environ_cache.h
    environ_cache.cpp
    input_parameters.h
    input_parameters.cpp
    kernel_name.h
    kernel_name.cpp
    output_registry.h
    output_registry.cpp
    dispatch_writer.h
    dispatch_writer.cpp
    kernel_symbols_writer.h
    kernel_symbols_writer.cpp
    sdk_wrapper.h
    sdk_wrapper.cpp
    pc_sampling_feature.h
    pc_sampling_feature.cpp
)

target_include_directories(${target} PUBLIC .)
# compression is PRIVATE: not exposed in public headers.
target_link_libraries(
    ${target}
    PUBLIC rocprofiler-sdk::rocprofiler-sdk gsl_assert pc-sampling-collector csv
    PRIVATE compression
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
