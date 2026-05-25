## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
add_subdirectory(pc_sampling_collector)

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
    env_parameters.h
    env_parameters.cpp
    sdk_wrapper.h
    sdk_wrapper.cpp
    tool_data.h
)

target_include_directories(${target} PUBLIC .)
target_link_libraries(
    ${target}
    PUBLIC rocprofiler-sdk::rocprofiler-sdk gsl_assert
    PRIVATE pc-sampling-collector nlohmann_json
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
