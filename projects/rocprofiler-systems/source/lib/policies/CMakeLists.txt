# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems policy headers
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-policies-library INTERFACE)
add_library(
    rocprofiler-systems::policies-library
    ALIAS rocprofiler-systems-policies-library
)
add_library(
    rocprofiler-systems::rocprofiler-systems-policies-library
    ALIAS rocprofiler-systems-policies-library
)

target_sources(
    rocprofiler-systems-policies-library
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/thread_state_policy.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/backend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/externals.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/domain_service/backend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/domain_service/externals.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/gpu_perf_counters/backend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/gpu_perf_counters/externals.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/tracing_config/backend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-sdk/tracing_config/externals.hpp
)

get_filename_component(
    POLICIES_SOURCE_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    DIRECTORY
)

target_include_directories(
    rocprofiler-systems-policies-library
    INTERFACE $<BUILD_INTERFACE:${POLICIES_SOURCE_INCLUDE_DIR}>
)
