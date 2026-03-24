# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-bin-common static library
#
# Shared functionality for bin tools (run, sample, etc.)
#
# ------------------------------------------------------------------------------#

add_library(
    rocprofiler-systems-bin-common
    STATIC
    json_config.cpp
    preset_registry.cpp
    common_utils.cpp
)

target_include_directories(
    rocprofiler-systems-bin-common
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..
)

target_link_libraries(
    rocprofiler-systems-bin-common
    PUBLIC
        rocprofiler-systems::rocprofiler-systems-json
        rocprofiler-systems::rocprofiler-systems-common-library
)

# Create alias for consistent naming
add_library(
    rocprofiler-systems::rocprofiler-systems-bin-common
    ALIAS rocprofiler-systems-bin-common
)
