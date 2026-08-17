# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems type-traits headers
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-type-traits-library INTERFACE)
add_library(
    rocprofiler-systems::type-traits-library
    ALIAS rocprofiler-systems-type-traits-library
)
add_library(
    rocprofiler-systems::rocprofiler-systems-type-traits-library
    ALIAS rocprofiler-systems-type-traits-library
)

target_sources(
    rocprofiler-systems-type-traits-library
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thread_state_policy.hpp
)

get_filename_component(
    TYPE_TRAITS_SOURCE_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    DIRECTORY
)

target_include_directories(
    rocprofiler-systems-type-traits-library
    INTERFACE $<BUILD_INTERFACE:${TYPE_TRAITS_SOURCE_INCLUDE_DIR}>
)
