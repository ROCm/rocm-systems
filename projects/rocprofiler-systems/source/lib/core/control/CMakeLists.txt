# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(control_sources ${CMAKE_CURRENT_LIST_DIR}/session.cpp)

set(control_headers ${CMAKE_CURRENT_LIST_DIR}/session.hpp)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${control_sources} ${control_headers}
)
