# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#
set(component_sources)

set(component_headers ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${component_sources} ${component_headers}
)
