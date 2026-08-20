# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(config_headers ${CMAKE_CURRENT_LIST_DIR}/trace_period_config.hpp)

target_sources(rocprofiler-systems-core-library PRIVATE ${config_headers})

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
