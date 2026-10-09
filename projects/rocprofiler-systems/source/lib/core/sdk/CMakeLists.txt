# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(sdk_headers
    ${CMAKE_CURRENT_LIST_DIR}/tracing-config-deps.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracing-config.hpp
)

target_sources(rocprofiler-systems-core-library PRIVATE ${sdk_headers})

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
