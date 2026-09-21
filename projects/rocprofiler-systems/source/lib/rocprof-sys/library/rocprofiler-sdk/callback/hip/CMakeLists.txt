# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(callback_domains_hip_headers
    ${CMAKE_CURRENT_LIST_DIR}/compiler_api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime_api.hpp
)

set(callback_domains_hip_headers ${callback_domains_hip_headers} PARENT_SCOPE)
set(callback_domains_hip_test_sources ${callback_domains_hip_test_sources} PARENT_SCOPE)
