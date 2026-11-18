# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(callback_domains_roctx_headers
    ${CMAKE_CURRENT_LIST_DIR}/control_api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core_api.hpp
)

set(callback_domains_roctx_headers ${callback_domains_roctx_headers} PARENT_SCOPE)
set(callback_domains_roctx_test_sources
    ${callback_domains_roctx_test_sources}
    PARENT_SCOPE
)
