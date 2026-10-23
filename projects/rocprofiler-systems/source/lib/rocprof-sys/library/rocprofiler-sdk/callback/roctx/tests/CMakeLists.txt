# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(callback_domains_roctx_test_sources
    ${CMAKE_CURRENT_LIST_DIR}/test_control_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_core_api.cpp
)

set(callback_domains_roctx_test_sources
    ${callback_domains_roctx_test_sources}
    PARENT_SCOPE
)
