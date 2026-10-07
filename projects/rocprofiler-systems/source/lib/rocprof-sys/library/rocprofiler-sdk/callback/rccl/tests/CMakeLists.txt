# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(callback_domains_rccl_test_sources
    ${CMAKE_CURRENT_LIST_DIR}/test_device_resolver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_event_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_rccl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_transferred_bytes_tracker.cpp
)

set(callback_domains_rccl_test_sources ${callback_domains_rccl_test_sources} PARENT_SCOPE)
