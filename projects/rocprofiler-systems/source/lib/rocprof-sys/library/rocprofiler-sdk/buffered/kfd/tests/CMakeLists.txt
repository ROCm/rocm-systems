# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(kfd_domains_test_sources
    ${CMAKE_CURRENT_LIST_DIR}/test_queue.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_page_fault.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_page_migrate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_event_queue.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_event_unmap_from_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_event_dropped_events.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_event_page_fault.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_event_page_migrate.cpp
    PARENT_SCOPE
)
