# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(kfd_domains_headers
    ${CMAKE_CURRENT_LIST_DIR}/event_dropped_events.hpp
    ${CMAKE_CURRENT_LIST_DIR}/event_page_fault.hpp
    ${CMAKE_CURRENT_LIST_DIR}/event_page_migrate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/event_queue.hpp
    ${CMAKE_CURRENT_LIST_DIR}/event_unmap_from_gpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/page_fault.hpp
    ${CMAKE_CURRENT_LIST_DIR}/page_migrate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/queue.hpp
)

set(kfd_domains_headers ${kfd_domains_headers} PARENT_SCOPE)
