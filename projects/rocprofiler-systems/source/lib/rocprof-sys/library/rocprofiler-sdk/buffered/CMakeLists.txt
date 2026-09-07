# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(rocprofiler_sdk_kfd_events_headers
    ${CMAKE_CURRENT_LIST_DIR}/kfd_event_dropped_events.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_event_page_fault.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_event_page_migrate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_event_queue.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_event_unmap_from_gpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_page_fault.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_page_migrate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_queue.hpp
)

set(rocprofiler_sdk_kfd_events_headers ${rocprofiler_sdk_kfd_events_headers} PARENT_SCOPE)
