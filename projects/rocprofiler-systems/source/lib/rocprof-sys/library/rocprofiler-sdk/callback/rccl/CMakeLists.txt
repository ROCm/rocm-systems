# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(callback_domains_rccl_headers
    ${CMAKE_CURRENT_LIST_DIR}/device_resolver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/event_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rccl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/transferred_bytes_tracker.hpp
)

set(callback_domains_rccl_headers ${callback_domains_rccl_headers} PARENT_SCOPE)
set(callback_domains_rccl_test_sources ${callback_domains_rccl_test_sources} PARENT_SCOPE)
