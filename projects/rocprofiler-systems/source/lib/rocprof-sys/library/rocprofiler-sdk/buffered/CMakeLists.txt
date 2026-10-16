# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_subdirectory(kfd)

set(buffered_domains_root_headers
    ${CMAKE_CURRENT_LIST_DIR}/kernel_dispatch.hpp
    ${CMAKE_CURRENT_LIST_DIR}/memory_allocation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/memory_copy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/scratch_memory.hpp
)

set(buffered_domains_headers ${kfd_domains_headers} ${buffered_domains_root_headers})

set(buffered_domains_headers ${buffered_domains_headers} PARENT_SCOPE)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
