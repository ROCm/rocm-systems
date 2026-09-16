# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_subdirectory(hsa)
add_subdirectory(hip)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(callback_domains_headers
    ${CMAKE_CURRENT_LIST_DIR}/code_object.hpp
    ${callback_domains_hsa_headers}
    ${callback_domains_hip_headers}
)

set(callback_domains_headers ${callback_domains_headers} PARENT_SCOPE)
