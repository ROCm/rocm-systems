# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(callback_domains_headers ${CMAKE_CURRENT_LIST_DIR}/code_object.hpp)

set(callback_domains_headers ${callback_domains_headers} PARENT_SCOPE)
