# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(callback_domains_ompt_headers
    ${CMAKE_CURRENT_LIST_DIR}/decoder.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ompt.hpp
)

set(callback_domains_ompt_headers ${callback_domains_ompt_headers} PARENT_SCOPE)
set(callback_domains_ompt_test_sources ${callback_domains_ompt_test_sources} PARENT_SCOPE)
