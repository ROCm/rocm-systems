# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(callback_domains_hsa_headers
    ${CMAKE_CURRENT_LIST_DIR}/amd_ext_api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core_api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/finalize_ext_api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/image_ext_api.hpp
)

set(callback_domains_hsa_headers ${callback_domains_hsa_headers} PARENT_SCOPE)
set(callback_domains_hsa_test_sources ${callback_domains_hsa_test_sources} PARENT_SCOPE)
