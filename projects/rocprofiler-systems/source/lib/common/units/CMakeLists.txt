# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(units_headers
    ${CMAKE_CURRENT_LIST_DIR}/data_size.hpp
    ${CMAKE_CURRENT_LIST_DIR}/frequency.hpp
    ${CMAKE_CURRENT_LIST_DIR}/power.hpp
    ${CMAKE_CURRENT_LIST_DIR}/quantity.hpp
)

target_sources(rocprofiler-systems-common-library INTERFACE ${units_headers})

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
