# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Common Sources
set(pmc_common_sources
    ${CMAKE_CURRENT_LIST_DIR}/collector_slice.hpp
    ${CMAKE_CURRENT_LIST_DIR}/settings.hpp
)

# Add to parent variable (will be used by pmc CMakeLists.txt)
set(pmc_common_sources ${pmc_common_sources} PARENT_SCOPE)

# Tests
if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
