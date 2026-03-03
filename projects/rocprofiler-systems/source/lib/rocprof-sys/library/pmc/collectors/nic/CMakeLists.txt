# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Add tests subdirectory
add_subdirectory(tests)

# PMC NIC Collector Sources
set(pmc_nic_sources
    ${CMAKE_CURRENT_LIST_DIR}/collector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device.hpp
)

# Add to parent variable (will be used by collectors CMakeLists.txt)
set(pmc_nic_sources ${pmc_nic_sources} PARENT_SCOPE)
