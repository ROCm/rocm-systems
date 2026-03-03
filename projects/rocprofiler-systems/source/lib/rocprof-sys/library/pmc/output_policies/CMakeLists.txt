# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Output Policies Sources
set(pmc_output_policies_sources
    ${CMAKE_CURRENT_LIST_DIR}/perfetto_policy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cache_policy.hpp
)

# Add to parent variable (will be used by pmc CMakeLists.txt)
set(pmc_output_policies_sources ${pmc_output_policies_sources} PARENT_SCOPE)
