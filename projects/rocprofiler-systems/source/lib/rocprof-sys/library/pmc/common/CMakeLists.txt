# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Common Types
set(pmc_common_types_sources ${CMAKE_CURRENT_LIST_DIR}/types.hpp)

# Add to parent variable (will be used by pmc CMakeLists.txt)
set(pmc_common_types_sources ${pmc_common_types_sources} PARENT_SCOPE)
