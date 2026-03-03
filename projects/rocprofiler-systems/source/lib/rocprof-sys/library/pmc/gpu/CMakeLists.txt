# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC GPU Data Types

set(pmc_gpu_types_sources
    ${CMAKE_CURRENT_LIST_DIR}/types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample.hpp
)

# Add to parent variable (will be used by pmc CMakeLists.txt)
set(pmc_gpu_types_sources ${pmc_gpu_types_sources} PARENT_SCOPE)
