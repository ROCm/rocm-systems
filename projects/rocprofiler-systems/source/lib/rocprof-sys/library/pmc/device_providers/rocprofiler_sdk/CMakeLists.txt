# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Add subdirectories
add_subdirectory(drivers)

# Rocprofiler-SDK Device Provider Sources
set(pmc_sdk_pmc_provider_sources
    ${CMAKE_CURRENT_LIST_DIR}/provider.hpp
    ${pmc_sdk_pmc_driver_sources}
)

# Add to parent variable
set(pmc_sdk_pmc_provider_sources ${pmc_sdk_pmc_provider_sources} PARENT_SCOPE)
