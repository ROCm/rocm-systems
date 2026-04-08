# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Add subdirectories
add_subdirectory(drivers)

# AMD SMI Device Provider Sources
set(pmc_amd_smi_provider_sources
    ${CMAKE_CURRENT_LIST_DIR}/provider.hpp
    ${pmc_amd_smi_driver_sources}
)

# Add to parent variable (will be used by device_providers CMakeLists.txt)
set(pmc_amd_smi_provider_sources ${pmc_amd_smi_provider_sources} PARENT_SCOPE)
