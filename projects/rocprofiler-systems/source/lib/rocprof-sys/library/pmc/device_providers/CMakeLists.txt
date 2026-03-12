# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Device Providers
add_subdirectory(amd_smi)

# Collect all device provider sources
set(pmc_device_provider_sources ${pmc_amd_smi_provider_sources})

# Add to parent variable (will be used by pmc CMakeLists.txt)
set(pmc_device_provider_sources ${pmc_device_provider_sources} PARENT_SCOPE)
