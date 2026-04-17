# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# PMC Device Providers
add_subdirectory(amd_smi)
add_subdirectory(procfs)
add_subdirectory(rocprofiler_sdk)

# Collect all device provider sources
set(pmc_device_provider_sources
    ${pmc_amd_smi_provider_sources}
    ${pmc_procfs_provider_sources}
    ${pmc_gpu_perf_counter_provider_sources}
)

# Add to parent variable (will be used by pmc CMakeLists.txt)
set(pmc_device_provider_sources ${pmc_device_provider_sources} PARENT_SCOPE)
