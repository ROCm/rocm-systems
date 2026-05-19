# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_subdirectory(drivers)

# Procfs Device Provider Sources
set(pmc_procfs_provider_sources
    ${CMAKE_CURRENT_LIST_DIR}/provider.hpp
    ${pmc_procfs_driver_sources}
)

set(pmc_procfs_provider_sources ${pmc_procfs_provider_sources} PARENT_SCOPE)
