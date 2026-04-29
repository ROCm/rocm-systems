# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Procfs Driver Sources
set(pmc_procfs_driver_sources ${CMAKE_CURRENT_LIST_DIR}/driver.hpp)

set(pmc_procfs_driver_sources ${pmc_procfs_driver_sources} PARENT_SCOPE)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
