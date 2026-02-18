# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# AMD SMI Driver Sources
set(pmc_amd_smi_driver_sources ${CMAKE_CURRENT_LIST_DIR}/driver.hpp)

# Add to parent variable (will be used by amd_smi provider CMakeLists.txt)
set(pmc_amd_smi_driver_sources ${pmc_amd_smi_driver_sources} PARENT_SCOPE)
