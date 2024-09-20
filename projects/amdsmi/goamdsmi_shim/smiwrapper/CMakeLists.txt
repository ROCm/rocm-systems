# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc.

set(go_amd_smi_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/goamdsmi.h
  ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_go_shim.h
  CACHE INTERNAL "")

set(go_amd_smi_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_go_shim.c
    CACHE INTERNAL "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${go_amd_smi_amdsmi_includes})

add_library(go_amd_smi_ OBJECT
            ${go_amd_smi_sources}
            ${go_amd_smi_headers})

### Shared libraries need PIC
set_property(TARGET ${go_amd_smi_} PROPERTY POSITION_INDEPENDENT_CODE 1)

