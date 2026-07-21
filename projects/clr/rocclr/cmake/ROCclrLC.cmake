# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

find_package(amd_comgr REQUIRED CONFIG
  PATHS
    ${ROCM_PATH}
    ${ROCM_INSTALL_PATH}
  PATH_SUFFIXES
    cmake/amd_comgr
    lib/cmake/amd_comgr)

target_link_libraries(rocclr PUBLIC amd_comgr)

# Comgr DLL name for Windows dynamic loading
if(WIN32)
  set(COMGR_DLL_NAME "amd_comgr.dll" CACHE STRING "Windows Comgr DLL name for dynamic loading")
  target_compile_definitions(rocclr PRIVATE COMGR_DLL_NAME="${COMGR_DLL_NAME}")
endif()
