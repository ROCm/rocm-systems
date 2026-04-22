# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Allow ROCM_INSTALL_PATH to be supplied via the environment.
if(NOT ROCM_INSTALL_PATH AND DEFINED ENV{ROCM_INSTALL_PATH})
  set(ROCM_INSTALL_PATH "$ENV{ROCM_INSTALL_PATH}")
endif()

find_package(amd_comgr 2.9 CONFIG
  PATHS
    ${ROCM_INSTALL_PATH}
    /opt/rocm/
  PATH_SUFFIXES
    cmake/amd_comgr
    lib/cmake/amd_comgr)

if (NOT amd_comgr_FOUND)
  find_package(amd_comgr 3.0 REQUIRED CONFIG
    PATHS
      ${ROCM_INSTALL_PATH}
      /opt/rocm/
    PATH_SUFFIXES
      cmake/amd_comgr
      lib/cmake/amd_comgr)
endif()

get_target_property(_amd_comgr_lib_type amd_comgr TYPE)
target_compile_definitions(rocclr PUBLIC)
if(_amd_comgr_lib_type STREQUAL "SHARED_LIBRARY")
  target_compile_definitions(rocclr PUBLIC COMGR_DYN_DLL)
endif()
target_link_libraries(rocclr PUBLIC amd_comgr)

# Comgr DLL name for Windows dynamic loading
if(WIN32)
  set(COMGR_DLL_NAME "amd_comgr.dll" CACHE STRING "Windows Comgr DLL name for dynamic loading")
  target_compile_definitions(rocclr PRIVATE COMGR_DLL_NAME="${COMGR_DLL_NAME}")
endif()
