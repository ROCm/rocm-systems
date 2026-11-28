# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

find_package(amd_comgr 3.2 REQUIRED CONFIG
  PATHS
    ${ROCM_PATH}
    ${ROCM_INSTALL_PATH}
  PATH_SUFFIXES
    cmake/amd_comgr
    lib/cmake/amd_comgr)

get_target_property(_amd_comgr_lib_type amd_comgr TYPE)
target_compile_definitions(rocclr PUBLIC)
if(_amd_comgr_lib_type STREQUAL "SHARED_LIBRARY")
  # comgr is loaded with dlopen (amd::Comgr): take only its include directories, so that
  # libamdhip64 has no DT_NEEDED entry for it whatever the linker's --as-needed setting is.
  target_compile_definitions(rocclr PUBLIC COMGR_DYN_DLL)
  target_link_libraries(rocclr PUBLIC $<COMPILE_ONLY:amd_comgr>)
else()
  target_link_libraries(rocclr PUBLIC amd_comgr)
endif()

# Comgr DLL name for Windows dynamic loading
if(WIN32)
  set(COMGR_DLL_NAME "amd_comgr.dll" CACHE STRING "Windows Comgr DLL name for dynamic loading")
  target_compile_definitions(rocclr PRIVATE COMGR_DLL_NAME="${COMGR_DLL_NAME}")
endif()
