# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

find_package(amd_comgr 2.9 CONFIG
  PATHS
    /opt/rocm/
    ${ROCM_INSTALL_PATH}
  PATH_SUFFIXES
    cmake/amd_comgr
    lib/cmake/amd_comgr)

if (NOT amd_comgr_FOUND)
  find_package(amd_comgr 3.0 REQUIRED CONFIG
    PATHS
      /opt/rocm/
      ${ROCM_INSTALL_PATH}
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

if(CLR_BUILD_HIP)
  # Temporary hack for versioned comgr needed by hiprtc
  file(STRINGS ${HIP_COMMON_DIR}/VERSION VERSION_LIST REGEX "^[0-9]+")
  list(GET VERSION_LIST 0 HIP_VERSION_MAJOR)
  list(GET VERSION_LIST 1 HIP_VERSION_MINOR)

  add_definitions(-DHIP_MAJOR_VERSION=${HIP_VERSION_MAJOR})
  add_definitions(-DHIP_MINOR_VERSION=${HIP_VERSION_MINOR})
endif()
