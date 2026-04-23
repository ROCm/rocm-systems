# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipMemGetAllocationGranularity.cc
    hipMemRetainAllocationHandle.cc)

if(HIP_PLATFORM MATCHES "nvidia")
  set(TEST_SRC
    ${TEST_SRC}
    hipMemMapArrayAsync.cc)
endif()

if(UNIX)
  if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC
      ${TEST_SRC}
      hipMemGetHandleForAddressRange.cc)
  endif()
  set(TEST_SRC
      ${TEST_SRC}
      hipMemExportToShareableHandle.cc
      hipMemImportFromShareableHandle.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      ${TEST_SRC}
      hipGetProcAddressVmmApis.cc)
endif()

set(TEST_SRC
    ${TEST_SRC}
    hipMemAddressFree.cc
    hipMemAddressReserve.cc
    hipMemCreate.cc
    hipMemSetGetAccess.cc
    hipMemGetAllocationPropertiesFromHandle.cc
    hipMemMap.cc
    hipMemRelease.cc
    hipMemUnmap.cc)

hip_add_exe_to_target(NAME VirtualMemoryManagementTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC})
