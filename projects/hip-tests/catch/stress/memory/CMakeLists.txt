# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

find_package(TBB REQUIRED)

# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipMalloc.cc
    memcpy.cc
    hipMemcpyMThreadMSize.cc
    hipMallocManagedStress.cc
    hipMemPrftchAsyncStressTst.cc
    hipHostMallocStress.cc
    hipHostRegisterStress.cc
)

if(UNIX)
  set(TEST_SRC ${TEST_SRC}
               hipHmmOvrSubscriptionTst.cc)
  add_executable(hold_memory EXCLUDE_FROM_ALL hold_memory.cc)
  add_dependencies(stress_test hold_memory)
endif()

hip_add_exe_to_target(NAME memory_stress
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test
                      LINKER_LIBS TBB::tbb)
