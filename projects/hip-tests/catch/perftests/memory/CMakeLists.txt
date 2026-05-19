# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
  hipPerfMemcpy.cc
  hipPerfBufferCopyRectSpeed.cc
  hipPerfBufferCopySpeed.cc
  hipPerfDevMemReadSpeed.cc
  hipPerfDevMemWriteSpeed.cc
  hipPerfMemFill.cc
  hipPerfMemMallocCpyFree.cc
  hipPerfMemset.cc
  hipPerfSampleRate.cc
  hipPerfSharedMemReadSpeed.cc
  hipPerfBufferCopySpeedP2P.cc
  hipPerfBufferCopySpeedAll2All.cc
  hipPerfMempool.cc
  hipPerfDeviceHeapMemory.cc
  hipPerfBufferCopyInterGpuPerformance.cc
  hipPerfMemcpyAsyncSpeed.cc
  hipPerfMemsetAsyncSpeed.cc
)

if(HIP_PLATFORM MATCHES "amd")
  if(WIN32)
    set(TEST_SRC ${TEST_SRC} hipPerfHostNumaAllocWin.cc)
    hip_add_exe_to_target(NAME perfMemoryTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME perf_test)
  else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
    find_package(NUMA QUIET)
    if(NUMA_FOUND)
      set(NUMA "${NUMA_LIBRARIES}")
    else()
      find_library(NUMA NAMES numa REQUIRED)
    endif()
    if(NUMA_FOUND)
      # Test code still use libnuma apis
      set(TEST_SRC ${TEST_SRC} hipPerfHostNumaAlloc.cc)
      hip_add_exe_to_target(NAME perfMemoryTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME perf_test
                      LINKER_LIBS numa)
    else()
      hip_add_exe_to_target(NAME perfMemoryTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME perf_test)
    endif()
  endif()
else()
  hip_add_exe_to_target(NAME perfMemoryTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME perf_test)
endif()
