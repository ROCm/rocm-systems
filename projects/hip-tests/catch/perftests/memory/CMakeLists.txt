# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
    find_package(NUMA)
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
