/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include "hip/hip_ext.h"

/**
* @addtogroup hipExtLaunchKernelGGL
* @{
* @ingroup ModuleTest
* `void hipExtLaunchKernelGGL (F kernel, const dim3 &numBlocks, const dim3 &dimBlocks,
  std::uint32_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent,
  hipEvent_t stopEvent, std::uint32_t flags, Args... args)` -
* Launches kernel with dimention parameters and shared memory on stream with
* templated kernel and arguments.
*/

/**
 * Test Description
 * ------------------------
 * - Test case to verify kernel execution time of the particular kernel.
 * - Test case to verify hipExtLaunchKernelGGL API by disabling time flag in event creation.

 * Test source
 * ------------------------
 * - catch/unit/module/hipExtLaunchKernelGGL.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/

__device__ int globalvar = 1;
__device__ void Delay(uint32_t interval, const uint32_t ticks_per_ms) {
  while (interval--) {
#if HT_AMD
    uint64_t start = wall_clock64();
    while (wall_clock64() - start < ticks_per_ms) {
      __builtin_amdgcn_s_sleep(10);
    }
#endif
#if HT_NVIDIA
    uint64_t start = clock64();
    while (clock64() - start < ticks_per_ms) {
    }
#endif
  }
}
__global__ void TwoSecKernel(int clockrate) { Delay(2000, clockrate); }
__global__ void FourSecKernel(int clockrate) { Delay(4000, clockrate); }

bool DisableTimeFlag() {
  bool testStatus = true;
  hipStream_t stream1;
  HIP_CHECK(hipSetDevice(0));
  hipError_t e;
  float time_2sec;
  hipEvent_t start_event1, end_event1;
  int clkRate = 0;
#if HT_AMD
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeWallClockRate, 0));
#endif
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
#endif
  HIP_CHECK(hipEventCreateWithFlags(&start_event1, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&end_event1, hipEventDisableTiming));
  HIP_CHECK(hipStreamCreate(&stream1));
  hipExtLaunchKernelGGL((TwoSecKernel), dim3(1), dim3(1), 0, stream1, start_event1, end_event1, 0,
                        clkRate);
  HIP_CHECK(hipStreamSynchronize(stream1));
  e = hipEventElapsedTime(&time_2sec, start_event1, end_event1);
  if (e == hipErrorInvalidHandle) {
    testStatus = true;
  } else {
    testStatus = false;
  }
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipEventDestroy(start_event1));
  HIP_CHECK(hipEventDestroy(end_event1));
  return testStatus;
}

bool ConcurencyCheck_GlobalVar(int conc_flag) {
  bool testStatus = true;
  hipStream_t stream1;
  int deviceGlobal_h = 0;
  HIP_CHECK(hipSetDevice(0));
  int clkRate = 0;
#if HT_AMD
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeWallClockRate, 0));
#endif
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
#endif
  HIP_CHECK(hipStreamCreate(&stream1));
  hipDeviceProp_t props{};
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  if ((std::string(props.gcnArchName).find("gfx1101") != std::string::npos) ||
      (std::string(props.gcnArchName).find("gfx1100") != std::string::npos)) {
    hipExtLaunchKernelGGL((FourSecKernel), dim3(1), dim3(1), 0, stream1, nullptr, nullptr,
                          conc_flag, clkRate);
  } else {
    hipExtLaunchKernelGGL((TwoSecKernel), dim3(1), dim3(1), 0, stream1, nullptr, nullptr, conc_flag,
                          clkRate);
  }
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipMemcpyFromSymbol(&deviceGlobal_h, globalvar, sizeof(int)));

  if (conc_flag && deviceGlobal_h != 0x5555) {
    testStatus = true;
  } else if (!conc_flag && deviceGlobal_h == 0x5555) {
    testStatus = true;
  } else {
    testStatus = false;
  }
  HIP_CHECK(hipStreamDestroy(stream1));
  return testStatus;
}

bool KernelTimeExecution() {
  constexpr int FIVESEC_KERNEL = 4999;
  constexpr int THREESEC_KERNEL = 2999;
  bool testStatus = true;
  hipStream_t stream1;
  HIP_CHECK(hipSetDevice(0));
  hipEvent_t start_event1, end_event1, start_event2, end_event2;
  float time_4sec, time_2sec;
  int clkRate = 0;
#if HT_AMD
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeWallClockRate, 0));
#endif
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
#endif

  HIP_CHECK(hipEventCreate(&start_event1));
  HIP_CHECK(hipEventCreate(&end_event1));
  HIP_CHECK(hipEventCreate(&start_event2));
  HIP_CHECK(hipEventCreate(&end_event2));
  HIP_CHECK(hipStreamCreate(&stream1));
  hipDeviceProp_t props{};
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  hipExtLaunchKernelGGL((FourSecKernel), dim3(1), dim3(1), 0, stream1, start_event1, end_event1, 0,
                        clkRate);
  hipExtLaunchKernelGGL((TwoSecKernel), dim3(1), dim3(1), 0, stream1, start_event2, end_event2, 0,
                        clkRate);
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipEventElapsedTime(&time_4sec, start_event1, end_event1));
  HIP_CHECK(hipEventElapsedTime(&time_2sec, start_event2, end_event2));

  if ((time_4sec < static_cast<float>(FIVESEC_KERNEL)) &&
      (time_2sec < static_cast<float>(THREESEC_KERNEL))) {
    testStatus = true;
  } else {
    testStatus = false;
  }

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipEventDestroy(start_event1));
  HIP_CHECK(hipEventDestroy(end_event1));
  HIP_CHECK(hipEventDestroy(start_event2));
  HIP_CHECK(hipEventDestroy(end_event2));

  return testStatus;
}

TEST_CASE("Unit_hipExtLaunchKernelGGL_Functional") {
  bool testStatus = true;
// Disabled the concurency test as the firmware does not support concurrency
//   in the same stream
#if 0
    testStatus &= ConcurencyCheck_GlobalVar(0);
#endif
  SECTION("Kernel Execution Time") {
    testStatus &= KernelTimeExecution();
    REQUIRE(testStatus == true);
  }
  SECTION("Time flag Diabale") {
    testStatus &= DisableTimeFlag();
    REQUIRE(testStatus == true);
  }
}
