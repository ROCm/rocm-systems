/*
 Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

/**
 * @addtogroup hipPerfDispatchSpeed hipPerfDispatchSpeed
 * @{
 * @ingroup perfDispatchTest
 */

// #define ENABLE_DEBUG 1

#include <hip_test_common.hh>
#include <string.h>
#include <complex>

/**
 * Test Description
 * ------------------------
 *  - Verify the hipPerf Dispatch and Execution speed, AKA total kernel latency
 * Test source
 * ------------------------
 *  - perftests/dispatch/hipPerfDispatchSpeed.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

unsigned int testList[] = {1, 10, 100, 1000, 10000};

// dummy kernel that just dispatches and does nothing
__global__ void _dispatchSpeed(float* outBuf) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i < 0) outBuf[i] = 0.0f;
};

// kernel that has an execution of count, in GPU clock ticks
__global__ void _TimingKernel(uint64_t count) {
  uint64_t begin_time = __builtin_amdgcn_s_memrealtime();
  uint64_t curr_time = begin_time;
  do {
    curr_time = __builtin_amdgcn_s_memrealtime();
  } while (begin_time + count > curr_time);
}

enum TimingMode { TimingMode_WallTime, TimingMode_HIPEvent, TimingMode_NumModes };

TEST_CASE("Perf_hipPerfDispatchAndExecutionSpeed") {
  hipError_t err = hipSuccess;

  unsigned int testListSize = sizeof(testList) / sizeof(testList[0]);
  int numTests = testListSize;
  int warmup = 10;  // number of warmup iterations

  DEBUG_PRINT("numTests %d", numTests);

  // set up timing kernel
  uint64_t timer_freq_in_hz;
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeWallClockRate, 0));
  timer_freq_in_hz = clock_rate * 1000;
  uint64_t timing_in_us = 4;  // CHANGE THIS TO CHANGE EXECUTION TIME
  const uint64_t timing_count = timer_freq_in_hz * timing_in_us / 1000000;

  int iterations = 100;  // number of times to run the test to get an average time

  float* srcBuffer = NULL;
  unsigned int bufSize_ = 64 * sizeof(float);
  err = hipMalloc(&srcBuffer, bufSize_);
  REQUIRE(err == hipSuccess);

  hipEvent_t startEvent, stopEvent;

  HIP_CHECK(hipEventCreate(&startEvent));
  HIP_CHECK(hipEventCreate(&stopEvent));


  // run twice for both dispatch speed and full kernel latency
  for (int j = 0; j < 2; j++) {
    bool useTimingKernel = (j == 1);
    if (useTimingKernel) {
      CONSOLE_PRINT("\nTIMING KERNEL TEST ()");
      CONSOLE_PRINT("--------------------------------------------------------------");

    } else {
      CONSOLE_PRINT("EMPTY KERNEL TEST");
      CONSOLE_PRINT("--------------------------------------------------------------");
    }


    // loop through all possible timing methods
    for (unsigned int i = 0; i < TimingMode_NumModes; i++) {
      TimingMode mode = static_cast<TimingMode>(i);
      CONSOLE_PRINT("\nTIMING METHOD:");

      switch (mode) {
        case TimingMode_WallTime:
          CONSOLE_PRINT("Wall Time");
          break;
        case TimingMode_HIPEvent:
          CONSOLE_PRINT("HIP Events");
          break;
        default:
          CONSOLE_PRINT("Unknown Mode");
      }

      // go through test iterations
      for (int test = 0; test < numTests; test++) {
        int openTest = test % testListSize;

        int threads = (bufSize_ / sizeof(float));
        int threads_per_block = 64;
        int blocks = (threads / threads_per_block) + (threads % threads_per_block);
        double finalPerf = 0.0;
        double wallMicroSec = 0.0;

        std::chrono::high_resolution_clock::time_point startWall, stopWall;

        // warmup
        for (int i = 0; i < warmup; i++) {
          hipLaunchKernelGGL(_TimingKernel, dim3(blocks), dim3(threads_per_block), 0,
                             hipStream_t(0), timing_count);
        }
        HIP_CHECK(hipStreamSynchronize(0));

        for (int it = 0; it < iterations; it++) {
          switch (mode) {
            case TimingMode_WallTime:
              startWall = std::chrono::high_resolution_clock::now();
              break;
            case TimingMode_HIPEvent:
              HIP_CHECK(hipEventRecord(startEvent, 0));
              break;
            default:
              CONSOLE_PRINT("Unknown Mode");
          }

          for (unsigned int i = 0; i < testList[openTest]; i++) {
            if (useTimingKernel) {
              // use the timing kernel to measure dispatch and execution speed
              hipLaunchKernelGGL(_TimingKernel, dim3(blocks), dim3(threads_per_block), 0,
                                 hipStream_t(0), timing_count);
            } else {
              // use the dispatch speed kernel
              hipLaunchKernelGGL(_dispatchSpeed, dim3(blocks), dim3(threads_per_block), 0,
                                 hipStream_t(0), srcBuffer);
            }
          }

          switch (mode) {
            case TimingMode_WallTime: {
              err = hipStreamSynchronize(0);
              REQUIRE(err == hipSuccess);
              stopWall = std::chrono::high_resolution_clock::now();
              wallMicroSec =
                  std::chrono::duration<double, std::micro>(stopWall - startWall).count();
              finalPerf += wallMicroSec / testList[openTest];
              break;
            }
            case TimingMode_HIPEvent: {
              HIP_CHECK(hipEventRecord(stopEvent, 0));
              HIP_CHECK(hipEventSynchronize(stopEvent));
              float elapsed;
              HIP_CHECK(hipEventElapsedTime(&elapsed, startEvent, stopEvent));  // in milliseconds
              finalPerf += (elapsed * 1000.0f) / testList[openTest];            // convert ms to µs
              break;
            }
            default:
              CONSOLE_PRINT("Unknown Mode");
          }
        }

        finalPerf /= iterations;  // average the performance over all iterations


        CONSOLE_PRINT("HIPPerfDispatchSpeed[%3d] %7d dispatches              (us/disp) %3f", test,
                      testList[openTest], (float)finalPerf);
      }
    }
  }

  HIP_CHECK(hipEventDestroy(startEvent));
  HIP_CHECK(hipEventDestroy(stopEvent));

  HIP_CHECK(hipFree(srcBuffer));
}


/**
 * End doxygen group perfDispatchTest.
 * @}
 */
