/*
 Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <chrono>
#include <hip_test_common.hh>

using namespace std;

__global__ static void _noop_kernel() {}


TEST_CASE("Perf_KernelLaunchLatency_IncreasingNumberOfStreams") {
  vector<int> streamsNumber{1, 10, 50, 100, 1000, 5000};
  hipError_t err = hipSuccess;
  hipEvent_t start, stop;
  int32_t NITER = 100000;

  bool isBlocking = true;
  bool launchOnNullStream = true;

  SECTION("BlockingStreams, KernelLaunchOnNullStream") {
    std::cout << "BlockingStreams, LaunchOnNullStream" << std::endl;
    isBlocking = true;
    launchOnNullStream = true;
  }

  SECTION("BlockingStreams, KernelLaunchOnCreatedStream") {
    std::cout << "BlockingStreams, LaunchOnCreatedStream" << std::endl;
    isBlocking = true;
    launchOnNullStream = false;
  }

  SECTION("NonBlockingStreams, KernelLaunchOnNullStream") {
    std::cout << "NonBlockingStreams, LaunchOnNullStream" << std::endl;
    isBlocking = false;
    launchOnNullStream = true;
  }

  SECTION("NonBlockingStreams, KernelLaunchOnCreatedStream") {
    std::cout << "NonBlockingStreams, LaunchOnCreatedStream" << std::endl;
    isBlocking = false;
    launchOnNullStream = false;
  }

  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (auto& numHipStreams : streamsNumber) {
    vector<hipStream_t> streams(numHipStreams);

    if (isBlocking) {
      for (int i = 0; i < numHipStreams; ++i) {
        HIP_CHECK(hipStreamCreate(&streams[i]));
      }
    } else {
      for (int i = 0; i < numHipStreams; ++i) {
        HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));
      }
    }

    auto kernelStart = std::chrono::steady_clock::now();
    for (int i = 0; i < NITER; ++i) {
      if (launchOnNullStream) {
        HIP_CHECK(hipEventRecord(start, NULL));
        _noop_kernel<<<1, 1>>>();
        HIP_CHECK(hipEventRecord(stop, NULL));
      } else {
        HIP_CHECK(hipEventRecord(start, streams[0]));
        _noop_kernel<<<1, 1, 0, streams[0]>>>();
        HIP_CHECK(hipEventRecord(stop, streams[0]));
      }
      do {
        err = hipEventQuery(stop);
      } while (err == hipErrorNotReady);
    }
    auto kernelStop = std::chrono::steady_clock::now();
    double usec = std::chrono::duration<double, std::micro>(kernelStop - kernelStart).count();

    cout << "hipLaunchKernel average duration with " << numHipStreams
         << " streams: " << usec / NITER << " us" << std::endl;

    HIP_CHECK(hipDeviceSynchronize());

    for (int i = 0; i < numHipStreams; ++i) {
      HIP_CHECK(hipStreamDestroy(streams[i]));
    }
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}
