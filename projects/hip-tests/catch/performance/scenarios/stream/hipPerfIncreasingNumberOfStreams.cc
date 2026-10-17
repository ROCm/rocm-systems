/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <chrono>
#include <hip_test_common.hh>

#include "stream_create_helper.hh"

using namespace std;

__global__ static void _noop_kernel() {}


HIP_TEST_CASE(Performance_KernelLaunchLatency_IncreasingNumberOfStreams) {
  // 5000 is deliberately not in this list. No part tested can create that many streams -- the
  // ROCr path stops at ~4000 and PAL runs out of device memory well before -- and merely
  // attempting it is not a clean failure: the stream that fails leaves the process in a state
  // where later work dies with an illegal memory access (ROCr) or every subsequent HIP call
  // returns hipErrorLaunchFailure (PAL, see stream_create_helper.hh). It never produced a
  // measurement, so asking for it only costs the rest of the run.
  vector<size_t> streamsNumber{1, 10, 50, 100, 1000};
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
    vector<hipStream_t> streams;
    streams.reserve(numHipStreams);

    // How many streams a device can hold is a device property, not a scenario defect: report
    // and skip rather than fail the run.
    if (!stream_test::CreateStreamsOrReport(numHipStreams, !isBlocking, streams)) {
      HIP_CHECK(hipEventDestroy(start));
      HIP_CHECK(hipEventDestroy(stop));
      SKIP("Device cannot provide " << numHipStreams << " streams");
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

    for (hipStream_t s : streams) {
      HIP_CHECK(hipStreamDestroy(s));
    }
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}
