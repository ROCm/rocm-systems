/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>
#include <resource_guards.hh>

namespace {
inline constexpr size_t kLaunchIters = 10;
}  // anonymous namespace

template <typename T> void captureSequenceSimple(T* hostMem1, T* devMem1, T* hostMem2, size_t N,
                                                 hipStream_t captureStream) {
  size_t Nbytes = N * sizeof(T);

  HIP_CHECK(hipMemsetAsync(devMem1, 0, Nbytes, captureStream));
  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem1, Nbytes, hipMemcpyHostToDevice, captureStream));
  HIP_CHECK(hipMemcpyAsync(hostMem2, devMem1, Nbytes, hipMemcpyDeviceToHost, captureStream));
}

template <typename T> void captureSequenceLinear(T* hostMem1, T* devMem1, T* hostMem2, T* devMem2,
                                                 size_t N, hipStream_t captureStream) {
  {
    (void)(hostMem2);
  }  // unused hostMem2
  size_t Nbytes = N * sizeof(T);

  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem1, Nbytes, hipMemcpyHostToDevice, captureStream));

  HIP_CHECK(hipMemsetAsync(devMem2, 0, Nbytes, captureStream));
}

template <typename T> void captureSequenceBranched(T* hostMem1, T* devMem1, T* hostMem2, T* devMem2,
                                                   size_t N, hipStream_t captureStream,
                                                   std::vector<hipStream_t>& streams,
                                                   std::vector<hipEvent_t>& events) {
  {
    (void)(hostMem2);
  }  // unused hostMem2
  size_t Nbytes = N * sizeof(T);

  HIP_CHECK(hipEventRecord(events[0], captureStream));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[0], 0));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(hipMemsetAsync(devMem1, 0, Nbytes, streams[0]));
  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem1, Nbytes, hipMemcpyHostToDevice, streams[0]));
  HIP_CHECK(hipEventRecord(events[1], streams[0]));
  HIP_CHECK(hipMemsetAsync(devMem2, 0, Nbytes, streams[1]));
  HIP_CHECK(hipEventRecord(events[2], streams[1]));
  HIP_CHECK(hipStreamWaitEvent(captureStream, events[1], 0));
  HIP_CHECK(hipStreamWaitEvent(captureStream, events[2], 0));
}

template <typename T>
void captureSequenceCompute(T* devMem1, T* hostMem2, T* devMem2, size_t N, hipStream_t stream) {
  size_t Nbytes = N * sizeof(T);
  constexpr unsigned threadsPerBlock = 256;
  const unsigned blocks =
      (N % threadsPerBlock == 0) ? (N / threadsPerBlock) : ((N / threadsPerBlock) + 1);

  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                     devMem1, devMem2, N);

  HIP_CHECK(hipMemcpyAsync(hostMem2, devMem2, Nbytes, hipMemcpyDeviceToHost, stream));
}
