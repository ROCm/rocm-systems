/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>
#include <resource_guards.hh>

namespace {
inline constexpr size_t kLaunchIters = 10;
}  // anonymous namespace

template <typename T>
void captureSequenceSimple(T* hostMem1, T* devMem1, T* hostMem2, size_t N,
                           hipStream_t captureStream) {
  size_t Nbytes = N * sizeof(T);

  HIP_CHECK(hipMemsetAsync(devMem1, 0, Nbytes, captureStream));
  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem1, Nbytes, hipMemcpyHostToDevice, captureStream));
  HIP_CHECK(hipMemcpyAsync(hostMem2, devMem1, Nbytes, hipMemcpyDeviceToHost, captureStream));
}

template <typename T>
void captureSequenceLinear(T* hostMem1, T* devMem1, T* hostMem2, T* devMem2, size_t N,
                           hipStream_t captureStream) {

  {(void)(hostMem2);} //unused hostMem2
  size_t Nbytes = N * sizeof(T);

  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem1, Nbytes, hipMemcpyHostToDevice, captureStream));

  HIP_CHECK(hipMemsetAsync(devMem2, 0, Nbytes, captureStream));
}

template <typename T>
void captureSequenceBranched(T* hostMem1, T* devMem1, T* hostMem2, T* devMem2, size_t N,
                             hipStream_t captureStream, std::vector<hipStream_t>& streams,
                             std::vector<hipEvent_t>& events) {
  {(void)(hostMem2);} //unused hostMem2
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
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;

  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                     devMem1, devMem2, N);

  HIP_CHECK(hipMemcpyAsync(hostMem2, devMem2, Nbytes, hipMemcpyDeviceToHost, stream));
}
