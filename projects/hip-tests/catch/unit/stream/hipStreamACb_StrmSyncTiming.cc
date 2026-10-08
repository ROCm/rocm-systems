/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenario :
Validate behaviour of HIP when multiple hipStreaAddCallback() are called over
multiple Threads.
*/

#include <stdio.h>
#include <atomic>
#include <chrono>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#ifdef __HIP_PLATFORM_AMD__
#define HIPRT_CB
#endif

hipStream_t mystream;
size_t N_elmts = 4096;
std::atomic<bool> cbStarted{false};
std::atomic<bool> release{false};
std::atomic<bool> cbDone{false};
std::atomic<int> Data_mismatch{0};

__global__ void vector_square(float* C_d, float* A_d, size_t N_elmts) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N_elmts; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }

  // Delay the thread 1
  if (offset == 1) {
    unsigned long long int wait_t = 3200000000, start = clock64(), cur;
    do {
      cur = clock64() - start;
    } while (cur < wait_t);
  }
}

float *A_h, *C_h;

static void HIPRT_CB Callback1(hipStream_t stream, hipError_t status, void* userData) {
  (void)stream;
  (void)status;
  (void)userData;
  cbStarted = true;

  // Validate the data
  for (size_t i = 0; i < N_elmts; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      Data_mismatch++;
    }
  }

  // Hold the callback mid-execution until the host releases it, so the host can
  // observe the stream as not-idle while the callback is blocked
  while (!release) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  cbDone = true;
}

/**
 Test multiple hipStreamAddCallback() called over
 multiple Threads.
 */
HIP_TEST_CASE(Unit_hipStreamAddCallback_StrmSyncTiming) {
  float *A_d, *C_d;
  size_t Nbytes = N_elmts * sizeof(float);

  A_h = (float*)malloc(Nbytes);
  HIPCHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (float*)malloc(Nbytes);
  HIPCHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < N_elmts; i++) {
    A_h[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));

  HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));

  HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));

  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (N_elmts + 255) / threadsPerBlock;

  hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock), 0, mystream, C_d, A_d,
                     N_elmts);
  HIPCHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mystream));
  HIPCHECK(hipStreamAddCallback(mystream, Callback1, NULL, 0));

  // Wait for callback to start; have deadline so we observe failure instead of hang
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!cbStarted && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(cbStarted);

  // Stream must be busy while the callback is mid execution
  const hipError_t busy = hipStreamQuery(mystream);
  
  release = true;
  HIPCHECK(hipStreamSynchronize(mystream));

  REQUIRE(busy == hipErrorNotReady);
  REQUIRE(cbDone);
  HIPCHECK(hipStreamQuery(mystream));
  HIPCHECK(hipStreamDestroy(mystream));
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));
  free(A_h);
  free(C_h);

  REQUIRE(Data_mismatch.load() == 0);
}
