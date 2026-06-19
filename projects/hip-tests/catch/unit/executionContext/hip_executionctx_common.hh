/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <cstdlib>

inline hipError_t GetSmResourceDesc(hipDevResourceDesc_t* desc) {
  hipDevice_t device;
  hipDevResource resource{};

  hipError_t ret = hipDeviceGet(&device, 0);
  if (ret != hipSuccess) {
    return ret;
  }

  ret = hipDeviceGetDevResource(device, &resource, hipDevResourceTypeSm);
  if (ret != hipSuccess) {
    return ret;
  }

  return hipDevResourceGenerateDesc(desc, &resource, 1);
}

// Convenience helper used by the detach tests: builds a green execution
// context from the device's full SM resource and creates a single
// non-blocking stream attached to it. Both handles are returned via
// out-parameters so callers can drive their own teardown (the typical
// pattern being: hipExecutionCtxDestroy(ctx) to detach the stream, then
// hipStreamDestroy(stream) to release it).
inline void MakeCtxAndStream(hipExecutionCtx_t& ctx, hipStream_t& stream) {
  HIP_CHECK(hipSetDevice(0));

  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0));
  REQUIRE(stream != nullptr);
}

// Creates a green context and stream from the given SM resource, runs a
// vectorADD kernel (C = A + B), and verifies the result on the host.
// The context, stream, and all allocations are cleaned up before returning.
inline void RunVectorAddOnResource(hipDevResource* resource, int device) {
  hipDevResourceDesc_t desc{};
  HIP_CHECK(hipDevResourceGenerateDesc(&desc, resource, 1));

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, device, 0));
  REQUIRE(ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0));
  REQUIRE(stream != nullptr);

  constexpr size_t kN = 1024;
  const size_t kBytes = kN * sizeof(int);

  int* h_a = reinterpret_cast<int*>(malloc(kBytes));
  int* h_b = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c = reinterpret_cast<int*>(malloc(kBytes));
  REQUIRE(h_a != nullptr);
  REQUIRE(h_b != nullptr);
  REQUIRE(h_c != nullptr);

  for (size_t i = 0; i < kN; i++) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(i * 2);
    h_c[i] = 0;
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c, kBytes));

  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, stream));

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, stream>>>(d_a, d_b, d_c, kN);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpyAsync(h_c, d_c, kBytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < kN; i++) {
    REQUIRE(h_c[i] == h_a[i] + h_b[i]);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}
