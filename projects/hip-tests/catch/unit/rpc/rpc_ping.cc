/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rpc_test_common.hh"

#include <vector>

#if HIP_TEST_HAS_RPC

__global__ void rpcIncrementKernel(int* results, uint32_t iters) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t cnt = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    rpc::Client::Port port = rpc::client.open<LIBC_TEST_INCREMENT>();
    port.send_and_recv([=](rpc::Buffer* buf, uint32_t) { buf->data[0] = cnt; },
                       [&](rpc::Buffer* buf, uint32_t) { cnt = buf->data[0]; });
  }
  results[tid] = (cnt == iters) ? 1 : 0;
}

__global__ void rpcNoopKernel() {
  if ((blockIdx.x * blockDim.x + threadIdx.x) % 2)
    rpc::client.open<LIBC_NOOP>().send([](rpc::Buffer* buf, uint32_t) { buf->data[0] = 1; });
  else
    rpc::client.open<LIBC_NOOP>().send([](rpc::Buffer* buf, uint32_t) { buf->data[0] = 2; });
}

static void requireRpcRoundTrip() {
  int* result = nullptr;
  HIP_CHECK(hipMalloc(&result, sizeof(*result)));
  HIP_CHECK(hipMemset(result, 0, sizeof(*result)));
  rpcIncrementKernel<<<1, 1>>>(result, 1);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int host_result = 0;
  HIP_CHECK(hipMemcpy(&host_result, result, sizeof(host_result), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(result));
  REQUIRE(host_result == 1);
}

TEST_CASE("Unit_Rpc_Increment_MultiThread") {
  requireRpcSupport();

  constexpr uint32_t kBlocks = 2;
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kTotal = kBlocks * kThreads;
  constexpr uint32_t kIters = 64;

  int* device_results = nullptr;
  HIP_CHECK(hipMalloc(&device_results, kTotal * sizeof(int)));
  HIP_CHECK(hipMemset(device_results, 0xff, kTotal * sizeof(int)));

  rpcIncrementKernel<<<kBlocks, kThreads>>>(device_results, kIters);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> results(kTotal);
  HIP_CHECK(hipMemcpy(results.data(), device_results, kTotal * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_results));
  for (uint32_t i = 0; i < kTotal; ++i) {
    INFO("thread " << i);
    REQUIRE(results[i] == 1);
  }
}

TEST_CASE("Unit_Rpc_Increment_MultiDevice") {
  requireRpcSupport();

  int original_device = 0;
  int device_count = 0;
  HIP_CHECK(hipGetDevice(&original_device));
  HIP_CHECK(hipGetDeviceCount(&device_count));
  std::vector<int*> results(device_count);

  for (int device = 0; device < device_count; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipMalloc(&results[device], sizeof(*results[device])));
    HIP_CHECK(hipMemset(results[device], 0, sizeof(*results[device])));
    rpcIncrementKernel<<<1, 1>>>(results[device], 1);
    HIP_CHECK(hipGetLastError());
  }
  for (int device = 0; device < device_count; ++device) {
    HIP_CHECK(hipSetDevice(device));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, results[device], sizeof(result), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(results[device]));
    REQUIRE(result == 1);
  }
  HIP_CHECK(hipSetDevice(original_device));
}

TEST_CASE("Unit_Rpc_Noop_Divergent") {
  requireRpcSupport();

  rpcNoopKernel<<<2, 64>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

TEST_CASE("Unit_Rpc_DeviceReset") {
  requireRpcSupport();

  requireRpcRoundTrip();
  HIP_CHECK(hipDeviceReset());

  requireRpcSupport();
  requireRpcRoundTrip();
}

#endif  // HIP_TEST_HAS_RPC
