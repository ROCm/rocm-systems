/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

// The RPC interface is only available when the compiler ships the LLVM libc
// shared headers.
#if __has_include(<shared/rpc.h>)

#include <shared/rpc.h>
#include <shared/rpc_opcodes.h>

namespace rpc {
[[gnu::visibility("protected")]] __device__ Client client asm("__llvm_rpc_client");
}  // namespace rpc

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

TEST_CASE("Unit_Rpc_Increment_MultiThread") {
  constexpr uint32_t kBlocks = 2;
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kTotal = kBlocks * kThreads;
  constexpr uint32_t kIters = 64;

  int* results = nullptr;
  HIP_CHECK(hipMallocManaged(&results, kTotal * sizeof(int)));
  for (uint32_t i = 0; i < kTotal; ++i) results[i] = -1;

  rpcIncrementKernel<<<kBlocks, kThreads>>>(results, kIters);
  HIP_CHECK(hipDeviceSynchronize());

  for (uint32_t i = 0; i < kTotal; ++i) {
    INFO("thread " << i);
    REQUIRE(results[i] == 1);
  }
  HIP_CHECK(hipFree(results));
}

TEST_CASE("Unit_Rpc_Noop_Divergent") {
  rpcNoopKernel<<<2, 64>>>();
  HIP_CHECK(hipDeviceSynchronize());
}

#endif  // __has_include(<shared/rpc.h>)
