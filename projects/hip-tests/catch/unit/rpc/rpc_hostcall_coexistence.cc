/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rpc_test_common.hh"

#if HIP_TEST_HAS_RPC

extern "C" __device__ unsigned long long __ockl_devmem_request(unsigned long long address,
                                                               unsigned long long size);

__global__ void rpcHostcallKernel(int* results) {
  if (blockIdx.x == 0) {
    unsigned long long address = __ockl_devmem_request(0, 64);
    if (address != 0) {
      __ockl_devmem_request(address, 0);
      results[0] = 1;
    }
    return;
  }

  uint64_t value = 0;
  rpc::Client::Port port = rpc::client.open<LIBC_TEST_INCREMENT>();
  port.send_and_recv([&](rpc::Buffer* buf, uint32_t) { buf->data[0] = value; },
                     [&](rpc::Buffer* buf, uint32_t) { value = buf->data[0]; });
  results[1] = value == 1;
}

TEST_CASE("Unit_Rpc_Hostcall_Coexistence") {
  requireRpcSupport();

  int* results = nullptr;
  HIP_CHECK(hipMalloc(&results, 2 * sizeof(*results)));
  HIP_CHECK(hipMemset(results, 0, 2 * sizeof(*results)));
  rpcHostcallKernel<<<2, 1>>>(results);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int host_results[2] = {};
  HIP_CHECK(hipMemcpy(host_results, results, sizeof(host_results), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(results));
  REQUIRE(host_results[0] == 1);
  REQUIRE(host_results[1] == 1);
}

#endif  // HIP_TEST_HAS_RPC
