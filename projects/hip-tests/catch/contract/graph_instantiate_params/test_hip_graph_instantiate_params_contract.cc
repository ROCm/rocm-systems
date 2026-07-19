/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kByteCount = 64;

hipMemsetParams MakeByteMemsetParams(void* device_ptr, unsigned int value) {
  hipMemsetParams params{};
  params.dst = device_ptr;
  params.value = value;
  params.pitch = kByteCount;
  params.elementSize = sizeof(uint8_t);
  params.width = kByteCount;
  params.height = 1;
  return params;
}
}  // namespace

// @asserts: hipGraphInstantiateWithParams - instantiating a valid graph reports success, clears errNode_out, and yields a launchable exec
HIP_TEST_CASE(Contract_GraphInstantiateParams_ReportsSuccessAndLaunches) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  hipMemsetParams memset_params = MakeByteMemsetParams(device_ptr, 0x5A);
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &memset_params));

  // Instantiating a valid graph through the params-struct entry point must
  // report success, leave the error node cleared, and produce a runnable exec.
  hipGraphInstantiateParams params{};
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiateWithParams(&exec, graph, &params));
  cleanup.Add([exec] { (void)hipGraphExecDestroy(exec); });
  REQUIRE(exec != nullptr);
  REQUIRE(params.result_out == hipGraphInstantiateSuccess);
  REQUIRE(params.errNode_out == nullptr);

  // The produced executable must launch and apply the memset like any other
  // instantiated graph.
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint8_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));
  REQUIRE(host == 0x5A);
}

// @asserts: hipGraphInstantiateWithParams - the upload flag plus upload stream is accepted-or-unsupported and still yields a launchable exec
HIP_TEST_CASE(Contract_GraphInstantiateParams_UploadStream_LaunchesUploadedGraph) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipStream_t upload_stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipStreamCreate(&upload_stream));
  cleanup.Add([upload_stream] { (void)hipStreamDestroy(upload_stream); });

  hipMemsetParams memset_params = MakeByteMemsetParams(device_ptr, 0x3C);
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &memset_params));

  // Requesting upload at instantiation time (auto-upload flag plus an upload
  // stream) must still yield a successful, launchable executable. The upload is
  // an optimization; correctness of the subsequent launch is what the contract
  // guarantees.
  hipGraphInstantiateParams params{};
  params.flags = hipGraphInstantiateFlagUpload;
  params.uploadStream = upload_stream;

  hipGraphExec_t exec = nullptr;
  const hipError_t status = hipGraphInstantiateWithParams(&exec, graph, &params);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Instantiation with upload flag is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  cleanup.Add([exec] { (void)hipGraphExecDestroy(exec); });
  REQUIRE(exec != nullptr);
  REQUIRE(params.result_out == hipGraphInstantiateSuccess);

  // Drain the upload stream, then launch and verify the memset landed.
  HIP_CHECK(hipStreamSynchronize(upload_stream));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint8_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));
  REQUIRE(host == 0x3C);
}
