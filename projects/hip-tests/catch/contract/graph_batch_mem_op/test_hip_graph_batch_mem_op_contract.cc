/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr uint32_t kWriteValue = 0xABCD1234u;

// Batch memory operations build on the stream wait/write value capability,
// exposed through hipDeviceAttributeCanUseStreamWaitValue. Where that capability
// is absent the batch-mem-op graph node family is unsupported, so the test skips
// rather than reporting a contract violation.
void RequireStreamWaitValueSupport() {
  int device = -1;
  HIP_CHECK(hipGetDevice(&device));
  int can_use = 0;
  HIP_CHECK(hipDeviceGetAttribute(&can_use, hipDeviceAttributeCanUseStreamWaitValue, device));
  if (can_use == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }
}

hipCtx_t CurrentContext() {
  hipCtx_t ctx = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&ctx));
  REQUIRE(ctx != nullptr);
  return ctx;
}

// Builds a single write-value-32 batch operation targeting the given device
// address.
hipStreamBatchMemOpParams WriteValueOp(hipDeviceptr_t address, uint32_t value) {
  hipStreamBatchMemOpParams op{};
  op.writeValue.operation = hipStreamMemOpWriteValue32;
  op.writeValue.address = address;
  op.writeValue.value = value;
  op.writeValue.flags = 0;
  return op;
}

hipBatchMemOpNodeParams MakeNodeParams(hipStreamBatchMemOpParams* op_array, unsigned int count) {
  hipBatchMemOpNodeParams params{};
  params.ctx = CurrentContext();
  params.count = count;
  params.paramArray = op_array;
  params.flags = 0;
  return params;
}
}  // namespace

// @asserts: hipGraphAddBatchMemOpNode - launching a graph with a write-value-32 batch-mem-op node writes the value to the target address
HIP_TEST_CASE(Contract_GraphBatchMemOp_AddNode_LaunchesWriteValue) {
  RequireStreamWaitValueSupport();
  hip::contract::ContractCleanup cleanup;

  uint32_t* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, sizeof(uint32_t)));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipMemset(device_ptr, 0, sizeof(uint32_t)));

  hipStreamBatchMemOpParams op =
      WriteValueOp(reinterpret_cast<hipDeviceptr_t>(device_ptr), kWriteValue);
  hipBatchMemOpNodeParams node_params = MakeNodeParams(&op, 1);

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  // A batch-mem-op node containing a single write-value-32 operation must, when
  // the graph is launched, write the value to the target device address.
  const hipError_t add_status = hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &node_params);
  if (add_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memory operation graph nodes are not supported by this runtime path.");
  }
  HIP_CHECK(add_status);

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cleanup.Add([exec] { (void)hipGraphExecDestroy(exec); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));
  REQUIRE(host == kWriteValue);
}

// @asserts: hipGraphBatchMemOpNodeGetParams - reports back the operation count the batch-mem-op node was created with
HIP_TEST_CASE(Contract_GraphBatchMemOp_GetParams_RoundTripsCount) {
  RequireStreamWaitValueSupport();
  hip::contract::ContractCleanup cleanup;

  uint32_t* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, sizeof(uint32_t)));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  hipStreamBatchMemOpParams op =
      WriteValueOp(reinterpret_cast<hipDeviceptr_t>(device_ptr), kWriteValue);
  hipBatchMemOpNodeParams node_params = MakeNodeParams(&op, 1);

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  const hipError_t add_status = hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &node_params);
  if (add_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memory operation graph nodes are not supported by this runtime path.");
  }
  HIP_CHECK(add_status);

  // The node getter reports the operation count the node was created with.
  hipBatchMemOpNodeParams retrieved{};
  HIP_CHECK(hipGraphBatchMemOpNodeGetParams(node, &retrieved));
  REQUIRE(retrieved.count == 1);
}

// @asserts: hipGraphBatchMemOpNodeSetParams - re-parameterizing a node before instantiate makes the launch write the updated value
HIP_TEST_CASE(Contract_GraphBatchMemOp_SetParams_UpdatesWriteValueBeforeInstantiate) {
  RequireStreamWaitValueSupport();
  hip::contract::ContractCleanup cleanup;

  uint32_t* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, sizeof(uint32_t)));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipMemset(device_ptr, 0, sizeof(uint32_t)));

  hipStreamBatchMemOpParams initial_op =
      WriteValueOp(reinterpret_cast<hipDeviceptr_t>(device_ptr), 0x22222222u);
  hipBatchMemOpNodeParams initial_params = MakeNodeParams(&initial_op, 1);

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  const hipError_t add_status =
      hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &initial_params);
  if (add_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memory operation graph nodes are not supported by this runtime path.");
  }
  HIP_CHECK(add_status);

  // Re-parameterize the node with a different write value through the
  // pre-instantiation graph-node setter. The launched graph must reflect the
  // updated value, proving the setter actually replaces the node's operation
  // rather than being a no-op.
  hipStreamBatchMemOpParams updated_op =
      WriteValueOp(reinterpret_cast<hipDeviceptr_t>(device_ptr), kWriteValue);
  hipBatchMemOpNodeParams updated_params = MakeNodeParams(&updated_op, 1);
  HIP_CHECK(hipGraphBatchMemOpNodeSetParams(node, &updated_params));

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cleanup.Add([exec] { (void)hipGraphExecDestroy(exec); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));
  REQUIRE(host == kWriteValue);
}

// @asserts: hipGraphExecBatchMemOpNodeSetParams - updating an instantiated node's write value takes effect on the next launch without re-instantiation
HIP_TEST_CASE(Contract_GraphBatchMemOp_ExecSetParams_UpdatesWriteValueAfterInstantiate) {
  RequireStreamWaitValueSupport();
  hip::contract::ContractCleanup cleanup;

  uint32_t* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, sizeof(uint32_t)));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipMemset(device_ptr, 0, sizeof(uint32_t)));

  hipStreamBatchMemOpParams initial_op =
      WriteValueOp(reinterpret_cast<hipDeviceptr_t>(device_ptr), 0x11111111u);
  hipBatchMemOpNodeParams initial_params = MakeNodeParams(&initial_op, 1);

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  const hipError_t add_status =
      hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &initial_params);
  if (add_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memory operation graph nodes are not supported by this runtime path.");
  }
  HIP_CHECK(add_status);

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cleanup.Add([exec] { (void)hipGraphExecDestroy(exec); });

  // Re-parameterize the instantiated node with a new write value through the
  // executable setter. The next launch must write the updated value without a
  // re-instantiation.
  hipStreamBatchMemOpParams updated_op =
      WriteValueOp(reinterpret_cast<hipDeviceptr_t>(device_ptr), kWriteValue);
  hipBatchMemOpNodeParams updated_params = MakeNodeParams(&updated_op, 1);
  HIP_CHECK(hipGraphExecBatchMemOpNodeSetParams(exec, node, &updated_params));

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));
  REQUIRE(host == kWriteValue);
}
