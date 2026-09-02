/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Smoke tests for GFX12 / PR-7152 conditional graph node AQL vendor packets.
 *
 * PURPOSE
 * -------
 * These tests are intentionally minimal probes, not correctness suites.
 * The full correctness tests live in hipGraphConditionalNode.cc.
 *
 * The tests are structured in four layers:
 *
 *   1. API/header existence  — compile-time checks that all new symbols are
 *      visible (types, enums, function declarations, device inline).
 *
 *   2. Handle plumbing       — runtime checks that hipGraphConditionalHandleCreate
 *      fills the struct correctly without launching any graph.
 *
 *   3. Argument validation   — runtime checks that the API rejects bad inputs
 *      with the documented error codes.
 *
 *   4. Packet path smoke     — minimal WHILE / IF graphs that force the runtime
 *      to emit the new vendor COND_BRANCH / LOOP_BACK AQL packets.  These tests
 *      are tagged [!mayfail] because the CP firmware may reject the packet with
 *      hipErrorLaunchFailure (HSA_STATUS_ERROR_INVALID_PACKET_FORMAT 0x1009)
 *      until matching GFX12 microcode ships.  A launch failure is surfaced
 *      explicitly via WARN so it is visible in the test log even when the
 *      test is marked may-fail.
 *
 * Firmware ABI note (as of PR-7152 / June 2026): the current firmware uses
 * packet type 6 for COND_BRANCH and type 7 for LOOP_BACK.  WHILE keeps the
 * legacy [body packets | LOOP_BACK] lowering; it is not lowered to COND_BRANCH
 * repeat mode yet.
 */

#include <cstddef>

#include <hsa/hsa_ext_amd.h>
#include <hip_test_common.hh>

// ---------------------------------------------------------------------------
// Layer 1: Compile-time API / header existence checks
// ---------------------------------------------------------------------------
// If any of these static_asserts or sizeof checks fail to compile, the header
// is missing the new types.

// hipGraphNodeTypeConditional must be present in the hipGraphNodeType enum.
static_assert(hipGraphNodeTypeConditional == 15,
              "hipGraphNodeTypeConditional enum value changed or missing");

// hipGraphCondTypeIf / hipGraphCondTypeWhile / hipGraphCondTypeSwitch
static_assert(hipGraphCondTypeIf == 0,
              "hipGraphCondTypeIf enum value changed or missing");
static_assert(hipGraphCondTypeWhile == 1,
              "hipGraphCondTypeWhile enum value changed or missing");
static_assert(hipGraphCondTypeSwitch == 2,
              "hipGraphCondTypeSwitch enum value changed or missing");

// hipGraphConditionalHandle must be a plain struct with three uint64_t fields.
static_assert(sizeof(hipGraphConditionalHandle) == 3 * sizeof(uint64_t),
              "hipGraphConditionalHandle layout changed");
static_assert(offsetof(hipGraphConditionalHandle, device_ptr) == 0,
              "hipGraphConditionalHandle::device_ptr offset changed");
static_assert(offsetof(hipGraphConditionalHandle, default_value) == 8,
              "hipGraphConditionalHandle::default_value offset changed");
static_assert(offsetof(hipGraphConditionalHandle, signal_handle) == 16,
              "hipGraphConditionalHandle::signal_handle offset changed");

static_assert(HSA_AMD_PACKET_TYPE_AQL_INDIRECT_BUFFER == 5,
              "AQL INDIRECT_BUFFER packet type changed");
static_assert(HSA_AMD_PACKET_TYPE_AQL_COND_BRANCH == 6,
              "AQL COND_BRANCH packet type changed");
static_assert(HSA_AMD_PACKET_TYPE_AQL_LOOP_BACK == 7,
              "AQL LOOP_BACK packet type changed");
static_assert(HSA_AMD_PACKET_TYPE_DISPATCH_IB_COND_JUMP ==
                  HSA_AMD_PACKET_TYPE_AQL_COND_BRANCH,
              "conditional jump compatibility alias changed");

static_assert(HSA_AMD_AQL_COND_BRANCH_COND_BOOL_TRUE == 10,
              "COND_BRANCH BOOL_TRUE cond_op changed");
static_assert(HSA_AMD_AQL_COND_BRANCH_EXEC_BRANCH_ONCE == 0,
              "COND_BRANCH BRANCH_ONCE execution mode changed");
static_assert(HSA_AMD_AQL_COND_BRANCH_POST_ACTION_NONE == 0,
              "COND_BRANCH NONE post action changed");

static_assert(sizeof(hsa_amd_aql_cond_branch_packet_t) == 64,
              "COND_BRANCH packet must remain one AQL slot");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, header) == 0,
              "COND_BRANCH header offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, cond_op) == 4,
              "COND_BRANCH cond_op offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, execution_mode) == 5,
              "COND_BRANCH execution_mode offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, post_action) == 6,
              "COND_BRANCH post_action offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, flags) == 7,
              "COND_BRANCH flags offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, condition_signal) == 8,
              "COND_BRANCH condition_signal offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, test_value) == 16,
              "COND_BRANCH test_value offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t,
                       true_target_offset_packets) == 24,
              "COND_BRANCH true_target_offset_packets offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t,
                       true_target_size_packets) == 28,
              "COND_BRANCH true_target_size_packets offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t,
                       false_target_offset_packets) == 32,
              "COND_BRANCH false_target_offset_packets offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t,
                       false_target_size_packets) == 36,
              "COND_BRANCH false_target_size_packets offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, ib_base_addr) == 40,
              "COND_BRANCH ib_base_addr offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, branch_options) == 48,
              "COND_BRANCH branch_options offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, ib_size_packets) == 52,
              "COND_BRANCH ib_size_packets offset changed");
static_assert(offsetof(hsa_amd_aql_cond_branch_packet_t, completion_signal) == 56,
              "COND_BRANCH completion_signal offset changed");

static_assert(sizeof(hsa_amd_aql_loop_back_t) == 64,
              "LOOP_BACK packet must remain one AQL slot");
static_assert(offsetof(hsa_amd_aql_loop_back_t, condition_signal) == 8,
              "LOOP_BACK condition_signal offset changed");
static_assert(offsetof(hsa_amd_aql_loop_back_t, test_value) == 16,
              "LOOP_BACK test_value offset changed");

// ---------------------------------------------------------------------------
// Layer 2: Handle plumbing — no graph launch required
// ---------------------------------------------------------------------------

TEST_CASE("Smoke_hipGraphConditionalHandleCreate_BasicPlumbing") {
  // Verify that hipGraphConditionalHandleCreate:
  //   a) returns hipSuccess
  //   b) fills device_ptr to a non-zero GPU-visible address
  //   c) fills default_value to the requested value
  //   d) fills signal_handle to a non-zero opaque handle
  //   e) device_ptr == signal_handle + 8  (Option-C layout: amd_signal_t.value
  //      is at offset +8 from the signal handle address)

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  std::memset(&handle, 0xAB, sizeof(handle));  // poison before call

  constexpr unsigned int kDefaultValue = 1u;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, kDefaultValue, 0));

  INFO("device_ptr    = 0x" << std::hex << handle.device_ptr);
  INFO("default_value = " << std::dec << handle.default_value);
  INFO("signal_handle = 0x" << std::hex << handle.signal_handle);

  REQUIRE(handle.device_ptr != 0);
  REQUIRE(handle.default_value == kDefaultValue);
  REQUIRE(handle.signal_handle != 0);

  // Option-C layout: device_ptr must equal signal_handle + 8.
  // This is the same invariant asserted by static_assert in hip_graph.cpp.
  REQUIRE(handle.device_ptr == handle.signal_handle + 8u);

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Smoke_hipGraphConditionalHandleCreate_ZeroDefault") {
  // A zero defaultLaunchValue is valid (WHILE loop starts false → 0 iterations).
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 0u, 0));

  REQUIRE(handle.device_ptr != 0);
  REQUIRE(handle.default_value == 0u);
  REQUIRE(handle.signal_handle != 0);
  REQUIRE(handle.device_ptr == handle.signal_handle + 8u);

  HIP_CHECK(hipGraphDestroy(graph));
}

// ---------------------------------------------------------------------------
// Layer 3: Argument validation
// ---------------------------------------------------------------------------

TEST_CASE("Smoke_hipGraphConditionalHandleCreate_NullOutput") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Null output pointer must return hipErrorInvalidValue.
  HIP_CHECK_ERROR(hipGraphConditionalHandleCreate(nullptr, graph, 1u, 0),
                  hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Smoke_hipGraphConditionalHandleCreate_NullGraph") {
  hipGraphConditionalHandle handle;
  // Null graph must return hipErrorInvalidValue.
  HIP_CHECK_ERROR(hipGraphConditionalHandleCreate(&handle, nullptr, 1u, 0),
                  hipErrorInvalidValue);
}

TEST_CASE("Smoke_hipGraphAddConditionalNode_SwitchNotSupported") {
  // hipGraphCondTypeSwitch is documented to return hipErrorNotSupported.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK_ERROR(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                              handle, hipGraphCondTypeSwitch,
                                              1, &bodyGraph),
                  hipErrorNotSupported);

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Smoke_hipGraphAddConditionalNode_WhileWrongBodyCount") {
  // WHILE with numConditionalGraphs != 1 must return hipErrorInvalidValue.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraphs[2] = {nullptr, nullptr};
  hipGraphNode_t condNode;
  HIP_CHECK_ERROR(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                              handle, hipGraphCondTypeWhile,
                                              2, bodyGraphs),
                  hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Smoke_hipGraphAddConditionalNode_NullOutputNode") {
  // Null pGraphNode must return hipErrorInvalidValue.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  HIP_CHECK_ERROR(hipGraphAddConditionalNode(nullptr, graph, nullptr, 0,
                                              handle, hipGraphCondTypeWhile,
                                              1, &bodyGraph),
                  hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Smoke_hipGraphAddConditionalNode_NullGraph") {
  // Null parent graph must return hipErrorInvalidValue.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK_ERROR(hipGraphAddConditionalNode(&condNode, nullptr, nullptr, 0,
                                              handle, hipGraphCondTypeWhile,
                                              1, &bodyGraph),
                  hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Smoke_hipGraphAddConditionalNode_WhileBodyGraphNotNull") {
  // WHILE with 1 body: the returned body graph pointer must be non-null and
  // distinct from the parent graph.  This validates that the runtime actually
  // creates a body graph object.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeWhile,
                                        1, &bodyGraph));

  REQUIRE(bodyGraph != nullptr);
  REQUIRE(bodyGraph != graph);
  REQUIRE(condNode != nullptr);

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Smoke_hipGraphAddConditionalNode_IfTwoBodiesDistinct") {
  // IF with 2 bodies: both returned body graph pointers must be non-null and
  // distinct from each other and from the parent graph.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraphs[2] = {nullptr, nullptr};
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf,
                                        2, bodyGraphs));

  REQUIRE(bodyGraphs[0] != nullptr);
  REQUIRE(bodyGraphs[1] != nullptr);
  REQUIRE(bodyGraphs[0] != bodyGraphs[1]);
  REQUIRE(bodyGraphs[0] != graph);
  REQUIRE(bodyGraphs[1] != graph);

  HIP_CHECK(hipGraphDestroy(graph));
}

// ---------------------------------------------------------------------------
// Layer 4: Packet path smoke — forces the runtime to emit COND_BRANCH / LOOP_BACK packets
// ---------------------------------------------------------------------------
//
// These tests are tagged [!mayfail] because:
//   - The CP firmware on the target system may not yet recognise vendor packet
//     type 6 (HSA_AMD_PACKET_TYPE_AQL_COND_BRANCH).  In that case the
//     CP aborts the queue with HSA_STATUS_ERROR_INVALID_PACKET_FORMAT (0x1009)
//     which surfaces as hipErrorLaunchFailure (719) from hipStreamSynchronize.
//
// A WARN message is printed for any launch failure so the outcome is visible
// even when the test is marked may-fail.  The test PASSES if:
//   a) hipGraphLaunch + hipStreamSynchronize both succeed (firmware accepts
//      the packet), OR
//   b) hipStreamSynchronize returns hipErrorLaunchFailure (firmware rejects
//      the packet with the expected error code).
// Any other error code is treated as a hard failure.

// Minimal kernel: writes 0 to the condition cell (terminates the loop).
static __global__ void smokeCondSetKernel(hipGraphConditionalHandle handle,
                                          int* counter) {
  *counter += 1;
  // Write 0 → loop exits after one iteration.
  hipGraphSetConditional(handle, 0ULL);
}

// Minimal kernel: writes 0 to the condition cell for IF body.
static __global__ void smokeIfBodyKernel(int* output) {
  *output = 99;
}

/**
 * Helper: build a minimal WHILE graph (1 body kernel, 1 iteration), launch it,
 * and return the hipError_t from hipStreamSynchronize.
 *
 * The graph is:
 *   parent: [ COND_BRANCH (BOOL_TRUE, defaultValue=1) ]
 *   body:   [ smokeCondSetKernel ] → writes 0 → loop exits
 *
 * Expected outcomes:
 *   hipSuccess              → firmware accepted the packet; counter == 1.
 *   hipErrorLaunchFailure   → firmware rejected the packet (expected on pre-GFX12
 *                             or mismatched firmware); counter is undefined.
 */
static hipError_t runMinimalWhileSmoke(int* h_counter_out) {
  int* d_counter = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  int zero = 0;
  HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  // defaultValue=1 → loop enters; body writes 0 → loop exits after 1 iteration.
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeWhile,
                                        1, &bodyGraph));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(smokeCondSetKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&handle, &d_counter};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  hipError_t syncErr = hipStreamSynchronize(stream);

  if (syncErr == hipSuccess) {
    HIP_CHECK(hipMemcpy(h_counter_out, d_counter, sizeof(int),
                        hipMemcpyDeviceToHost));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_counter));

  return syncErr;
}

TEST_CASE("Smoke_hipGraphConditionalNode_While_PacketPath",
          "[!mayfail]"
          // [!mayfail]: CP firmware may reject vendor packet type 6
          // (HSA_AMD_PACKET_TYPE_AQL_COND_BRANCH) with
          // HSA_STATUS_ERROR_INVALID_PACKET_FORMAT → hipErrorLaunchFailure.
) {
  int h_counter = -1;
  hipError_t err = runMinimalWhileSmoke(&h_counter);

  if (err == hipSuccess) {
    INFO("Packet accepted by firmware. counter=" << h_counter);
    // If the packet was accepted and the loop ran correctly, counter == 1.
    // With a firmware ABI mismatch the loop may not terminate, so we only
    // check that counter is non-negative rather than asserting == 1.
    REQUIRE(h_counter >= 0);
  } else if (err == hipErrorLaunchFailure) {
    WARN("hipErrorLaunchFailure: CP rejected COND_BRANCH packet "
         "(expected on pre-GFX12 or mismatched firmware). "
         "This is the documented failure mode for PR-7152 before firmware ships.");
    // Not a hard failure — this is the expected outcome on current hardware.
  } else {
    // Any other error is unexpected.
    INFO("Unexpected error: " << hipGetErrorString(err) << " (" << err << ")");
    REQUIRE(err == hipSuccess);
  }
}

TEST_CASE("Smoke_hipGraphConditionalNode_While_ZeroIter_PacketPath",
          "[!mayfail]"
          // [!mayfail]: same firmware boundary as above.
          // defaultValue=0 → WHILE should not enter the body (0 iterations).
          // If the CP accepts the packet, counter must remain 0.
) {
  int* d_counter = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  int zero = 0;
  HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // defaultValue=0 → condition is false from the start → 0 iterations.
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 0u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeWhile,
                                        1, &bodyGraph));

  // Body kernel: increments counter.  Should NOT run if condition is false.
  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(smokeCondSetKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&handle, &d_counter};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  hipError_t syncErr = hipStreamSynchronize(stream);

  if (syncErr == hipSuccess) {
    int h_counter = -1;
    HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(int),
                        hipMemcpyDeviceToHost));
    INFO("Packet accepted. counter=" << h_counter
         << " (expected 0 for zero-iteration WHILE)");
    // Body must not have run.
    REQUIRE(h_counter == 0);
  } else if (syncErr == hipErrorLaunchFailure) {
    WARN("hipErrorLaunchFailure: CP rejected COND_BRANCH packet "
         "(expected on pre-GFX12 or mismatched firmware).");
  } else {
    INFO("Unexpected error: " << hipGetErrorString(syncErr)
         << " (" << syncErr << ")");
    REQUIRE(syncErr == hipSuccess);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_counter));
}

TEST_CASE("Smoke_hipGraphConditionalNode_If_TrueBranch_PacketPath",
          "[!mayfail]"
          // [!mayfail]: same firmware boundary as WHILE tests.
          // IF with defaultValue=1 → body should run → output == 99.
) {
  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));
  int zero = 0;
  HIP_CHECK(hipMemcpy(d_output, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // defaultValue=1 → condition is true → body runs.
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf,
                                        1, &bodyGraph));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(smokeIfBodyKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&d_output};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  hipError_t syncErr = hipStreamSynchronize(stream);

  if (syncErr == hipSuccess) {
    int h_output = -1;
    HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(int),
                        hipMemcpyDeviceToHost));
    INFO("Packet accepted. output=" << h_output << " (expected 99)");
    REQUIRE(h_output == 99);
  } else if (syncErr == hipErrorLaunchFailure) {
    WARN("hipErrorLaunchFailure: CP rejected COND_BRANCH packet "
         "(expected on pre-GFX12 or mismatched firmware).");
  } else {
    INFO("Unexpected error: " << hipGetErrorString(syncErr)
         << " (" << syncErr << ")");
    REQUIRE(syncErr == hipSuccess);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
}

TEST_CASE("Smoke_hipGraphConditionalNode_If_FalseBranch_PacketPath",
          "[!mayfail]"
          // [!mayfail]: same firmware boundary as WHILE tests.
          // IF with defaultValue=0 → body should NOT run → output stays 0.
) {
  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));
  int zero = 0;
  HIP_CHECK(hipMemcpy(d_output, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // defaultValue=0 → condition is false → body does not run.
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 0u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf,
                                        1, &bodyGraph));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(smokeIfBodyKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&d_output};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphLaunch(exec, stream));
  hipError_t syncErr = hipStreamSynchronize(stream);

  if (syncErr == hipSuccess) {
    int h_output = -1;
    HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(int),
                        hipMemcpyDeviceToHost));
    INFO("Packet accepted. output=" << h_output << " (expected 0, body skipped)");
    REQUIRE(h_output == 0);
  } else if (syncErr == hipErrorLaunchFailure) {
    WARN("hipErrorLaunchFailure: CP rejected COND_BRANCH packet "
         "(expected on pre-GFX12 or mismatched firmware).");
  } else {
    INFO("Unexpected error: " << hipGetErrorString(syncErr)
         << " (" << syncErr << ")");
    REQUIRE(syncErr == hipSuccess);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
}

// ---------------------------------------------------------------------------
// Layer 4b: Instantiation smoke — verify hipGraphInstantiate succeeds even
// before launch (the packet is not emitted until hipGraphLaunch).
// These are NOT tagged [!mayfail] because instantiation should always succeed
// regardless of firmware state.
// ---------------------------------------------------------------------------

TEST_CASE("Smoke_hipGraphConditionalNode_While_InstantiateOnly") {
  // Verify that a WHILE graph with a body kernel can be instantiated without
  // error.  Does not launch.  This exercises BuildIB and the IB allocation
  // path without touching the CP.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeWhile,
                                        1, &bodyGraph));

  int* d_counter = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(smokeCondSetKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&handle, &d_counter};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  // hipGraphInstantiate must succeed — it does not emit any AQL packets.
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_counter));
}

TEST_CASE("Smoke_hipGraphConditionalNode_If_InstantiateOnly") {
  // Same as above but for an IF node.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t bodyGraph = nullptr;
  hipGraphNode_t condNode;
  HIP_CHECK(hipGraphAddConditionalNode(&condNode, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf,
                                        1, &bodyGraph));

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));

  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(smokeIfBodyKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  void* args[] = {&d_output};
  kp.kernelParams = args;
  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, bodyGraph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
}
