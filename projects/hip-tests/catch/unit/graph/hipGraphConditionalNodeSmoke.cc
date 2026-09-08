/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Smoke tests for conditional graph node AQL IB packets.
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
 *   4. Packet path smoke     — minimal WHILE / IF graphs that exercise
 *      AQL_IB_COND_JUMP entry, tail jumps, and termination.
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

static_assert(HSA_AMD_PACKET_TYPE_AQL_IB_COND_JUMP == 6,
              "AQL IB conditional jump packet type changed");
static_assert(HSA_AMD_PACKET_TYPE_AQL_IB_JUMP == 7,
              "AQL IB jump packet type changed");
static_assert(HSA_AMD_PACKET_TYPE_RESERVED5 == 5,
              "AQL IB format 5 must remain reserved");

static_assert(HSA_AMD_AQL_IB_COND_JUMP_OP_EQ == 0);
static_assert(HSA_AMD_AQL_IB_COND_JUMP_OP_BOOL_TRUE == 10);
static_assert(HSA_AMD_AQL_IB_COND_JUMP_OP_BOOL_FALSE == 11);
static_assert(sizeof(hsa_amd_aql_ib_cond_jump_packet_t) == 64,
              "conditional jump packet must remain one AQL slot");
static_assert(offsetof(hsa_amd_aql_ib_cond_jump_packet_t, condition_signal) == 8);
static_assert(offsetof(hsa_amd_aql_ib_cond_jump_packet_t, true_target_base_addr) == 24);
static_assert(offsetof(hsa_amd_aql_ib_cond_jump_packet_t, false_target_base_addr) == 32);
static_assert(offsetof(hsa_amd_aql_ib_cond_jump_packet_t, completion_signal) == 48);

static_assert(sizeof(hsa_amd_aql_ib_jump_packet_t) == 64,
              "unconditional jump packet must remain one AQL slot");
static_assert(offsetof(hsa_amd_aql_ib_jump_packet_t, target_size_packets) == 4);
static_assert(offsetof(hsa_amd_aql_ib_jump_packet_t, target_base_addr) == 8);
static_assert(offsetof(hsa_amd_aql_ib_jump_packet_t, completion_signal) == 48);

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
// Layer 4: AQL IB packet-path smoke
// ---------------------------------------------------------------------------
//
// These tests validate the observable graph and AQL packet semantics. A launch
// failure is reported explicitly before the test fails.

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

static __global__ void smokeIfElseBodyKernel(int* output, int value) {
  *output = value;
}

static __global__ void smokeWhileCountKernel(hipGraphConditionalHandle handle,
                                              int* counter, int limit) {
  const int value = atomicAdd(counter, 1) + 1;
  hipGraphSetConditional(handle, value < limit ? 1ULL : 0ULL);
}

static __global__ void smokeObserveKernel(const int* input, int* output) {
  *output = *input;
}

/**
 * Helper: build a minimal WHILE graph (1 body kernel, 1 iteration), launch it,
 * and return the hipError_t from hipStreamSynchronize.
 *
 * The graph is:
 *   parent: [ AQL_IB_COND_JUMP (BOOL_TRUE, defaultValue=1) ]
 *   body:   [ smokeCondSetKernel ] → writes 0 → loop exits
 *
 * Expected outcome: launch succeeds and counter == 1.
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

TEST_CASE("Smoke_hipGraphConditionalNode_While_PacketPath") {
  int h_counter = -1;
  hipError_t err = runMinimalWhileSmoke(&h_counter);

  if (err == hipSuccess) {
    INFO("Conditional graph completed. counter=" << h_counter);
    REQUIRE(h_counter == 1);
  } else if (err == hipErrorLaunchFailure) {
    WARN("hipErrorLaunchFailure while executing AQL IB conditional graph");
    REQUIRE(err == hipSuccess);
  } else {
    // Any other error is unexpected.
    INFO("Unexpected error: " << hipGetErrorString(err) << " (" << err << ")");
    REQUIRE(err == hipSuccess);
  }
}

TEST_CASE("Smoke_hipGraphConditionalNode_While_ZeroIter_PacketPath",
          "[graph][conditional]"
          // defaultValue=0 → WHILE should not enter the body (0 iterations).
          // The counter must remain 0.
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
    WARN("hipErrorLaunchFailure while executing AQL IB conditional graph");
    REQUIRE(syncErr == hipSuccess);
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

TEST_CASE("Smoke_hipGraphConditionalNode_While_RepeatedAndPqContinuation") {
  // This exercises the complete CFG rather than only its first edge:
  //
  //   PQ COND_JUMP -> body -> IB COND_JUMP -> body ... -> null
  //                  (five iterations)                  |
  //                                                     v
  //                                          following PQ dispatch
  //
  // Every back edge must observe the preceding kernel's condition write. The
  // dependent parent-graph kernel must run only after the complete
  // selected IB path has finished and queue execution has resumed.
  int* d_counter = nullptr;
  int* d_observed = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_observed, sizeof(int)));
  HIP_CHECK(hipMemset(d_counter, 0, sizeof(int)));
  HIP_CHECK(hipMemset(d_observed, 0, sizeof(int)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, 1u, 0));

  hipGraph_t body_graph = nullptr;
  hipGraphNode_t cond_node;
  HIP_CHECK(hipGraphAddConditionalNode(&cond_node, graph, nullptr, 0,
                                        handle, hipGraphCondTypeWhile, 1,
                                        &body_graph));

  int limit = 5;
  void* body_args[] = {&handle, &d_counter, &limit};
  hipKernelNodeParams body_params = {};
  body_params.func = reinterpret_cast<void*>(smokeWhileCountKernel);
  body_params.gridDim = dim3(1);
  body_params.blockDim = dim3(1);
  body_params.kernelParams = body_args;
  hipGraphNode_t body_node;
  HIP_CHECK(hipGraphAddKernelNode(&body_node, body_graph, nullptr, 0,
                                  &body_params));

  void* observe_args[] = {&d_counter, &d_observed};
  hipKernelNodeParams observe_params = {};
  observe_params.func = reinterpret_cast<void*>(smokeObserveKernel);
  observe_params.gridDim = dim3(1);
  observe_params.blockDim = dim3(1);
  observe_params.kernelParams = observe_args;
  hipGraphNode_t observe_node;
  HIP_CHECK(hipGraphAddKernelNode(&observe_node, graph, &cond_node, 1,
                                  &observe_params));

  hipGraphExec_t exec;
  hipStream_t stream;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  hipError_t sync_err = hipStreamSynchronize(stream);

  if (sync_err == hipSuccess) {
    int counter = 0;
    int observed = 0;
    HIP_CHECK(hipMemcpy(&counter, d_counter, sizeof(int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&observed, d_observed, sizeof(int),
                        hipMemcpyDeviceToHost));
    REQUIRE(counter == limit);
    REQUIRE(observed == limit);
  } else if (sync_err == hipErrorLaunchFailure) {
    WARN("hipErrorLaunchFailure while executing repeated AQL IB conditional graph");
    REQUIRE(sync_err == hipSuccess);
  } else {
    INFO("Unexpected error: " << hipGetErrorString(sync_err)
         << " (" << sync_err << ")");
    REQUIRE(sync_err == hipSuccess);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_observed));
  HIP_CHECK(hipFree(d_counter));
}

TEST_CASE("Smoke_hipGraphConditionalNode_If_TrueBranch_PacketPath",
          "[graph][conditional]"
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
    WARN("hipErrorLaunchFailure while executing AQL IB conditional graph");
    REQUIRE(syncErr == hipSuccess);
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
          "[graph][conditional]"
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
    WARN("hipErrorLaunchFailure while executing AQL IB conditional graph");
    REQUIRE(syncErr == hipSuccess);
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

static hipError_t runIfElseSmoke(unsigned int default_value,
                                 int* h_output_out) {
  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));
  int zero = 0;
  HIP_CHECK(hipMemcpy(d_output, &zero, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphConditionalHandle handle;
  HIP_CHECK(hipGraphConditionalHandleCreate(&handle, graph, default_value, 0));

  hipGraph_t body_graphs[2] = {nullptr, nullptr};
  hipGraphNode_t cond_node;
  HIP_CHECK(hipGraphAddConditionalNode(&cond_node, graph, nullptr, 0,
                                        handle, hipGraphCondTypeIf, 2,
                                        body_graphs));

  int true_value = 101;
  int false_value = 202;
  void* true_args[] = {&d_output, &true_value};
  void* false_args[] = {&d_output, &false_value};
  hipKernelNodeParams true_kp = {};
  true_kp.func = reinterpret_cast<void*>(smokeIfElseBodyKernel);
  true_kp.gridDim = dim3(1);
  true_kp.blockDim = dim3(1);
  true_kp.kernelParams = true_args;
  hipKernelNodeParams false_kp = true_kp;
  false_kp.kernelParams = false_args;
  hipGraphNode_t true_node;
  hipGraphNode_t false_node;
  HIP_CHECK(hipGraphAddKernelNode(&true_node, body_graphs[0], nullptr, 0,
                                  &true_kp));
  HIP_CHECK(hipGraphAddKernelNode(&false_node, body_graphs[1], nullptr, 0,
                                  &false_kp));

  hipGraphExec_t exec;
  hipStream_t stream;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  hipError_t sync_err = hipStreamSynchronize(stream);
  if (sync_err == hipSuccess) {
    HIP_CHECK(hipMemcpy(h_output_out, d_output, sizeof(int),
                        hipMemcpyDeviceToHost));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
  return sync_err;
}

TEST_CASE("Smoke_hipGraphConditionalNode_IfElse_BranchIsolation_PacketPath") {
  const unsigned int default_value = GENERATE(0u, 1u);
  int output = -1;
  hipError_t err = runIfElseSmoke(default_value, &output);

  if (err == hipSuccess) {
    // The descriptors point at unrelated extents in one allocation. Executing
    // both arms or falling through from TRUE into FALSE is a failure.
    REQUIRE(output == (default_value != 0 ? 101 : 202));
  } else if (err == hipErrorLaunchFailure) {
    WARN("hipErrorLaunchFailure while executing AQL IB conditional graph");
    REQUIRE(err == hipSuccess);
  } else {
    INFO("Unexpected error: " << hipGetErrorString(err) << " (" << err << ")");
    REQUIRE(err == hipSuccess);
  }
}

// ---------------------------------------------------------------------------
// Layer 4b: Instantiation smoke — verify hipGraphInstantiate succeeds even
// before launch (the packet is not emitted until hipGraphLaunch).
// ---------------------------------------------------------------------------

TEST_CASE("Smoke_hipGraphConditionalNode_While_InstantiateOnly") {
  // Verify that a WHILE graph with a body kernel can be instantiated without
  // error.  Does not launch.  This exercises BuildIB and the IB allocation
  // path without submitting a packet.
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
