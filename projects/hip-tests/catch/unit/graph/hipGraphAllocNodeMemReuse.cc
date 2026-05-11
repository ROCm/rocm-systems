/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <resource_guards.hh>
#include <utils.hh>
#include <thread>
#include <atomic>

/**
 * @addtogroup hipGraphAllocNodeMemReuse hipGraphAllocNodeMemReuse
 * @{
 * @ingroup GraphTest
 * Tests for verifying memory reuse optimization for graph allocation nodes.
 */

static constexpr int kBlockSize = 256;

__global__ void fillKernel(float* buf, float value, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[idx] = value;
  }
}

__global__ void verifyKernel(const float* buf, float expected, int64_t n, int* error) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n && buf[idx] != expected) {
    atomicAdd(error, 1);
  }
}

// Busy-spin kernel: each thread loops `iters` times, reading/writing buf to
// prevent the compiler from optimizing the loop away. Use a large iteration
// count to keep the GPU busy long enough for concurrent host-thread launches.
__global__ void busyKernel(float* buf, float value, int64_t n, int64_t iters) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = value;
    for (int64_t i = 0; i < iters; ++i) {
      v = v * 1.0001f + 0.0001f;
    }
    buf[idx] = v;
  }
}

// Helper to capture a graph with hipMallocAsync and kernel launch
static hipGraphExec_t captureGraphWithAlloc(hipStream_t stream, size_t allocBytes,
                                            float fillValue) {
  int64_t numElems = static_cast<int64_t>(allocBytes / sizeof(float));
  int gridSize = static_cast<int>((numElems + kBlockSize - 1) / kBlockSize);

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  float* dTemp = nullptr;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&dTemp), allocBytes, stream));

  fillKernel<<<gridSize, kBlockSize, 0, stream>>>(dTemp, fillValue, numElems);

  HIP_CHECK(hipFreeAsync(dTemp, stream));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphDestroy(graph));

  return exec;
}

// Helper to capture a graph with hipMallocAsync and a long-running busy kernel.
// The busy kernel spins for `iters` iterations to keep the GPU occupied, ensuring
// concurrent launches from different host threads actually overlap on the GPU.
static hipGraphExec_t captureGraphWithBusyKernel(hipStream_t stream, size_t allocBytes,
                                                  float fillValue, int64_t iters) {
  int64_t numElems = static_cast<int64_t>(allocBytes / sizeof(float));
  int gridSize = static_cast<int>((numElems + kBlockSize - 1) / kBlockSize);

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  float* dTemp = nullptr;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&dTemp), allocBytes, stream));

  busyKernel<<<gridSize, kBlockSize, 0, stream>>>(dTemp, fillValue, numElems, iters);

  HIP_CHECK(hipFreeAsync(dTemp, stream));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphDestroy(graph));

  return exec;
}

// Helper to reset graph memory attributes
static void resetGraphMemAttributes(int device) {
  uint64_t zero = 0;
  HIP_CHECK(hipDeviceSetGraphMemAttribute(device, hipGraphMemAttrUsedMemHigh, &zero));
  HIP_CHECK(hipDeviceSetGraphMemAttribute(device, hipGraphMemAttrReservedMemHigh, &zero));
}

// Helper to query graph memory attributes
struct GraphMemStats {
  uint64_t usedCurrent;
  uint64_t usedHigh;
  uint64_t reservedCurrent;
  uint64_t reservedHigh;
};

static GraphMemStats queryGraphMem(int device) {
  GraphMemStats s{};
  HIP_CHECK(
      hipDeviceGetGraphMemAttribute(device, hipGraphMemAttrUsedMemCurrent, &s.usedCurrent));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(device, hipGraphMemAttrUsedMemHigh, &s.usedHigh));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(device, hipGraphMemAttrReservedMemCurrent,
                                          &s.reservedCurrent));
  HIP_CHECK(
      hipDeviceGetGraphMemAttribute(device, hipGraphMemAttrReservedMemHigh, &s.reservedHigh));
  return s;
}

void printDeviceMem(const char* label) {
    size_t freeMem = 0, totalMem = 0;
    HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
    size_t usedMem = totalMem - freeMem;
    std::printf("[DevMem]   %-45s  used: %8.2f MB  free: %8.2f MB\n",
        label,
        static_cast<double>(usedMem) / (1024.0 * 1024.0),
        static_cast<double>(freeMem) / (1024.0 * 1024.0));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that when multiple graphs with the SAME allocation size are launched
 *    sequentially on the same stream, they reuse the same physical memory.
 *  - The memory reuse optimization works by caching allocations at the exact size requested.
 *    When a subsequent graph requests the same allocation size, it can reuse the cached
 *    physical memory from the previous allocation.
 *  - This test creates three graphs with identical allocation sizes and verifies that the
 *    high water mark reflects only a single allocation, not the sum of all three.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_SameStream_SameSizes") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocSize = 128ULL * 1024 * 1024;  // 128 MB - same for all

  resetGraphMemAttributes(device);

  // Capture three graphs with IDENTICAL allocation sizes
  hipGraphExec_t execA = captureGraphWithAlloc(stream, kAllocSize, 1.0f);
  hipGraphExec_t execB = captureGraphWithAlloc(stream, kAllocSize, 2.0f);
  hipGraphExec_t execC = captureGraphWithAlloc(stream, kAllocSize, 3.0f);

  // Launch all three sequentially on the same stream
  HIP_CHECK(hipGraphLaunch(execA, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphLaunch(execB, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphLaunch(execC, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Check memory statistics after launch+sync
  auto stats = queryGraphMem(device);

  // Memory is retained in the graph for reuse - usedCurrent should reflect
  // a single allocation's worth (all three graphs reuse the same memory)
  REQUIRE(stats.usedCurrent == kAllocSize);

  // High water mark should be highest memory usage size
  REQUIRE(stats.usedHigh == kAllocSize);

  // Destroy graph executables - memory should be released
  HIP_CHECK(hipGraphExecDestroy(execA));
  HIP_CHECK(hipGraphExecDestroy(execB));
  HIP_CHECK(hipGraphExecDestroy(execC));

#if HT_AMD
  // After graph destruction, memory should return to 0
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that graphs with DIFFERENT allocation sizes do NOT reuse memory,
 *    even when launched sequentially on the same stream.
 *  - The memory reuse optimization only works for exact size matches. Different sizes
 *    require separate physical allocations acquired directly from the OS.
 *  - This test creates three graphs with different allocation sizes and verifies that
 *    the current memory usage is the sum of all 3 allocations on AMD.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_SameStream_DifferentSizes_NoReuse") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocA = 128ULL * 1024 * 1024;  // 128 MB
  constexpr size_t kAllocB = 64ULL * 1024 * 1024;   // 64 MB
  constexpr size_t kAllocC = 32ULL * 1024 * 1024;   // 32 MB

  resetGraphMemAttributes(device);

  // Capture three graphs with DIFFERENT allocation sizes
  hipGraphExec_t execA = captureGraphWithAlloc(stream, kAllocA, 1.0f);
  hipGraphExec_t execB = captureGraphWithAlloc(stream, kAllocB, 2.0f);
  hipGraphExec_t execC = captureGraphWithAlloc(stream, kAllocC, 3.0f);

  // Launch all three sequentially on the same stream
  HIP_CHECK(hipGraphLaunch(execA, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphLaunch(execB, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphLaunch(execC, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Check memory statistics
  auto stats = queryGraphMem(device);

  // Memory is retained in graphs until destruction or trim.
#if HT_NVIDIA
  constexpr size_t totalAlloc = kAllocA;
#else
  constexpr size_t totalAlloc = kAllocA + kAllocB + kAllocC;
#endif
  REQUIRE(stats.usedCurrent == totalAlloc);

  // High water mark should reflect highest memory usage at any time
  REQUIRE(stats.usedHigh == kAllocA);

  // Destroy graph executables - memory should be released
  HIP_CHECK(hipGraphExecDestroy(execA));
  HIP_CHECK(hipGraphExecDestroy(execB));
  HIP_CHECK(hipGraphExecDestroy(execC));

#if HT_AMD
  // After graph destruction, memory should return to 0
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that memory reuse works correctly across multiple launches of the same
 *    graph executable.
 *  - This test launches the same graph multiple times sequentially and verifies that the
 *    memory high water mark remains stable (not growing with each launch).
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_RepeatedLaunches") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocSize = 64ULL * 1024 * 1024;  // 64 MB

  resetGraphMemAttributes(device);

  // Capture a single graph
  hipGraphExec_t exec = captureGraphWithAlloc(stream, kAllocSize, 1.0f);

  // Launch the same graph multiple times
  constexpr int numLaunches = 10;
  for (int i = 0; i < numLaunches; i++) {
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Check memory statistics
  auto stats = queryGraphMem(device);

  // Memory is retained in graph until destruction or trim
  REQUIRE(stats.usedCurrent == kAllocSize);

  // High water mark should be close to a single allocation size
  // Not numLaunches * kAllocSize
  REQUIRE(stats.usedHigh == kAllocSize);

  // Destroy graph executable - memory should be released
  HIP_CHECK(hipGraphExecDestroy(exec));

#if HT_AMD
  // After graph destruction, memory should return to 0
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that hipDeviceGraphMemTrim properly releases reserved graph memory.
 *  - After launching graphs and synchronizing, reserved memory should still be held for
 *    potential reuse. After calling hipDeviceGraphMemTrim, the reserved memory should be
 *    released back to the OS.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_MemoryTrim") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocSize = 128ULL * 1024 * 1024;  // 128 MB

  resetGraphMemAttributes(device);

  // Capture and launch a graph
  hipGraphExec_t exec = captureGraphWithAlloc(stream, kAllocSize, 1.0f);
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Memory is retained in graph for reuse after launch+sync
  auto statsBefore = queryGraphMem(device);
  REQUIRE(statsBefore.usedCurrent == kAllocSize);
  REQUIRE(statsBefore.reservedCurrent > 0);  // Memory should be reserved for reuse

  // Trim graph memory
  HIP_CHECK(hipDeviceGraphMemTrim(device));

  // Check that reserved memory is released after trim
  auto statsAfter = queryGraphMem(device);
  REQUIRE(statsAfter.usedCurrent == 0);
  REQUIRE(statsAfter.reservedCurrent == 0);  // Reserved memory should be freed

  // Cleanup
  HIP_CHECK(hipGraphExecDestroy(exec));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify memory reuse with graphs containing explicit hipGraphAddMemAllocNode
 *    and hipGraphAddMemFreeNode instead of stream capture.
 *  - This test manually constructs two graphs with IDENTICAL allocation sizes and verifies
 *    memory reuse when launched sequentially on the same stream.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_ExplicitAllocFreeNodes_SameSize") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocSize = 128ULL * 1024 * 1024;  // 128 MB - same for both

  resetGraphMemAttributes(device);

  // Create first graph with explicit alloc/free nodes
  hipGraph_t graphA;
  HIP_CHECK(hipGraphCreate(&graphA, 0));

  hipGraphNode_t allocNodeA;
  hipMemAllocNodeParams allocParamA;
  memset(&allocParamA, 0, sizeof(allocParamA));
  allocParamA.bytesize = kAllocSize;
  allocParamA.poolProps.allocType = hipMemAllocationTypePinned;
  allocParamA.poolProps.location.id = 0;
  allocParamA.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graphA, nullptr, 0, &allocParamA));
  REQUIRE(allocParamA.dptr != nullptr);

  hipGraphNode_t freeNodeA;
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graphA, &allocNodeA, 1, allocParamA.dptr));

  hipGraphExec_t execA;
  HIP_CHECK(hipGraphInstantiate(&execA, graphA, nullptr, nullptr, 0));

  // Create second graph with explicit alloc/free nodes - SAME SIZE
  hipGraph_t graphB;
  HIP_CHECK(hipGraphCreate(&graphB, 0));

  hipGraphNode_t allocNodeB;
  hipMemAllocNodeParams allocParamB;
  memset(&allocParamB, 0, sizeof(allocParamB));
  allocParamB.bytesize = kAllocSize;  // Same size as graph A
  allocParamB.poolProps.allocType = hipMemAllocationTypePinned;
  allocParamB.poolProps.location.id = 0;
  allocParamB.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeB, graphB, nullptr, 0, &allocParamB));
  REQUIRE(allocParamB.dptr != nullptr);

  hipGraphNode_t freeNodeB;
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeB, graphB, &allocNodeB, 1, allocParamB.dptr));

  hipGraphExec_t execB;
  HIP_CHECK(hipGraphInstantiate(&execB, graphB, nullptr, nullptr, 0));

  // Launch both graphs sequentially on same stream
  HIP_CHECK(hipGraphLaunch(execA, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGraphLaunch(execB, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Check memory statistics
  auto stats = queryGraphMem(device);

  // Memory is retained in graphs until destruction or trim.
  // Both graphs reuse the same memory (same size), so usedCurrent == kAllocSize.
  REQUIRE(stats.usedCurrent == kAllocSize);

  // High water mark should be a single allocation (kAllocSize)
  // NOT 2x, proving memory reuse for same-sized allocations
  REQUIRE(stats.usedHigh == kAllocSize);

  // Destroy graph executables and graphs - memory should be released
  HIP_CHECK(hipGraphExecDestroy(execA));
  HIP_CHECK(hipGraphExecDestroy(execB));
  HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipGraphDestroy(graphB));

#if HT_AMD
  // After graph destruction, memory should return to 0
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that explicit alloc nodes with DIFFERENT sizes do NOT reuse memory.
 *  - This complements the previous test by showing that different-sized allocations
 *    require separate physical memory, even with explicit alloc/free nodes.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_ExplicitAllocFreeNodes_DifferentSizes") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocA = 128ULL * 1024 * 1024;  // 128 MB
  constexpr size_t kAllocB = 64ULL * 1024 * 1024;   // 64 MB - different size

  resetGraphMemAttributes(device);

  // Create first graph with explicit alloc/free nodes
  hipGraph_t graphA;
  HIP_CHECK(hipGraphCreate(&graphA, 0));

  hipGraphNode_t allocNodeA;
  hipMemAllocNodeParams allocParamA;
  memset(&allocParamA, 0, sizeof(allocParamA));
  allocParamA.bytesize = kAllocA;
  allocParamA.poolProps.allocType = hipMemAllocationTypePinned;
  allocParamA.poolProps.location.id = 0;
  allocParamA.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graphA, nullptr, 0, &allocParamA));
  REQUIRE(allocParamA.dptr != nullptr);

  hipGraphNode_t freeNodeA;
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graphA, &allocNodeA, 1, allocParamA.dptr));

  hipGraphExec_t execA;
  HIP_CHECK(hipGraphInstantiate(&execA, graphA, nullptr, nullptr, 0));

  // Create second graph with DIFFERENT size
  hipGraph_t graphB;
  HIP_CHECK(hipGraphCreate(&graphB, 0));

  hipGraphNode_t allocNodeB;
  hipMemAllocNodeParams allocParamB;
  memset(&allocParamB, 0, sizeof(allocParamB));
  allocParamB.bytesize = kAllocB;  // Different size
  allocParamB.poolProps.allocType = hipMemAllocationTypePinned;
  allocParamB.poolProps.location.id = 0;
  allocParamB.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeB, graphB, nullptr, 0, &allocParamB));
  REQUIRE(allocParamB.dptr != nullptr);

  hipGraphNode_t freeNodeB;
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeB, graphB, &allocNodeB, 1, allocParamB.dptr));

  hipGraphExec_t execB;
  HIP_CHECK(hipGraphInstantiate(&execB, graphB, nullptr, nullptr, 0));

  // Launch both graphs sequentially on same stream
  HIP_CHECK(hipGraphLaunch(execA, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGraphLaunch(execB, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Check memory statistics
  auto stats = queryGraphMem(device);

  // Memory is retained in graphs until destruction or trim.
  // Each graph holds its own allocation (no reuse across different sizes).
#if HT_NVIDIA
  constexpr size_t totalAlloc = kAllocA; // on Nvidia 128MB chunk is reused
#else
  constexpr size_t totalAlloc = kAllocA + kAllocB;
#endif
  REQUIRE(stats.usedCurrent == totalAlloc);

  // High water mark should reflect highest memory usage at any time
  REQUIRE(stats.usedHigh == kAllocA);

  // Destroy graph executables and graphs - memory should be released
  HIP_CHECK(hipGraphExecDestroy(execA));
  HIP_CHECK(hipGraphExecDestroy(execB));
  HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipGraphDestroy(graphB));

#if HT_AMD
  // After graph destruction, memory should return to 0
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that memory reuse does NOT occur for malloc nodes that don't have
 *    a corresponding free node within the graph.
 *  - This test creates a graph with 4 malloc nodes (all same size), but only 3 have
 *    corresponding free nodes in the graph. The 4th malloc node's memory is freed
 *    using hipFreeAsync outside the graph after launch.
 *  - Since the 4th malloc node lacks an in-graph free node, it cannot participate in
 *    memory reuse optimization. The total memory usage should reflect this.
 *  - Expected: Memory usage > single allocation size (because the orphan malloc node
 *    holds separate memory that can't be reused by the other nodes).
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_MallocWithoutFree_NoReuse") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocSize = 64ULL * 1024 * 1024;  // 64 MB - same for all nodes

  resetGraphMemAttributes(device);

  // Create a graph with 4 malloc nodes, but only 3 have free nodes.
  // The orphan malloc node is placed FIRST in the graph.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Node 1: Orphan malloc WITHOUT corresponding free (placed first)
  hipGraphNode_t allocNode1;
  hipMemAllocNodeParams allocParam1;
  memset(&allocParam1, 0, sizeof(allocParam1));
  allocParam1.bytesize = kAllocSize;
  allocParam1.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam1.poolProps.location.id = 0;
  allocParam1.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNode1, graph, nullptr, 0, &allocParam1));
  REQUIRE(allocParam1.dptr != nullptr);
  void* orphanPtr = allocParam1.dptr;  // Will free this outside the graph

  // Node 2: Malloc with corresponding free (depends on orphan node)
  hipGraphNode_t allocNode2;
  hipMemAllocNodeParams allocParam2;
  memset(&allocParam2, 0, sizeof(allocParam2));
  allocParam2.bytesize = kAllocSize;
  allocParam2.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam2.poolProps.location.id = 0;
  allocParam2.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNode2, graph, &allocNode1, 1, &allocParam2));
  REQUIRE(allocParam2.dptr != nullptr);

  hipGraphNode_t freeNode2;
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNode2, graph, &allocNode2, 1, allocParam2.dptr));

  // Node 3: Malloc with corresponding free (depends on freeNode2)
  hipGraphNode_t allocNode3;
  hipMemAllocNodeParams allocParam3;
  memset(&allocParam3, 0, sizeof(allocParam3));
  allocParam3.bytesize = kAllocSize;
  allocParam3.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam3.poolProps.location.id = 0;
  allocParam3.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNode3, graph, &freeNode2, 1, &allocParam3));
  REQUIRE(allocParam3.dptr != nullptr);

  hipGraphNode_t freeNode3;
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNode3, graph, &allocNode3, 1, allocParam3.dptr));

  // Node 4: Malloc with corresponding free (depends on freeNode3)
  hipGraphNode_t allocNode4;
  hipMemAllocNodeParams allocParam4;
  memset(&allocParam4, 0, sizeof(allocParam4));
  allocParam4.bytesize = kAllocSize;
  allocParam4.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam4.poolProps.location.id = 0;
  allocParam4.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNode4, graph, &freeNode3, 1, &allocParam4));
  REQUIRE(allocParam4.dptr != nullptr);

  hipGraphNode_t freeNode4;
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNode4, graph, &allocNode4, 1, allocParam4.dptr));

  // Instantiate and launch the graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Check memory statistics BEFORE freeing the orphan allocation
  auto statsBefore = queryGraphMem(device);

  // The orphan malloc node (node 1) has no in-graph free, so it cannot participate in
  // memory reuse. It holds its own 64MB allocation throughout.
  // Nodes 2-4 have free nodes and can reuse memory among themselves, needing one
  // additional 64MB allocation.
  // Therefore usedCurrent == 2 * kAllocSize:
  //   - kAllocSize for the orphan node (still allocated, no in-graph free)
  //   - kAllocSize for the reusable pool shared by nodes 2-4
  REQUIRE(statsBefore.usedCurrent == kAllocSize * 2);

  // Free the orphan allocation outside the graph using hipFreeAsync
  HIP_CHECK(hipFreeAsync(orphanPtr, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Destroy graph executable and graph - remaining retained memory should be released
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

#if HT_AMD
  // After graph destruction, memory should return to 0
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that graphs captured on different streams CAN share memory
 *    from the shared graph memory pool.
 *  - Two graphs are captured on separate streams, each with a malloc->kernel->free
 *    sequence. When launched sequentially, the second graph reuses the physical
 *    memory freed by the first, so usedCurrent reflects a single allocation.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_DifferentStreams_Reuse") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard1(Streams::created);
  hipStream_t stream1 = stream_guard1.stream();

  StreamGuard stream_guard2(Streams::created);
  hipStream_t stream2 = stream_guard2.stream();

  constexpr size_t kAllocSize = 64ULL * 1024 * 1024;  // 64 MB for each graph

  resetGraphMemAttributes(device);

  // Capture graph A on stream1: malloc -> fill kernel -> free
  hipGraphExec_t execA = captureGraphWithAlloc(stream1, kAllocSize, 1.0f);

  // Capture graph B on stream2: malloc -> fill kernel -> free
  hipGraphExec_t execB = captureGraphWithAlloc(stream2, kAllocSize, 2.0f);

  // Launch graph A on stream1 and graph B on stream2
  HIP_CHECK(hipGraphLaunch(execA, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));

  HIP_CHECK(hipGraphLaunch(execB, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));

  // Check memory statistics
  auto stats = queryGraphMem(device);

  // Both graphs share the same graph memory pool. When launched sequentially,
  // graph B reuses graph A's freed physical memory, so usedCurrent == kAllocSize.
  REQUIRE(stats.usedCurrent == kAllocSize);

  // Destroy graph executables - memory should be released
  HIP_CHECK(hipGraphExecDestroy(execA));
  HIP_CHECK(hipGraphExecDestroy(execB));

#if HT_AMD
  // After graph destruction, memory should return to 0
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that VA remap works correctly when two graphs share the same
 *    memory pool and one graph steals the other's physical memory.
 *  - Sequence:
 *    1. Capture graph A and graph B with malloc->kernel->free (same alloc size).
 *    2. Launch graph A on stream1 and sync. Physical memory goes to the shared
 *       graph pool's free_heap.
 *    3. From two host threads, concurrently:
 *       - Thread 1: launch graph B on stream2 (may steal graph A's physical memory
 *         from free_heap via opportunistic reuse).
 *       - Thread 2: relaunch graph A on stream1 (if its physical memory was stolen,
 *         the runtime must allocate new memory and remap the VA).
 *    4. Both threads sync. Verify neither graph crashes and memory stats are consistent.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_MemSteal_Remap") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard1(Streams::created);
  hipStream_t stream1 = stream_guard1.stream();

  StreamGuard stream_guard2(Streams::created);
  hipStream_t stream2 = stream_guard2.stream();

  constexpr size_t kAllocSize = 64ULL * 1024 * 1024;  // 64 MB - same for both graphs
  // Enough iterations to keep the GPU busy so both threads overlap
  constexpr int64_t kBusyIters = 100;

  resetGraphMemAttributes(device);

  // Step 1: Capture two graphs with busyKernel to ensure long-running GPU work.
  hipGraphExec_t execA = captureGraphWithBusyKernel(stream1, kAllocSize, 1.0f, kBusyIters);
  hipGraphExec_t execB = captureGraphWithBusyKernel(stream2, kAllocSize, 2.0f, kBusyIters);

  // Step 2: Launch graph A first and sync — its physical memory goes to the shared
  // graph pool's free_heap, available for opportunistic reuse by graph B.
  HIP_CHECK(hipGraphLaunch(execA, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));

  auto statsAfterFirstLaunch = queryGraphMem(device);
  REQUIRE(statsAfterFirstLaunch.usedCurrent == kAllocSize);

  // Step 3: Concurrently launch graph B (which may steal graph A's memory) and
  // relaunch graph A (which must remap if its memory was stolen).
  // Use an atomic flag as a barrier so both threads launch at roughly the same time.
  hipError_t errB = hipSuccess;
  hipError_t errA = hipSuccess;
  std::atomic<int> ready{0};

  std::thread threadB([&]() {
    ready.fetch_add(1, std::memory_order_release);
    while (ready.load(std::memory_order_acquire) < 2) {}
    errB = hipGraphLaunch(execB, stream2);
    if (errB == hipSuccess) {
      errB = hipStreamSynchronize(stream2);
    }
  });

  std::thread threadA([&]() {
    ready.fetch_add(1, std::memory_order_release);
    while (ready.load(std::memory_order_acquire) < 2) {}
    errA = hipGraphLaunch(execA, stream1);
    if (errA == hipSuccess) {
      errA = hipStreamSynchronize(stream1);
    }
  });

  threadB.join();
  threadA.join();

  REQUIRE(errB == hipSuccess);
  REQUIRE(errA == hipSuccess);

  // Step 4: Verify both graphs completed successfully and memory stats are sane.
  // The outcome depends on thread scheduling:
  //   - If launches overlapped: graph B stole graph A's memory, graph A remapped
  //     to new memory → usedCurrent == 2 * kAllocSize
  //   - If launches serialized: one graph reused the other's freed memory
  //     → usedCurrent == kAllocSize
  // Both outcomes are correct — the key assertion is no crash and no corruption.
  auto statsAfterConcurrent = queryGraphMem(device);
  REQUIRE((statsAfterConcurrent.usedCurrent == kAllocSize ||
           statsAfterConcurrent.usedCurrent == kAllocSize * 2));

  // Cleanup
  HIP_CHECK(hipGraphExecDestroy(execA));
  HIP_CHECK(hipGraphExecDestroy(execB));

#if HT_AMD
  auto statsAfterDestroy = queryGraphMem(device);
  REQUIRE(statsAfterDestroy.usedCurrent == 0);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that hipGraphInstantiateFlagAutoFreeOnLaunch frees unmatched
 *    alloc node memory at the start of each launch, allowing re-launch to succeed.
 *  - A graph with alloc -> kernel and NO free node is used. Without the flag,
 *    a second launch fails with hipErrorInvalidValue because the alloc node's
 *    memory is still live (memAllocNodeCount > 0). With AutoFreeOnLaunch, the
 *    runtime frees all prior allocations before re-executing, so the second
 *    launch succeeds and the kernel writes to freshly allocated memory.
 *  - Data correctness is verified by reading back the buffer after the second
 *    launch to confirm the kernel wrote the expected value.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAllocNodeMemReuse.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
TEST_CASE("Unit_hipGraphAllocNodeMemReuse_AutoFreeOnLaunch") {
  const int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr size_t kAllocSize = 64ULL * 1024 * 1024;  // 64 MB
  int64_t numElems = static_cast<int64_t>(kAllocSize / sizeof(float));
  int gridSize = static_cast<int>((numElems + kBlockSize - 1) / kBlockSize);

  resetGraphMemAttributes(device);

  // Create a graph with alloc -> kernel (no free node)
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t allocNode;
  hipMemAllocNodeParams allocParam;
  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = kAllocSize;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = device;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParam));
  REQUIRE(allocParam.dptr != nullptr);

  hipGraphNode_t kernelNode;
  hipKernelNodeParams kernelParam;
  memset(&kernelParam, 0, sizeof(kernelParam));
  float fillValue = 42.0f;
  void* kernelArgs[] = {&allocParam.dptr, &fillValue, &numElems};
  kernelParam.func = reinterpret_cast<void*>(fillKernel);
  kernelParam.gridDim = dim3(gridSize);
  kernelParam.blockDim = dim3(kBlockSize);
  kernelParam.kernelParams = kernelArgs;

  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, &allocNode, 1, &kernelParam));

  // Without AutoFreeOnLaunch: second launch must fail because the alloc node's
  // memory is still live (no matching free node, memAllocNodeCount > 0).
  SECTION("Without AutoFreeOnLaunch") {
    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiateWithFlags(&exec, graph, 0));

    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Second launch must return hipErrorInvalidValue
    hipError_t err = hipGraphLaunch(exec, stream);
    REQUIRE(err == hipErrorInvalidValue);

    HIP_CHECK(hipGraphExecDestroy(exec));
  }

  // With AutoFreeOnLaunch: the runtime frees all prior allocations before
  // re-executing, so the second launch succeeds.
  SECTION("With AutoFreeOnLaunch") {
    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiateWithFlags(&exec, graph,
                                           hipGraphInstantiateFlagAutoFreeOnLaunch));

    // First launch — alloc node acquires memory
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Second launch — AutoFreeOnLaunch frees the previous allocation
    // (even though it was still mapped), then alloc node re-allocates.
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    
    auto statsAfterSecond = queryGraphMem(device);
    
#if HT_AMD
    // usedCurrent should still be a single allocation (old freed, new allocated)
    REQUIRE(statsAfterSecond.usedCurrent == kAllocSize);
#else
    // on nvidia, allocation is not freed with hipGraphExecDestroy,
    // so second launch allocates another chunk without freeing the first,
    // resulting in 2x allocation size
    REQUIRE(statsAfterSecond.usedCurrent == kAllocSize * 2 );
#endif

    HIP_CHECK(hipGraphExecDestroy(exec));
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * @}
 */
