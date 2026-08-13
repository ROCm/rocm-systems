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
#include <chrono>
#include <limits>
#include <thread>

/**
 * @addtogroup hipGraphAddMemAllocNode hipGraphAddMemAllocNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddMemAllocNode (hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t
 * *pDependencies, size_t numDependencies, hipMemAllocNodeParams *pNodeParams)` -
 * Creates a memory allocation node and adds it to a graph.
 */

static constexpr auto element_count{512 * 1024 * 1024};
static constexpr size_t kOneGiB{1024ULL * 1024ULL * 1024ULL};

static __global__ void validateGPU(int* const vec, const int value, size_t N, unsigned int* mismatch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    if (vec[idx] != value) {
      atomicAdd(mismatch, 1);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode behavior with invalid arguments:
 *    -# Null graph node
 *    -# Null graph node
 *    -# Invalid numDependencies for null list of dependencies
 *    -# Invalid numDependencies and valid list for dependencies
 *    -# Null alloc params
 *    -# Invalid poolProps alloc type
 *    -# Invalid poolProps location type
 *    -# Invalid poolProps location id
 *    -# Bytesize is max size_t
 *    -# Invalid accessDescCount
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Negative_Params) {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  hipGraphNode_t alloc_node;
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipGraphCreate(&graph, 0))

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev))

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = 0;
  desc.flags = hipMemAccessFlagsProtReadWrite;

  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = N;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;
  alloc_param.accessDescs = &desc;
  alloc_param.accessDescCount = 1;

  SECTION("Passing nullptr to graph node") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(nullptr, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, nullptr, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 11, &alloc_param),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to alloc params") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Passing invalid poolProps alloc type") {
    alloc_param.poolProps.allocType = hipMemAllocationTypeInvalid;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  }

  SECTION("Passing invalid poolProps location type") {
    alloc_param.poolProps.location.type = hipMemLocationTypeInvalid;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.poolProps.location.type = hipMemLocationTypeDevice;
  }

  SECTION("Passing invalid poolProps location id") {
    alloc_param.poolProps.location.id = num_dev;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.poolProps.location.id = 0;
  }

#if HT_NVIDIA  // EXSWHTEC-353
  SECTION("Passing max size_t bytesize") {
    alloc_param.bytesize = std::numeric_limits<size_t>::max();
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorOutOfMemory);
    alloc_param.bytesize = N;
  }

  SECTION("Passing invalid accessDescCount") {
    alloc_param.accessDescCount = num_dev + 1;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.accessDescCount = 0;
  }
#endif

  HIP_CHECK(hipGraphDestroy(graph))
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode unsupported behavior:
 *    -# More than one instantiation of the graph exist at the same time
 *    -# Clone graph with mem alloc node
 *    -# Use graph with mem alloc node in a child node
 *    -# Delete edge of the graph with mem alloc node
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Negative_NotSupported) {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  hipGraphNode_t alloc_node;

  HIP_CHECK(hipGraphCreate(&graph, 0))

  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = N;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param))

  SECTION("More than one instantation of the graph exists") {
    hipGraphExec_t graph_exec1, graph_exec2;
    HIP_CHECK(hipGraphInstantiate(&graph_exec1, graph, nullptr, nullptr, 0))
    HIP_CHECK_ERROR(hipGraphInstantiate(&graph_exec2, graph, nullptr, nullptr, 0),
                    hipErrorNotSupported);
    HIP_CHECK(hipGraphExecDestroy(graph_exec1))
  }

#if HT_NVIDIA  // EXSWHTEC-353
  SECTION("Clone graph with mem alloc node") {
    hipGraph_t cloned_graph;
    HIP_CHECK_ERROR(hipGraphClone(&cloned_graph, graph), hipErrorNotSupported);
  }

  SECTION("Use graph in a child node") {
    hipGraph_t parent_graph;
    HIP_CHECK(hipGraphCreate(&parent_graph, 0))
    hipGraphNode_t child_graph_node;
    HIP_CHECK_ERROR(hipGraphAddChildGraphNode(&child_graph_node, parent_graph, nullptr, 0, graph),
                    hipErrorNotSupported);
    HIP_CHECK(hipGraphDestroy(parent_graph))
  }

  SECTION("Delete edge of the graph") {
    hipGraphNode_t empty_node;
    HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, &alloc_node, 1))
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, &alloc_node, &empty_node, 1),
                    hipErrorNotSupported);
  }
#endif

  HIP_CHECK(hipGraphDestroy(graph))
}

/* Create graph with memory nodes that copies memset data to host array */
static void createGraph(hipGraphExec_t* graph_exec, int* A_h, int fill_value,
                        int** device_alloc = nullptr) {
  constexpr size_t num_bytes = element_count * sizeof(int);

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0))

  hipGraphNode_t alloc_node;
  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = num_bytes;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param))
  REQUIRE(alloc_param.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(alloc_param.dptr);

  hipGraphNode_t memset_node;
  hipMemsetParams memset_params{};
  memset(&memset_params, 0, sizeof(memset_params));
  memset_params.dst = reinterpret_cast<void*>(A_d);
  memset_params.value = fill_value;
  memset_params.pitch = 0;
  memset_params.elementSize = sizeof(int);
  memset_params.width = element_count;
  memset_params.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_node, graph, &alloc_node, 1, &memset_params))

  hipGraphNode_t memcpy_node;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, &memset_node, 1, A_h, A_d, num_bytes,
                                    hipMemcpyDeviceToHost));

  if (device_alloc == nullptr) {
    hipGraphNode_t free_node;
    HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, &memcpy_node, 1, (void*)A_d))
  } else {
    *device_alloc = A_d;
  }

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0))

  HIP_CHECK(hipGraphDestroy(graph))
}

static void createFreeGraph(hipGraphExec_t* graph_exec, int* device_alloc) {
  hipGraph_t graph;
  hipGraphNode_t free_node;
  HIP_CHECK(hipGraphCreate(&graph, 0))

  HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, nullptr, 0, (void*)device_alloc))

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0))

  HIP_CHECK(hipGraphDestroy(graph))
}

static size_t oversizedGraphAllocBytes() {
  size_t free_mem = 0;
  size_t total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem))
  (void)free_mem;
  return (total_mem > std::numeric_limits<size_t>::max() - kOneGiB)
             ? std::numeric_limits<size_t>::max()
             : total_mem + kOneGiB;
}

static void createOversizedMemAllocGraph(hipGraph_t* graph, hipGraphExec_t* graph_exec) {
  hipGraphNode_t alloc_node;
  hipMemAllocNodeParams alloc_param{};
  alloc_param.bytesize = oversizedGraphAllocBytes();
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphCreate(graph, 0))
  hipError_t ret = hipGraphAddMemAllocNode(&alloc_node, *graph, nullptr, 0, &alloc_param);
  if (ret == hipErrorOutOfMemory) {
    HIP_CHECK(hipGraphDestroy(*graph))
    HIP_SKIP_TEST("Oversized graph mem-alloc node failed before launch-time allocation.");
  }
  HIP_CHECK(ret)
  if (alloc_param.dptr == nullptr) {
    HIP_CHECK(hipGraphDestroy(*graph))
    FAIL("hipGraphAddMemAllocNode succeeded but returned a null device pointer.");
  }
  HIP_CHECK(hipGraphInstantiate(graph_exec, *graph, nullptr, nullptr, 0))
}

static hipError_t queryUntilComplete(hipStream_t stream) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  hipError_t ret = hipErrorNotReady;
  do {
    ret = hipStreamQuery(stream);
    if (ret != hipErrorNotReady) {
      return ret;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return ret;
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify launch-time graph mem-alloc node failures are surfaced as out-of-memory
 *    errors either directly at graph launch or through stream completion APIs.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Negative_LaunchOutOfMemory) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }
  HIP_CHECK(hipSetDevice(0))

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  createOversizedMemAllocGraph(&graph, &graph_exec);

  SECTION("hipStreamSynchronize") {
    StreamGuard stream_guard(Streams::created);
    hipStream_t stream = stream_guard.stream();
    hipError_t launch_ret = hipGraphLaunch(graph_exec, stream);
    hipError_t sync_ret = hipStreamSynchronize(stream);
    HIP_CHECK(hipGraphExecDestroy(graph_exec))
    HIP_CHECK(hipGraphDestroy(graph))
    HIP_CHECK(hipDeviceGraphMemTrim(0))
    REQUIRE((launch_ret == hipSuccess || launch_ret == hipErrorOutOfMemory));
    REQUIRE(sync_ret == hipErrorOutOfMemory);
  }

  SECTION("hipStreamSynchronize nullptr") {
    hipError_t launch_ret = hipGraphLaunch(graph_exec, nullptr);
    hipError_t sync_ret = hipStreamSynchronize(nullptr);
    HIP_CHECK(hipGraphExecDestroy(graph_exec))
    HIP_CHECK(hipGraphDestroy(graph))
    HIP_CHECK(hipDeviceGraphMemTrim(0))
    REQUIRE((launch_ret == hipSuccess || launch_ret == hipErrorOutOfMemory));
    REQUIRE(sync_ret == hipErrorOutOfMemory);
  }

  SECTION("hipStreamSynchronize nullptr observes blocking stream error") {
    StreamGuard stream_guard(Streams::created);
    hipStream_t stream = stream_guard.stream();
    hipError_t launch_ret = hipGraphLaunch(graph_exec, stream);
    hipError_t sync_ret = hipStreamSynchronize(nullptr);
    HIP_CHECK(hipGraphExecDestroy(graph_exec))
    HIP_CHECK(hipGraphDestroy(graph))
    HIP_CHECK(hipDeviceGraphMemTrim(0))
    REQUIRE((launch_ret == hipSuccess || launch_ret == hipErrorOutOfMemory));
    REQUIRE(sync_ret == hipErrorOutOfMemory);
  }

  SECTION("hipStreamQuery") {
    StreamGuard stream_guard(Streams::created);
    hipStream_t stream = stream_guard.stream();
    hipError_t ret = hipGraphLaunch(graph_exec, stream);
    hipError_t second_query = hipSuccess;
    if (ret == hipSuccess) {
      ret = queryUntilComplete(stream);
      if (ret == hipErrorNotReady) {
        HIP_CHECK(hipGraphExecDestroy(graph_exec))
        HIP_CHECK(hipGraphDestroy(graph))
        FAIL("Timed out waiting for graph launch to complete.");
      }
      second_query = hipStreamQuery(stream);
    }
    HIP_CHECK(hipGraphExecDestroy(graph_exec))
    HIP_CHECK(hipGraphDestroy(graph))
    HIP_CHECK(hipDeviceGraphMemTrim(0))
    REQUIRE(ret == hipErrorOutOfMemory);
    REQUIRE(second_query == hipSuccess);
  }

  SECTION("hipStreamSynchronize nullptr drains every blocking stream") {
    StreamGuard stream_guard_a(Streams::created);
    hipStream_t stream_a = stream_guard_a.stream();
    StreamGuard stream_guard_b(Streams::created);
    hipStream_t stream_b = stream_guard_b.stream();

    hipGraph_t graph_b = nullptr;
    hipGraphExec_t graph_exec_b = nullptr;
    createOversizedMemAllocGraph(&graph_b, &graph_exec_b);

    hipError_t launch_ret_a = hipGraphLaunch(graph_exec, stream_a);
    hipError_t launch_ret_b = hipGraphLaunch(graph_exec_b, stream_b);
    hipError_t sync_ret = hipStreamSynchronize(nullptr);
    // Work on both streams is complete after the sync, so neither query waits.
    hipError_t query_a = hipStreamQuery(stream_a);
    hipError_t query_b = hipStreamQuery(stream_b);
    HIP_CHECK(hipGraphExecDestroy(graph_exec))
    HIP_CHECK(hipGraphDestroy(graph))
    HIP_CHECK(hipGraphExecDestroy(graph_exec_b))
    HIP_CHECK(hipGraphDestroy(graph_b))
    HIP_CHECK(hipDeviceGraphMemTrim(0))
    REQUIRE((launch_ret_a == hipSuccess || launch_ret_a == hipErrorOutOfMemory));
    REQUIRE((launch_ret_b == hipSuccess || launch_ret_b == hipErrorOutOfMemory));
    REQUIRE(sync_ret == hipErrorOutOfMemory);
    // Null-stream sync must clear BOTH streams' latched errors, not just the one reported.
    REQUIRE(query_a == hipSuccess);
    REQUIRE(query_b == hipSuccess);
  }
}

static void createOversizedMemAllocChildGraph(hipGraph_t* parent_graph,
                                               hipGraphExec_t* graph_exec) {
  hipGraphNode_t alloc_node;
  hipMemAllocNodeParams alloc_param{};
  alloc_param.bytesize = oversizedGraphAllocBytes();
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  hipGraph_t child_graph;
  HIP_CHECK(hipGraphCreate(&child_graph, 0))
  hipError_t ret = hipGraphAddMemAllocNode(&alloc_node, child_graph, nullptr, 0, &alloc_param);
  if (ret == hipErrorOutOfMemory) {
    HIP_CHECK(hipGraphDestroy(child_graph))
    HIP_SKIP_TEST("Oversized graph mem-alloc node failed before launch-time allocation.");
  }
  HIP_CHECK(ret)
  if (alloc_param.dptr == nullptr) {
    HIP_CHECK(hipGraphDestroy(child_graph))
    FAIL("hipGraphAddMemAllocNode succeeded but returned a null device pointer.");
  }

  HIP_CHECK(hipGraphCreate(parent_graph, 0))
  hipGraphNode_t child_node;
  // The child graph is cloned into the parent, so its handle can be freed immediately.
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, *parent_graph, nullptr, 0, child_graph))
  HIP_CHECK(hipGraphDestroy(child_graph))

  HIP_CHECK(hipGraphInstantiate(graph_exec, *parent_graph, nullptr, nullptr, 0))
}

/**
 * Test Description
 * ------------------------
 *  - Test that a launch-time mem-alloc OOM inside a child graph node is surfaced like a
 *    top-level GraphMemAllocNode. Regression test for ChildGraphNode::EnqueueCommands's
 *    classic path, which used to drop CreateCommand()/GetEnqueueStatus() failures.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Negative_LaunchOutOfMemory_ChildGraph) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }
  HIP_CHECK(hipSetDevice(0))

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  createOversizedMemAllocChildGraph(&graph, &graph_exec);

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();
  hipError_t launch_ret = hipGraphLaunch(graph_exec, stream);
  hipError_t sync_ret = hipStreamSynchronize(stream);
  HIP_CHECK(hipGraphExecDestroy(graph_exec))
  HIP_CHECK(hipGraphDestroy(graph))
  HIP_CHECK(hipDeviceGraphMemTrim(0))
  REQUIRE((launch_ret == hipSuccess || launch_ret == hipErrorOutOfMemory));
  REQUIRE(sync_ret == hipErrorOutOfMemory);
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly and graph behaves as
 * expected when free node is added to the same graph.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Positive_FreeInGraph) {
  hipGraphExec_t graph_exec;

  LinearAllocGuard<int> host_alloc =
      LinearAllocGuard<int>(LinearAllocs::malloc, element_count * sizeof(int));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr int fill_value = 11;
  createGraph(&graph_exec, host_alloc.ptr(), fill_value);
  HIP_CHECK(hipGraphLaunch(graph_exec, stream))
  HIP_CHECK(hipStreamSynchronize(stream))

  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec))
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly, graph behaves as expected
 * and allocated memory can can be accessed by outside the graph before memory is freed outside the
 * stream.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Positive_FreeOutsideStream) {
  hipGraphExec_t graph_exec;

  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, element_count * sizeof(int));
  LinearAllocGuard<unsigned int> mismatch_count_h =
      LinearAllocGuard<unsigned int>(LinearAllocs::malloc, sizeof(unsigned int));
  LinearAllocGuard<unsigned int> mismatch_count_d =
      LinearAllocGuard<unsigned int>(LinearAllocs::hipMalloc, sizeof(unsigned int));
  HIP_CHECK(hipMemset(mismatch_count_d.ptr(), 0, sizeof(unsigned int)))
  int* dev_p;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int fill_value = 12;

  createGraph(&graph_exec, host_alloc.ptr(), fill_value, &dev_p);
  HIP_CHECK(hipGraphLaunch(graph_exec, stream))
  validateGPU<<<block_count, thread_count, 0, stream>>>(dev_p, fill_value, element_count,
                                                        mismatch_count_d.ptr());
  // Since hipFree is synchronous, the stream must synchronize before freeing dev_p
  HIP_CHECK(hipStreamSynchronize(stream))
  HIP_CHECK(hipFree(dev_p))

  HIP_CHECK(hipMemcpy(mismatch_count_h.host_ptr(), mismatch_count_d.ptr(), sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
  REQUIRE(mismatch_count_h.host_ptr()[0] == 0);
  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec))
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly, graph behaves as expected
 * and allocated memory can can be accessed by outside the graph before memory is freed.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Positive_FreeOutsideGraph) {
  hipGraphExec_t graph_exec;

  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, element_count * sizeof(int));
  LinearAllocGuard<unsigned int> mismatch_count_h =
      LinearAllocGuard<unsigned int>(LinearAllocs::malloc, sizeof(unsigned int));
  LinearAllocGuard<unsigned int> mismatch_count_d =
      LinearAllocGuard<unsigned int>(LinearAllocs::hipMalloc, sizeof(unsigned int));
  HIP_CHECK(hipMemset(mismatch_count_d.ptr(), 0, sizeof(unsigned int)))
  int* dev_p;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int fill_value = 13;

  createGraph(&graph_exec, host_alloc.ptr(), fill_value, &dev_p);
  HIP_CHECK(hipGraphLaunch(graph_exec, stream))
  validateGPU<<<block_count, thread_count, 0, stream>>>(dev_p, fill_value, element_count,
                                                        mismatch_count_d.ptr());
  HIP_CHECK(hipFreeAsync(dev_p, stream))
  HIP_CHECK(hipStreamSynchronize(stream))

  HIP_CHECK(hipMemcpy(mismatch_count_h.host_ptr(), mismatch_count_d.ptr(), sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
  REQUIRE(mismatch_count_h.host_ptr()[0] == 0);
  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec))
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly, graph behaves as expected
 * and allocated memory can can be accessed by outside the graph before memory is freed in a
 * different graph.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Positive_FreeSeparateGraph) {
  hipGraphExec_t graph_exec1, graph_exec2;

  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, element_count * sizeof(int));
  LinearAllocGuard<unsigned int> mismatch_count_h =
      LinearAllocGuard<unsigned int>(LinearAllocs::malloc, sizeof(unsigned int));
  LinearAllocGuard<unsigned int> mismatch_count_d =
      LinearAllocGuard<unsigned int>(LinearAllocs::hipMalloc, sizeof(unsigned int));
  HIP_CHECK(hipMemset(mismatch_count_d.ptr(), 0, sizeof(unsigned int)))
  int* dev_p;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int fill_value = 13;

  createGraph(&graph_exec1, host_alloc.ptr(), fill_value, &dev_p);
  createFreeGraph(&graph_exec2, dev_p);
  HIP_CHECK(hipGraphLaunch(graph_exec1, stream))
  validateGPU<<<block_count, thread_count, 0, stream>>>(dev_p, fill_value, element_count,
                                                        mismatch_count_d.ptr());
  HIP_CHECK(hipGraphLaunch(graph_exec2, stream))
  HIP_CHECK(hipStreamSynchronize(stream))

  HIP_CHECK(hipMemcpy(mismatch_count_h.host_ptr(), mismatch_count_d.ptr(), sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
  REQUIRE(mismatch_count_h.host_ptr()[0] == 0);
  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec1))
  HIP_CHECK(hipGraphExecDestroy(graph_exec2))
}

/**
 * Test Description
 * ------------------------
 *  - Create a graph and add 3 node with hipGraphAddMemAllocNode,
 * initialize the memory allocated for 2 node.
 * Add kernel node which will do vectorADD for these 2 allocated
 * node and copy the result 3rd allocated node and verify.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Functional_1) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t N = 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  size_t NElem{N};

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipGraphNode_t allocNodeA, freeNodeA, allocNodeB, freeNodeB;
  hipGraphNode_t allocNodeC, freeNodeC;
  hipMemAllocNodeParams allocParam;

  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays<int>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0))
  HIP_CHECK(hipStreamCreate(&stream))

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam))
  REQUIRE(allocParam.dptr != nullptr);
  A_d = reinterpret_cast<int*>(allocParam.dptr);
  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeB, graph, &allocNodeA, 1, &allocParam))
  REQUIRE(allocParam.dptr != nullptr);
  B_d = reinterpret_cast<int*>(allocParam.dptr);
  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeC, graph, &allocNodeB, 1, &allocParam))
  REQUIRE(allocParam.dptr != nullptr);
  C_d = reinterpret_cast<int*>(allocParam.dptr);

  // Check shows that A_d, B_d & C_d DON'T share any virtual address each other
  REQUIRE(A_d != B_d);
  REQUIRE(B_d != C_d);
  REQUIRE(A_d != C_d);

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, &allocNodeC, 1, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, &allocNodeC, 1, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kernelNodeParams))

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1))
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1))

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, &kernel_vecAdd, 1, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(
      hipGraphAddMemFreeNode(&freeNodeA, graph, &memcpyD2H_C, 1, reinterpret_cast<void*>(A_d)));
  HIP_CHECK(
      hipGraphAddMemFreeNode(&freeNodeB, graph, &memcpyD2H_C, 1, reinterpret_cast<void*>(B_d)));
  HIP_CHECK(
      hipGraphAddMemFreeNode(&freeNodeC, graph, &memcpyD2H_C, 1, reinterpret_cast<void*>(C_d)));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))
  HIP_CHECK(hipGraphLaunch(graphExec, stream))
  HIP_CHECK(hipStreamSynchronize(stream))

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays<int>(nullptr, nullptr, nullptr, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph))
  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipStreamDestroy(stream))
  HIP_CHECK(hipDeviceGraphMemTrim(0))
}

/**
* Test Description
* ------------------------
*  - Create a graph and add a node with hipGraphAddMemAllocNode and
* hipGraphAddMemFreeNode and launch it. Validate memory allocation
  pointer shouldn't be null.
* Test source
* ------------------------
*  - /unit/graph/hipGraphAddMemAllocNode.cc
* Test requirements
* ------------------------
*  - HIP_VERSION >= 6.1
*/
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Functional_2) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t Nbytes = 512 * 1024 * 1024;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA;
  hipMemAllocNodeParams allocParam;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices))

  for (int i = 0; i < numDevices; ++i) {
    HIP_CHECK(hipSetDevice(i))

    size_t before = 0, after = 0;
    HIP_CHECK(hipDeviceGraphMemTrim(i));
    HIP_CHECK(hipDeviceGetGraphMemAttribute(i, hipGraphMemAttrUsedMemCurrent, &before));

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamCreate(&stream));

    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = i;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam))
    REQUIRE(allocParam.dptr != nullptr);
    HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1,
                                     reinterpret_cast<void*>(allocParam.dptr)));

    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(stream));

    REQUIRE(before == after);
  }
}

/**
 * Test Description
 * ------------------------
 * - Create a graph1 with hipGraphAddMemAllocNode and free it
 * on different graph2 with hipGraphAddMemFreeNode.
 * Test source
 * ------------------------
 * - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Functional_3) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t Nbytes = 512 * 1024 * 1024;
  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec1, graphExec2;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA;
  hipMemAllocNodeParams allocParam;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices))

  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i))

    size_t before = 0, after = 0;
    HIP_CHECK(hipDeviceGraphMemTrim(i));
    HIP_CHECK(hipDeviceGetGraphMemAttribute(i, hipGraphMemAttrUsedMemCurrent, &before));

    HIP_CHECK(hipGraphCreate(&graph1, 0));
    HIP_CHECK(hipGraphCreate(&graph2, 0));
    HIP_CHECK(hipStreamCreate(&stream));

    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = i;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph1, nullptr, 0, &allocParam))
    REQUIRE(allocParam.dptr != nullptr);
    HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph2, nullptr, 0,
                                     reinterpret_cast<void*>(allocParam.dptr)));

    HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0))
    HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0))

    HIP_CHECK(hipGraphLaunch(graphExec1, stream));
    HIP_CHECK(hipGraphLaunch(graphExec2, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipDeviceGraphMemTrim(i))
    HIP_CHECK(hipDeviceGetGraphMemAttribute(i, hipGraphMemAttrUsedMemCurrent, &after))

    HIP_CHECK(hipGraphDestroy(graph1))
    HIP_CHECK(hipGraphDestroy(graph2))
    HIP_CHECK(hipGraphExecDestroy(graphExec1))
    HIP_CHECK(hipGraphExecDestroy(graphExec2))
    HIP_CHECK(hipStreamDestroy(stream))

    REQUIRE(before == after);
  }
}

/**
 * Test Description
 * ------------------------
 * - Create a graph1 with hipGraphAddMemAllocNode and free it with
 * hipMemFreeAsync or hipMemFree.
 * Test source
 * ------------------------
 * - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Functional_4) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t Nbytes = 512 * 1024 * 1024;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t allocNodeA;
  hipMemAllocNodeParams allocParam;

  void* temp;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices))

  for (int i = 0; i < numDevices; ++i) {
    HIP_CHECK(hipSetDevice(i))

    size_t before = 0, after = 0;
    HIP_CHECK(hipDeviceGraphMemTrim(i));
    HIP_CHECK(hipDeviceGetGraphMemAttribute(i, hipGraphMemAttrUsedMemCurrent, &before));

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamCreate(&stream));

    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = i;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam))
    temp = allocParam.dptr;
    REQUIRE(temp != nullptr);

    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipGraphDestroy(graph))
    HIP_CHECK(hipGraphExecDestroy(graphExec))
    HIP_CHECK(hipStreamDestroy(stream))
    HIP_CHECK(hipDeviceGraphMemTrim(i))

    HIP_CHECK(hipFree(temp))
    HIP_CHECK(hipDeviceGraphMemTrim(i))
    HIP_CHECK(hipDeviceGetGraphMemAttribute(i, hipGraphMemAttrUsedMemCurrent, &after))

    REQUIRE(before == after);
  }
}
/**
* Test Description
* ------------------------
*  Negative Test for API hipGraphAddMemAllocNode - Argument validation check
1) Pass allocNode as nullptr.
2) Pass allocNode as empty structure.
3) Pass graph as nullptr.
4) Pass graph as empty structure.
5) Pass Dependencied node as null & number of dependencies as non zero.
6) Pass Dependencied node as valid & number of dependencies as 0
7) Pass allocParam as nullptr.
8) Pass allocParam as empty structure.
9) Pass allocParam.bytesize as zero.
10) Pass allocParam.bytesize as INT_MAX.
11) Pass allocParam.poolProps.allocType as hipMemAllocationTypeInvalid.
12) Pass allocParam.poolProps.allocType as hipMemAllocationTypeMax.
13) Pass allocParam.poolProps.allocType as some unavailable id as -1
14) Pass allocParam.poolProps.allocType as some unavailable id as INT_MAX
15) Pass allocParam.poolProps.location.id as -1.
16) Pass allocParam.poolProps.location.id as INT_MAX.
17) Pass allocParam.poolProps.location.type as hipMemLocationTypeInvalid
18) Pass allocParam.poolProps.location.type as hipMemLocationTypeDevice
19) Pass allocParam.poolProps.location.type as some unavailable id as -1
20) Pass allocParam.poolProps.location.type as some unavailable id as INT_MAX
* Test source
* ------------------------
*  - /unit/graph/hipGraphAddMemAllocNode.cc
* Test requirements
* ------------------------
*  - HIP_VERSION >= 6.1
*/
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Argument_Check) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t Nbytes = 512 * 1024 * 1024;
  hipGraph_t graph;
  hipGraphNode_t emptyNode, allocNodeA;
  hipMemAllocNodeParams allocParam;
  hipError_t ret;

  HIP_CHECK(hipGraphCreate(&graph, 0))
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0))

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam))
  REQUIRE(allocParam.dptr != nullptr);

  SECTION("Pass allocNode as nullptr.") {
    ret = hipGraphAddMemAllocNode(nullptr, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass allocNode as empty structure.") {
    hipGraphNode_t allocNodeT{};
    ret = hipGraphAddMemAllocNode(&allocNodeT, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipSuccess);
    REQUIRE(allocParam.dptr != nullptr);
  }
  SECTION("Pass graph as nullptr.") {
    ret = hipGraphAddMemAllocNode(&allocNodeA, nullptr, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass graph as empty structure.") {
    hipGraph_t graphT{};
    ret = hipGraphAddMemAllocNode(&allocNodeA, graphT, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Dependencies node as nullptr & number of dependencies as nonzero") {
    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 1, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Dependencied node as valid & number of dependencies as 0") {
    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, &emptyNode, 0, &allocParam);
    REQUIRE(ret == hipSuccess);
    REQUIRE(allocParam.dptr != nullptr);
  }
  SECTION("Pass allocParam as nullptr.") {
    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }
#if HT_NVIDIA
  SECTION("Pass allocParam as empty structure.") {
    hipMemAllocNodeParams allocParamT{};
    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParamT);
    REQUIRE(ret == hipErrorInvalidValue);
  }
#endif
  SECTION("Pass allocParam.bytesize as INT_MAX.") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = INT_MAX;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipSuccess);
    REQUIRE(allocParam.dptr != nullptr);
  }
#if HT_NVIDIA
  SECTION("Pass allocParam.bytesize as zero.") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = 0;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
  SECTION("Pass poolProps.allocType as hipMemAllocationTypeInvalid.") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypeInvalid;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(allocParam.dptr == nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass allocParam.poolProps.allocType as hipMemAllocationTypeMax") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypeMax;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
  SECTION("Pass allocParam.poolProps.allocType id as -1") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationType(-1);
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
  SECTION("Pass allocParam.poolProps.allocType id as INT_MAX") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationType(INT_MAX);
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
  SECTION("Pass allocParam.poolProps.location.id as -1.") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = -1;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
  SECTION("Pass allocParam.poolProps.location.id as INT_MAX.") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = INT_MAX;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
  SECTION("allocParam.poolProps.location.type as hipMemLocationTypeInvalid") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeInvalid;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
#endif
  SECTION("allocParam.poolProps.location.type as hipMemLocationTypeDevice") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationTypeDevice;

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipSuccess);
    REQUIRE(allocParam.dptr != nullptr);
  }
#if HT_NVIDIA
  SECTION("Pass allocParam.poolProps.location.type some unavailable id -1.") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationType(-1);

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
  SECTION("Pass allocParam.poolProps.location.type id INT_MAX") {
    memset(&allocParam, 0, sizeof(allocParam));
    allocParam.bytesize = Nbytes;
    allocParam.poolProps.allocType = hipMemAllocationTypePinned;
    allocParam.poolProps.location.id = 0;
    allocParam.poolProps.location.type = hipMemLocationType(INT_MAX);

    ret = hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam);
    REQUIRE(ret == hipErrorInvalidValue);
    REQUIRE(allocParam.dptr == nullptr);
  }
#endif
  HIP_CHECK(hipGraphDestroy(graph))
  HIP_CHECK(hipDeviceGraphMemTrim(0))
}

/**
 * Test Description
 * ------------------------
 * - Negative Test for API hipGraphAddMemAllocNode & hipGraphAddMemFreeNode
 * Create a graph with alloc and free node, Instanciate the graph twice and
 * validate that instantiate should return error.
 * Test source
 * ------------------------
 * - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Negative_Instanciate_Graph_Again) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t Nbytes = 512 * 1024 * 1024;
  hipGraph_t graph;
  hipGraphExec_t graphExec, graphExec1;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA;
  hipMemAllocNodeParams allocParam;
  hipError_t ret;

  HIP_CHECK(hipDeviceGraphMemTrim(0))
  HIP_CHECK(hipGraphCreate(&graph, 0))
  HIP_CHECK(hipStreamCreate(&stream))

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam))
  REQUIRE(allocParam.dptr != nullptr);
  void* temp = allocParam.dptr;

  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1, temp))

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))

  // Instanciate 2nd time with same graph, it should give error
  ret = hipGraphInstantiate(&graphExec1, graph, nullptr, nullptr, 0);
  REQUIRE(ret == hipErrorNotSupported);

  HIP_CHECK(hipGraphLaunch(graphExec, stream))
  HIP_CHECK(hipStreamSynchronize(stream))

  HIP_CHECK(hipGraphDestroy(graph))
  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipStreamDestroy(stream))
  HIP_CHECK(hipDeviceGraphMemTrim(0))
}

/**
* Test Description
* ------------------------
* - Negative Test for API hipGraphAddMemAllocNode & hipGraphAddMemFreeNode
* Create a graph with alloc and free node, try to free the same
  alloc pointer again using hipFree(), it should return error.
* Test source
* ------------------------
* - /unit/graph/hipGraphAddMemAllocNode.cc
* Test requirements
* ------------------------
* - HIP_VERSION >= 6.1
*/
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Negative_Free_Alloc_Memory_Again) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t Nbytes = 512 * 1024 * 1024;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA;
  hipMemAllocNodeParams allocParam;
  hipError_t ret;

  HIP_CHECK(hipDeviceGraphMemTrim(0))
  HIP_CHECK(hipGraphCreate(&graph, 0))
  HIP_CHECK(hipStreamCreate(&stream))

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam))
  REQUIRE(allocParam.dptr != nullptr);
  void* temp = allocParam.dptr;

  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1, temp))

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))

  HIP_CHECK(hipGraphLaunch(graphExec, stream))
  HIP_CHECK(hipStreamSynchronize(stream))

  // Free alloc pointer manually again, it should give error
  ret = hipFree(temp);
  REQUIRE(ret == hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(graph))
  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipStreamDestroy(stream))
  HIP_CHECK(hipDeviceGraphMemTrim(0))
}

/**
* Test Description
* ------------------------
*  Negative Test for API hipGraphAddMemAllocNode & hipGraphAddMemFreeNode
   1) Once free node added in any graph then we can't added it again.
   2) Clone graph should give error if a graph contain MemAlloc node.
   3) Clone graph should give error if a graph contain MemFree node.
   4) Destroy the MemAlloc node which was added in the graph.
   5) Destroy the MemFree node which was added in the graph.
* Test source
* ------------------------
* - /unit/graph/hipGraphAddMemAllocNode.cc
* Test requirements
* ------------------------
* - HIP_VERSION >= 6.1
*/
HIP_TEST_CASE(Unit_hipGraphAddMemAllocNode_Negative_With_Cloneed_Graph) {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0))
  if (!mem_pool_support) {
    HIP_SKIP_TEST("Runtime doesn't support Memory Pool. Skip the test case.");
  }

  constexpr size_t N = 512 * 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, graph_2, clone_graph, clone_graph_2;
  hipGraphExec_t graphExec, graphExec_2;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA, freeNodeB = nullptr;
  hipMemAllocNodeParams allocParam;
  hipError_t ret;

  HIP_CHECK(hipGraphCreate(&graph, 0))
  HIP_CHECK(hipGraphCreate(&graph_2, 0))
  HIP_CHECK(hipStreamCreate(&stream))

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParam))
  REQUIRE(allocParam.dptr != nullptr);
  void* temp = allocParam.dptr;

  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph_2, nullptr, 0, temp))

  SECTION("Once free node added in any graph then we can't added it again") {
    hipError_t ret1 = hipGraphAddMemFreeNode(&freeNodeB, graph, nullptr, 0, temp);
    REQUIRE(ret1 == hipErrorInvalidValue);
  }
  SECTION("Clone graph should give error if a graph contain memalloc node") {
    ret = hipGraphClone(&clone_graph, graph);
    REQUIRE(ret == hipErrorNotSupported);
  }
  SECTION("Clone graph should give error if a graph contain memfree node") {
    ret = hipGraphClone(&clone_graph_2, graph_2);
    REQUIRE(ret == hipErrorNotSupported);
  }
  SECTION("Destroy the MemAlloc node which was added in the graph") {
    ret = hipGraphDestroyNode(allocNodeA);
    REQUIRE(ret == hipErrorNotSupported);
  }
#if HT_AMD
  SECTION("Destroy the MemFree node which was added in the graph") {
    ret = hipGraphDestroyNode(freeNodeB);
    REQUIRE(ret == hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))
  HIP_CHECK(hipGraphLaunch(graphExec, stream))
  HIP_CHECK(hipGraphInstantiate(&graphExec_2, graph_2, nullptr, nullptr, 0))
  HIP_CHECK(hipGraphLaunch(graphExec_2, stream))
  HIP_CHECK(hipStreamSynchronize(stream))

  HIP_CHECK(hipGraphDestroy(graph))
  HIP_CHECK(hipGraphDestroy(graph_2))
  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipGraphExecDestroy(graphExec_2))
  HIP_CHECK(hipStreamDestroy(stream))
}
/**
 * End doxygen group GraphTest.
 * @}
 */
