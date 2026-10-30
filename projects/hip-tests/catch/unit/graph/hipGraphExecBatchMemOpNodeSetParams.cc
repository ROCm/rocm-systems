/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_memop.hh>
#include <resource_guards.hh>
/**
 * @addtogroup hipGraphExecBatchMemOpNodeSetParams
 hipGraphExecBatchMemOpNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec,
 hipGraphNode_t hNode, const hipBatchMemOpNodeParams* nodeParams)`
 * - Sets the parameters for a batch mem op node in the given graphExec
 */

/**
 * Test Description
 * ------------------------
 * - Verify the Negative cases of the hipGraphExecBatchMemOpNodeSetParams API.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecBatchMemOpNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipGraphExecBatchMemOpNodeSetParams_NegativeTsts) {
  HIP_CHECK(hipInit(0));
  hipGraph_t graph, graph1;
  hipGraphExec_t graphExec;
  hipCtx_t ctx;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));
  // Create a HIP graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  INFO("Graph created.");

  static hipStreamBatchMemOpParams paramArray[2], newParamArray[2];
  std::vector<hipDeviceptr_t> opsArray(1);
  HIP_CHECK(hipMalloc((void**)&opsArray[0], sizeof(uint32_t)));

  paramArray[0].operation = hipStreamMemOpWriteValue32;
  paramArray[0].writeValue.address = opsArray[0];
  paramArray[0].writeValue.value = 1000;
  paramArray[0].writeValue.flags = 0x0;
  // paramArray[0].writeValue.alias = 0;

  paramArray[1].operation = hipStreamMemOpWaitValue32;
  paramArray[1].waitValue.address = opsArray[0];
  paramArray[1].waitValue.value = 1000;
  paramArray[1].waitValue.flags = hipStreamWaitValueEq;
  // paramArray[i].waitValue.alias = 0;

  int totalOps = 2;
  // Setup the batch memory operation node parameters
  hipBatchMemOpNodeParams batchNodeParams;
  batchNodeParams.ctx = ctx;                // Use the current HIP context
  batchNodeParams.count = totalOps;         // Total number of memory operations
  batchNodeParams.paramArray = paramArray;  // Pointer to the array of memory operations
  batchNodeParams.flags = 0;                // No special flags

  // Add a batch memory operation node to the graph
  hipGraphNode_t batchMemOpNode, batchMemOpNode_1;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&batchMemOpNode, graph, nullptr, 0, &batchNodeParams));
  HIP_CHECK(hipGraphAddBatchMemOpNode(&batchMemOpNode_1, graph1, nullptr, 0, &batchNodeParams));
  INFO("hipGraphAddBatchMemOpNode added successfully.");

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  INFO("Graph instantiated.");
  for (int i = 0; i < totalOps; i++) {
    newParamArray[i] = paramArray[i];
  }
  newParamArray[0].writeValue.value = 2000;
  newParamArray[1].waitValue.flags = hipStreamWaitValueGte;

  hipBatchMemOpNodeParams newBatchNodeParams;
  newBatchNodeParams.ctx = ctx;
  newBatchNodeParams.count = totalOps;
  newBatchNodeParams.paramArray = newParamArray;
  newBatchNodeParams.flags = 0;

  hipBatchMemOpNodeParams invalidNewBatchNodeParams;
  invalidNewBatchNodeParams.ctx = ctx;
  invalidNewBatchNodeParams.count = 400;
  invalidNewBatchNodeParams.paramArray = newParamArray;
  invalidNewBatchNodeParams.flags = -4;

  SECTION("Graph Executable as nullptr") {
    HIP_CHECK_ERROR(
        hipGraphExecBatchMemOpNodeSetParams(nullptr, batchMemOpNode, &newBatchNodeParams),
        hipErrorInvalidValue);
  }

  SECTION("Batch Memory Node as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(graphExec, nullptr, &newBatchNodeParams),
                    hipErrorInvalidValue);
  }
// Disabled for NVIDIA due to the defect SWDEV-502247
#if HT_AMD
  SECTION("Batch node Parameters as nullptr") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(graphExec, batchMemOpNode, nullptr),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("Irrelevant Batch Node") {
    HIP_CHECK_ERROR(
        hipGraphExecBatchMemOpNodeSetParams(graphExec, batchMemOpNode_1, &newBatchNodeParams),
        hipErrorInvalidValue);
  }
// Disabled due to defect SWDEV-502219
#if 0
  SECTION("Invalid Batch Node Params") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(
                        graphExec, batchMemOpNode, &invalidNewBatchNodeParams),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("Unchanged Batch node Parameters") {
    HIP_CHECK_ERROR(
        hipGraphExecBatchMemOpNodeSetParams(graphExec, batchMemOpNode, &batchNodeParams),
        hipSuccess);
  }
  HIP_CHECK(hipFree((void*)opsArray[0]));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipCtxDestroy(ctx));
}
__global__ void BatchMemOpTypeGateKernel(int* output) { output[0] = 1; }

/**
 * Test Description
 * ------------------------
 * - Verify that the batch-mem-op node APIs reject handles of other node types instead of
 *   reinterpreting them (the node-type gate), for the template, exec and query entry points.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecBatchMemOpNodeSetParams.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecBatchMemOpNodeSetParams_WrongNodeType) {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t ctx;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipDeviceptr_t opsAddress;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&opsAddress), sizeof(uint32_t)));
  int* kernelOutput = nullptr;
  HIP_CHECK(hipMalloc(&kernelOutput, sizeof(int)));

  hipStreamBatchMemOpParams paramArray[1] = {};
  paramArray[0].operation = hipStreamMemOpWriteValue32;
  paramArray[0].writeValue.address = opsAddress;
  paramArray[0].writeValue.value = 1000;
  paramArray[0].writeValue.flags = 0;
  hipBatchMemOpNodeParams batchNodeParams = {};
  batchNodeParams.ctx = ctx;
  batchNodeParams.count = 1;
  batchNodeParams.paramArray = paramArray;
  batchNodeParams.flags = 0;

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t batchNode, kernelNode;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&batchNode, graph, nullptr, 0, &batchNodeParams));
  void* args[] = {&kernelOutput};
  hipKernelNodeParams kernelParams{};
  kernelParams.func = reinterpret_cast<void*>(BatchMemOpTypeGateKernel);
  kernelParams.gridDim = dim3(1);
  kernelParams.blockDim = dim3(1);
  kernelParams.kernelParams = args;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams));
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Exec setter with a kernel node handle") {
    HIP_CHECK_ERROR(hipGraphExecBatchMemOpNodeSetParams(graphExec, kernelNode, &batchNodeParams),
                    hipErrorInvalidValue);
  }
  SECTION("Template setter with a kernel node handle") {
    HIP_CHECK_ERROR(hipGraphBatchMemOpNodeSetParams(kernelNode, &batchNodeParams),
                    hipErrorInvalidValue);
  }
  SECTION("Getter with a kernel node handle") {
    hipBatchMemOpNodeParams out = {};
    HIP_CHECK_ERROR(hipGraphBatchMemOpNodeGetParams(kernelNode, &out), hipErrorInvalidValue);
  }
  SECTION("Batch node itself still accepted") {
    HIP_CHECK(hipGraphExecBatchMemOpNodeSetParams(graphExec, batchNode, &batchNodeParams));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(kernelOutput));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(opsAddress)));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 * - Verify that hipGraphExecBatchMemOpNodeSetParams takes effect on the next launch (the
 *   instantiated packets are refreshed, not replayed with the old op array).
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecBatchMemOpNodeSetParams.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecBatchMemOpNodeSetParams_UpdateTakesEffect) {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  int waitValueSupport = 0;
  auto attrErr =
      hipDeviceGetAttribute(&waitValueSupport, hipDeviceAttributeCanUseStreamWaitValue, 0);
  if (attrErr != hipSuccess || waitValueSupport == 0) {
    HIP_SKIP_TEST("hipStreamWaitValue is not supported on this device.");
  }
#if !HT_AMD
  HIP_SKIP_TEST("hipMallocSignalMemory is not supported on non-AMD backends.");
#endif
  hipCtx_t ctx;
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipDeviceptr_t devPtr = 0;
#if HT_AMD
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtr), sizeof(uint64_t),
                                  hipMallocSignalMemory));
#endif
  HIP_CHECK(hipMemset(reinterpret_cast<void*>(devPtr), 0, sizeof(uint64_t)));

  hipStreamBatchMemOpParams params[1] = {};
  params[0].operation = hipStreamMemOpWriteValue32;
  params[0].writeValue.address = devPtr;
  params[0].writeValue.value = 1000;
  params[0].writeValue.flags = hipStreamWriteValueDefault;
  hipBatchMemOpNodeParams nodeParams = {};
  nodeParams.ctx = ctx;
  nodeParams.count = 1;
  nodeParams.paramArray = params;
  nodeParams.flags = 0;

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &nodeParams));
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  auto launch_and_read = [&]() {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    uint32_t result = 0;
    HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    return result;
  };
  REQUIRE(launch_and_read() == 1000);

  params[0].writeValue.value = 2000;
  HIP_CHECK(hipGraphExecBatchMemOpNodeSetParams(graphExec, node, &nodeParams));
  REQUIRE(launch_and_read() == 2000);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
  HIP_CHECK(hipCtxDestroy(ctx));
}

#if HT_AMD
namespace {
constexpr unsigned kMemOpUpdateElements = 128;

__global__ void MemOpUpdateNoop(unsigned) {}

template <typename T> __global__ void MemOpUpdateProducer(unsigned* input, T* gate, unsigned seed,
                                                          unsigned long long delayTicks) {
  const auto start = wall_clock64();
  while (wall_clock64() - start < delayTicks) {
  }
  for (unsigned i = 0; i < kMemOpUpdateElements; ++i) input[i] = seed + 13 * i;
  __threadfence_system();
  __hip_atomic_store(gate, T{1}, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void MemOpUpdateConsumer(const unsigned* input, unsigned* output) {
  const unsigned i = threadIdx.x;
  if (i < kMemOpUpdateElements) output[i] = 3 * input[i] + 7;
}

struct MemOpUpdateSignal {
  uint64_t* ptr = nullptr;
  MemOpUpdateSignal() {
    HIP_CHECK(
        hipExtMallocWithFlags(reinterpret_cast<void**>(&ptr), sizeof(*ptr), hipMallocSignalMemory));
    __atomic_store_n(ptr, uint64_t{0}, __ATOMIC_RELEASE);
  }
  ~MemOpUpdateSignal() { static_cast<void>(hipFree(ptr)); }
};

struct MemOpUpdateGraph {
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  ~MemOpUpdateGraph() {
    if (exec) static_cast<void>(hipGraphExecDestroy(exec));
    if (graph) static_cast<void>(hipGraphDestroy(graph));
  }
};

template <typename T> void CheckMemOpUpdateCompletion(bool initiallyHasWrites, bool updateGraph) {
  MemOpUpdateSignal gateMemory, tokenMemory;
  auto* gate = reinterpret_cast<T*>(gateMemory.ptr);
  auto* token = reinterpret_cast<T*>(tokenMemory.ptr);
  LinearAllocGuard<unsigned> input(LinearAllocs::hipMalloc,
                                   kMemOpUpdateElements * sizeof(unsigned));
  LinearAllocGuard<unsigned> output(LinearAllocs::hipMalloc,
                                    2 * kMemOpUpdateElements * sizeof(unsigned));
  StreamGuard producer(Streams::withFlags, hipStreamNonBlocking);
  StreamGuard launch(Streams::withFlags, hipStreamNonBlocking);
  MemOpUpdateGraph graph;
  HIP_CHECK(hipGraphCreate(&graph.graph, 0));

  hipKernelNodeParams noop{};
  unsigned epoch = 0;
  void* noopArgs[] = {&epoch};
  noop.func = reinterpret_cast<void*>(MemOpUpdateNoop);
  noop.gridDim = noop.blockDim = dim3(1);
  noop.kernelParams = noopArgs;
  hipGraphNode_t previous;
  HIP_CHECK(hipGraphAddKernelNode(&previous, graph.graph, nullptr, 0, &noop));
  const hipGraphNode_t root = previous;
  // Three diamonds make the segment dependency depth exceed four, so the
  // default scheduler keeps multiple streams instead of collapsing this graph.
  for (unsigned stage = 0; stage < 3; ++stage) {
    hipGraphNode_t branches[2], join;
    HIP_CHECK(hipGraphAddKernelNode(&branches[0], graph.graph, &previous, 1, &noop));
    HIP_CHECK(hipGraphAddKernelNode(&branches[1], graph.graph, &previous, 1, &noop));
    HIP_CHECK(hipGraphAddKernelNode(&join, graph.graph, branches, 2, &noop));
    previous = join;
  }

  hipStreamBatchMemOpParams ops[2]{};
  ops[0].operation = sizeof(T) == 4 ? hipStreamMemOpWaitValue32 : hipStreamMemOpWaitValue64;
  ops[0].waitValue.address = reinterpret_cast<hipDeviceptr_t>(gate);
  ops[0].waitValue.value64 = 1;
  ops[0].waitValue.flags = hipStreamWaitValueGte;
  ops[1].operation = sizeof(T) == 4 ? hipStreamMemOpWriteValue32 : hipStreamMemOpWriteValue64;
  ops[1].writeValue.address = reinterpret_cast<hipDeviceptr_t>(token);
  ops[1].writeValue.value64 = static_cast<T>(0x100000001ULL);
  ops[1].writeValue.flags = hipStreamWriteValueDefault;
  hipBatchMemOpNodeParams params{};
  HIP_CHECK(hipCtxGetCurrent(&params.ctx));
  params.count = initiallyHasWrites ? 2 : 1;
  params.paramArray = ops;
  hipGraphNode_t memop;
  HIP_CHECK(hipGraphAddBatchMemOpNode(&memop, graph.graph, &previous, 1, &params));

  // The memop ends a segment at this fork. At least one consumer runs on a
  // different stream and needs the memop kernel's completion signal.
  hipGraphNode_t consumers[2], join;
  auto* src = input.ptr();
  for (unsigned branch = 0; branch < 2; ++branch) {
    auto* dst = output.ptr() + branch * kMemOpUpdateElements;
    void* args[] = {&src, &dst};
    hipKernelNodeParams consumer{};
    consumer.func = reinterpret_cast<void*>(MemOpUpdateConsumer);
    consumer.gridDim = dim3(1);
    consumer.blockDim = dim3(kMemOpUpdateElements);
    consumer.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&consumers[branch], graph.graph, &memop, 1, &consumer));
  }
  HIP_CHECK(hipGraphAddEmptyNode(&join, graph.graph, consumers, 2));
  HIP_CHECK(hipGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));

  int device = 0, clockKHz = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&clockKHz, hipDeviceAttributeWallClockRate, device));
  auto delayTicks = static_cast<unsigned long long>(clockKHz) * 5;  // 5 ms
  for (unsigned iteration = 0; iteration < 64; ++iteration) {
    const bool hasWrites = initiallyHasWrites != ((iteration & 1) != 0);
    unsigned seed = 101 + iteration * 107;
    const T expectedToken = static_cast<T>(0x100000001ULL) + iteration;
    params.count = hasWrites ? 2 : 1;
    ops[1].writeValue.value64 = expectedToken;
    if (iteration != 0) {
      if (updateGraph) {
        // Leave a same-size packet update pending before the layout change in
        // hipGraphExecUpdate's batch, then verify that subsequent launches work.
        epoch = iteration;
        HIP_CHECK(hipGraphKernelNodeSetParams(root, &noop));
        HIP_CHECK(hipGraphBatchMemOpNodeSetParams(memop, &params));
        hipGraphNode_t errorNode = nullptr;
        hipGraphExecUpdateResult result;
        HIP_CHECK(hipGraphExecUpdate(graph.exec, graph.graph, &errorNode, &result));
        REQUIRE(result == hipGraphExecUpdateSuccess);
      } else {
        HIP_CHECK(hipGraphExecBatchMemOpNodeSetParams(graph.exec, memop, &params));
      }
    }
    __atomic_store_n(gate, T{0}, __ATOMIC_RELEASE);
    __atomic_store_n(token, T{0}, __ATOMIC_RELEASE);
    HIP_CHECK(hipMemsetAsync(input.ptr(), 0, input.size_bytes(), producer.stream()));
    HIP_CHECK(hipMemsetAsync(output.ptr(), 0, output.size_bytes(), launch.stream()));
    void* producerArgs[] = {&src, &gate, &seed, &delayTicks};
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<const void*>(MemOpUpdateProducer<T>), dim3(1),
                              dim3(1), producerArgs, 0, producer.stream()));
    HIP_CHECK(hipGraphLaunch(graph.exec, launch.stream()));
    HIP_CHECK(hipStreamSynchronize(launch.stream()));
    HIP_CHECK(hipStreamSynchronize(producer.stream()));
    // Drain all graph streams before an assertion or another update. This is
    // after the consumers have run, so it cannot hide an early data read.
    HIP_CHECK(hipDeviceSynchronize());
    unsigned actual[2 * kMemOpUpdateElements]{};
    HIP_CHECK(hipMemcpy(actual, output.ptr(), sizeof(actual), hipMemcpyDeviceToHost));
    unsigned mismatches = 0;
    for (unsigned i = 0; i < 2 * kMemOpUpdateElements; ++i) {
      mismatches += actual[i] != 3 * (seed + 13 * (i % kMemOpUpdateElements)) + 7;
    }
    INFO("iteration=" << iteration << ", graph update=" << updateGraph
                      << ", has writes=" << hasWrites << ", branch outputs=" << actual[0] << ", "
                      << actual[kMemOpUpdateElements] << ", expected=" << 3 * seed + 7);
    REQUIRE(mismatches == 0);
    REQUIRE(__atomic_load_n(token, __ATOMIC_ACQUIRE) == (hasWrites ? expectedToken : T{0}));
  }
}
}  // namespace

/**
 * Test Description
 * ------------------------
 * - Update a segment-ending batch between WAIT-only and WAIT+WRITE in both
 *   directions. Dependent segments must wait for the final memop kernel after
 *   its captured packet layout grows or shrinks. Verify real producer data on
 *   both branches and repeat the updates across graph replays, using both the
 *   executable node setter and whole-graph updates with another pending update.
 * Test source
 * ------------------------
 * - unit/graph/hipGraphExecBatchMemOpNodeSetParams.cc
 * Test requirements
 * ------------------------
 * - AMD segmented graph executor and signal memory support
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipGraphExecBatchMemOpNodeSetParams_LayoutChange, uint32_t, uint64_t) {
  if (!HipTest::IsRocrMemOpBackendSelected()) {
    HIP_SKIP_TEST("Requires ROCr batch memops; select GPU_ENABLE_PAL=0 for automatic selection.");
  }
  const char* classic = std::getenv("DEBUG_HIP_GRAPH_CLASSIC_PATH");
  if (classic && (std::atoi(classic) != 0 || std::strcmp(classic, "true") == 0)) {
    HIP_SKIP_TEST("Requires the segmented graph executor");
  }
  int device = 0, supported = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeCanUseStreamWaitValue, device));
  if (!supported) HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  const bool updateGraph = GENERATE(false, true);
  SECTION("WaitToWrite") { CheckMemOpUpdateCompletion<TestType>(false, updateGraph); }
  SECTION("WriteToWait") { CheckMemOpUpdateCompletion<TestType>(true, updateGraph); }
}
#endif

/**
 * End doxygen group GraphTest.
 * @}
 */
