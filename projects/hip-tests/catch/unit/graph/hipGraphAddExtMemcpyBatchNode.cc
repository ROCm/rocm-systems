/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#include <numeric>
#include <vector>

/**
 * @addtogroup hipGraphAddExtMemcpyBatchNode hipGraphAddExtMemcpyBatchNode
 * @{
 * @ingroup GraphTest
 * `hipError_t hipGraphAddExtMemcpyBatchNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
 * const hipGraphNode_t* pDependencies, size_t numDependencies,
 * const hipExtMemcpyBatchNodeParams* nodeParams);`
 * - Adds a batch of 1D memory copies to a graph as a single node.
 *
 * Also covers the accompanying parameter APIs:
 * - hipGraphExtMemcpyBatchNodeGetParams
 * - hipGraphExtMemcpyBatchNodeSetParams
 * - hipGraphExecExtMemcpyBatchNodeSetParams
 * - recording of hipMemcpyBatchAsync during stream capture
 */

static __global__ void EmptyKernel() { return; }

namespace {

constexpr size_t kNumCopies = 4;
constexpr size_t kElems = 1024;
constexpr size_t kBytes = kElems * sizeof(int);

// A batch of device destinations fed from distinct host sources, held together so a test can
// build node parameters and check the results without repeating the bookkeeping.
class BatchBuffers {
 public:
  explicit BatchBuffers(int seed) : seed_(seed), host_(kNumCopies) {
    for (size_t copy_idx = 0; copy_idx < kNumCopies; ++copy_idx) {
      host_[copy_idx].resize(kElems);
      std::iota(host_[copy_idx].begin(), host_[copy_idx].end(), seed + copy_idx * kElems);

      void* device = nullptr;
      HIP_CHECK(hipMalloc(&device, kBytes));
      HIP_CHECK(hipMemset(device, 0, kBytes));
      dsts_.push_back(device);
      srcs_.push_back(host_[copy_idx].data());
      sizes_.push_back(kBytes);
    }
  }

  ~BatchBuffers() {
    for (void* device : dsts_) {
      static_cast<void>(hipFree(device));
    }
  }

  BatchBuffers(const BatchBuffers&) = delete;
  BatchBuffers& operator=(const BatchBuffers&) = delete;

  hipExtMemcpyBatchNodeParams Params() {
    hipExtMemcpyBatchNodeParams params{};
    params.dsts = dsts_.data();
    params.srcs = srcs_.data();
    params.sizes = sizes_.data();
    params.count = kNumCopies;
    return params;
  }

  void ClearDevice() {
    for (void* device : dsts_) {
      HIP_CHECK(hipMemset(device, 0, kBytes));
    }
  }

  void VerifyCopied() const {
    std::vector<int> readback(kElems);
    for (size_t copy_idx = 0; copy_idx < kNumCopies; ++copy_idx) {
      HIP_CHECK(hipMemcpy(readback.data(), dsts_[copy_idx], kBytes, hipMemcpyDeviceToHost));
      INFO("copy index " << copy_idx << " of batch seeded with " << seed_);
      REQUIRE(readback == host_[copy_idx]);
    }
  }

  void VerifyUntouched() const {
    std::vector<int> readback(kElems);
    for (size_t copy_idx = 0; copy_idx < kNumCopies; ++copy_idx) {
      HIP_CHECK(hipMemcpy(readback.data(), dsts_[copy_idx], kBytes, hipMemcpyDeviceToHost));
      INFO("copy index " << copy_idx << " of batch seeded with " << seed_);
      REQUIRE(readback == std::vector<int>(kElems, 0));
    }
  }

  void** dsts() { return dsts_.data(); }
  void** srcs() { return srcs_.data(); }
  size_t* sizes() { return sizes_.data(); }

 private:
  int seed_;
  std::vector<std::vector<int>> host_;
  std::vector<void*> dsts_;
  std::vector<void*> srcs_;
  std::vector<size_t> sizes_;
};

size_t GetNodeCount(hipGraph_t graph) {
  size_t num_nodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &num_nodes));
  return num_nodes;
}

hipGraphNode_t GetOnlyNode(hipGraph_t graph) {
  size_t num_nodes = 1;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphGetNodes(graph, &node, &num_nodes));
  return node;
}

hipGraphNode_t FindBatchNode(hipGraph_t graph) {
  size_t num_nodes = GetNodeCount(graph);
  std::vector<hipGraphNode_t> nodes(num_nodes);
  HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &num_nodes));
  for (hipGraphNode_t candidate : nodes) {
    hipGraphNodeType node_type;
    HIP_CHECK(hipGraphNodeGetType(candidate, &node_type));
    if (node_type == hipGraphNodeTypeExtMemcpyBatch) return candidate;
  }
  return nullptr;
}

hipKernelNodeParams EmptyKernelNodeParams() {
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(EmptyKernel);
  params.gridDim = dim3(1, 1, 1);
  params.blockDim = dim3(1, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = nullptr;
  params.extra = nullptr;
  return params;
}

// The two supported ways of getting a batch into a graph. Both must produce an equivalent node,
// so tests that care about node behaviour rather than construction run against each.
enum class BuildMethod { kExplicitNode, kStreamCapture };

const char* ToString(BuildMethod method) {
  return method == BuildMethod::kExplicitNode ? "hipGraphAddExtMemcpyBatchNode"
                                              : "hipStreamBeginCapture/hipStreamEndCapture";
}

// Builds a single-node graph holding the given batch, either by adding the node directly or by
// capturing an equivalent hipMemcpyBatchAsync. Returns the graph and its lone node.
void BuildBatchGraph(BuildMethod method, const hipExtMemcpyBatchNodeParams& params,
                     hipGraph_t* graph, hipGraphNode_t* node) {
  if (method == BuildMethod::kExplicitNode) {
    HIP_CHECK(hipGraphCreate(graph, 0));
    HIP_CHECK(hipGraphAddExtMemcpyBatchNode(node, *graph, nullptr, 0, &params));
  } else {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    size_t fail_idx = 0;
    HIP_CHECK(hipMemcpyBatchAsync(params.dsts, params.srcs, params.sizes, params.count,
                                  params.attrs, params.attrsIdxs, params.numAttrs, &fail_idx,
                                  stream));
    HIP_CHECK(hipStreamEndCapture(stream, graph));
    HIP_CHECK(hipStreamDestroy(stream));
    *node = GetOnlyNode(*graph);
  }

  REQUIRE(GetNodeCount(*graph) == 1);
  hipGraphNodeType node_type;
  HIP_CHECK(hipGraphNodeGetType(*node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeExtMemcpyBatch);
}

}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 * - Adds a batch of copies as a single node, verifies the node type, and checks that launching
 *   the instantiated graph performs every copy in the batch.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddExtMemcpyBatchNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphAddExtMemcpyBatchNode_Positive_Basic) {
  // The whole batch must collapse into one node rather than one node per copy, whichever way the
  // graph is built.
  const auto method = GENERATE(BuildMethod::kExplicitNode, BuildMethod::kStreamCapture);
  INFO("graph built with " << ToString(method));

  BatchBuffers buffers(/*seed=*/1);
  hipExtMemcpyBatchNodeParams params = buffers.Params();

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  BuildBatchGraph(method, params, &graph, &node);

  // Capture must defer the work rather than perform it.
  buffers.VerifyUntouched();

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  buffers.VerifyCopied();

  // The node owns a copy of the parameter arrays, so a relaunch must repeat the same work.
  buffers.ClearDevice();
  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  buffers.VerifyCopied();

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 * - Verifies the negative cases of hipGraphAddExtMemcpyBatchNode.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddExtMemcpyBatchNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphAddExtMemcpyBatchNode_Negative_Parameters) {
  BatchBuffers buffers(/*seed=*/2);
  hipExtMemcpyBatchNodeParams params = buffers.Params();

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;

  SECTION("Node as null pointer") {
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(nullptr, graph, nullptr, 0, &params),
                    hipErrorInvalidValue);
  }

  SECTION("Graph as null pointer") {
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, nullptr, nullptr, 0, &params),
                    hipErrorInvalidValue);
  }

  SECTION("Node params as null pointer") {
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Dependencies as null pointer with non-zero count") {
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 1, &params),
                    hipErrorInvalidValue);
  }

  SECTION("Copy count is zero") {
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.count = 0;
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("Destination array as null pointer") {
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.dsts = nullptr;
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("Source array as null pointer") {
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.srcs = nullptr;
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("Size array as null pointer") {
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.sizes = nullptr;
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("A single destination entry is null") {
    std::vector<void*> dsts(buffers.dsts(), buffers.dsts() + kNumCopies);
    dsts[kNumCopies - 1] = nullptr;
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.dsts = dsts.data();
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("A single copy size is zero") {
    std::vector<size_t> sizes(buffers.sizes(), buffers.sizes() + kNumCopies);
    sizes[1] = 0;
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.sizes = sizes.data();
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("Attributes array as null pointer with non-zero count") {
    size_t attrs_idx = 0;
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.attrs = nullptr;
    invalid.attrsIdxs = &attrs_idx;
    invalid.numAttrs = 1;
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("First attribute index does not start at zero") {
    hipMemcpyAttributes attrs{};
    attrs.srcAccessOrder = hipMemcpySrcAccessOrderStream;
    size_t attrs_idx = 1;
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.attrs = &attrs;
    invalid.attrsIdxs = &attrs_idx;
    invalid.numAttrs = 1;
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  SECTION("More attributes than copies") {
    std::vector<hipMemcpyAttributes> attrs(kNumCopies + 1);
    std::vector<size_t> attrs_idxs(kNumCopies + 1);
    std::iota(attrs_idxs.begin(), attrs_idxs.end(), 0);
    for (auto& attr : attrs) attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
    hipExtMemcpyBatchNodeParams invalid = params;
    invalid.attrs = attrs.data();
    invalid.attrsIdxs = attrs_idxs.data();
    invalid.numAttrs = attrs.size();
    HIP_CHECK_ERROR(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &invalid),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 * - Verifies that hipGraphExtMemcpyBatchNodeGetParams returns the batch the node was created
 *   with, and that hipGraphExtMemcpyBatchNodeSetParams retargets the node before instantiation.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddExtMemcpyBatchNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphExtMemcpyBatchNode_Positive_GetSetParams) {
  BatchBuffers original(/*seed=*/3);
  BatchBuffers replacement(/*seed=*/4);
  hipExtMemcpyBatchNodeParams params = original.Params();

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &params));

  hipExtMemcpyBatchNodeParams readback{};
  HIP_CHECK(hipGraphExtMemcpyBatchNodeGetParams(node, &readback));
  REQUIRE(readback.count == kNumCopies);
  for (size_t copy_idx = 0; copy_idx < kNumCopies; ++copy_idx) {
    INFO("copy index " << copy_idx);
    REQUIRE(readback.dsts[copy_idx] == original.dsts()[copy_idx]);
    REQUIRE(readback.srcs[copy_idx] == original.srcs()[copy_idx]);
    REQUIRE(readback.sizes[copy_idx] == original.sizes()[copy_idx]);
  }

  hipExtMemcpyBatchNodeParams updated = replacement.Params();
  HIP_CHECK(hipGraphExtMemcpyBatchNodeSetParams(node, &updated));
  HIP_CHECK(hipGraphExtMemcpyBatchNodeGetParams(node, &readback));
  REQUIRE(readback.dsts[0] == replacement.dsts()[0]);

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  replacement.VerifyCopied();
  original.VerifyUntouched();

  SECTION("Get params on a null node") {
    HIP_CHECK_ERROR(hipGraphExtMemcpyBatchNodeGetParams(nullptr, &readback), hipErrorInvalidValue);
  }

  SECTION("Get params into a null pointer") {
    HIP_CHECK_ERROR(hipGraphExtMemcpyBatchNodeGetParams(node, nullptr), hipErrorInvalidValue);
  }

  SECTION("Set params from a null pointer") {
    HIP_CHECK_ERROR(hipGraphExtMemcpyBatchNodeSetParams(node, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 * - Verifies that hipGraphExecExtMemcpyBatchNodeSetParams retargets an instantiated graph, and
 *   that changing the number of copies is rejected because the batch size is fixed at
 *   instantiation.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddExtMemcpyBatchNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphExecExtMemcpyBatchNodeSetParams_Positive_Basic) {
  BatchBuffers original(/*seed=*/5);
  BatchBuffers replacement(/*seed=*/6);
  hipExtMemcpyBatchNodeParams params = original.Params();

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &params));
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  hipExtMemcpyBatchNodeParams updated = replacement.Params();
  HIP_CHECK(hipGraphExecExtMemcpyBatchNodeSetParams(graph_exec, node, &updated));
  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  replacement.VerifyCopied();
  original.VerifyUntouched();

  SECTION("Changing the copy count is rejected") {
    hipExtMemcpyBatchNodeParams resized = replacement.Params();
    resized.count = kNumCopies - 1;
    HIP_CHECK_ERROR(hipGraphExecExtMemcpyBatchNodeSetParams(graph_exec, node, &resized),
                    hipErrorInvalidValue);
  }

  SECTION("Null exec graph is rejected") {
    HIP_CHECK_ERROR(hipGraphExecExtMemcpyBatchNodeSetParams(nullptr, node, &updated),
                    hipErrorInvalidValue);
  }

  SECTION("Null node is rejected") {
    HIP_CHECK_ERROR(hipGraphExecExtMemcpyBatchNodeSetParams(graph_exec, nullptr, &updated),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 * - Verifies that hipGraphExecUpdate propagates a retargeted batch node into an already
 *   instantiated graph, and reports a parameter change when the batch size differs.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddExtMemcpyBatchNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphExtMemcpyBatchNode_Positive_ExecUpdate) {
  BatchBuffers original(/*seed=*/7);
  BatchBuffers replacement(/*seed=*/8);
  hipExtMemcpyBatchNodeParams params = original.Params();

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddExtMemcpyBatchNode(&node, graph, nullptr, 0, &params));
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  // Retargeting the graph node alone must not disturb the already instantiated exec.
  hipExtMemcpyBatchNodeParams updated = replacement.Params();
  HIP_CHECK(hipGraphExtMemcpyBatchNodeSetParams(node, &updated));
  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  original.VerifyCopied();
  replacement.VerifyUntouched();

  original.ClearDevice();
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateSuccess;
  HIP_CHECK(hipGraphExecUpdate(graph_exec, graph, &error_node, &update_result));
  REQUIRE(update_result == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  replacement.VerifyCopied();
  original.VerifyUntouched();

  SECTION("A resized batch is reported as a parameter change") {
    hipExtMemcpyBatchNodeParams resized = replacement.Params();
    resized.count = kNumCopies - 1;
    HIP_CHECK(hipGraphExtMemcpyBatchNodeSetParams(node, &resized));
    HIP_CHECK_ERROR(hipGraphExecUpdate(graph_exec, graph, &error_node, &update_result),
                    hipErrorGraphExecUpdateFailure);
    REQUIRE(update_result == hipGraphExecUpdateErrorParametersChanged);
    REQUIRE(error_node == node);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 * - Places an empty kernel before and after the batch so the node has to link into a dependency
 *   chain rather than stand alone, and checks the chain is identical whether it is built with
 *   hipGraphAddExtMemcpyBatchNode or recorded with hipStreamBeginCapture/hipStreamEndCapture.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddExtMemcpyBatchNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphExtMemcpyBatchNode_Positive_KernelDependencies) {
  const auto method = GENERATE(BuildMethod::kExplicitNode, BuildMethod::kStreamCapture);
  INFO("graph built with " << ToString(method));

  BatchBuffers buffers(/*seed=*/9);
  BatchBuffers replacement(/*seed=*/10);
  hipExtMemcpyBatchNodeParams params = buffers.Params();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  hipGraph_t graph = nullptr;

  if (method == BuildMethod::kStreamCapture) {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    EmptyKernel<<<1, 1, 0, stream>>>();
    size_t fail_idx = 0;
    HIP_CHECK(hipMemcpyBatchAsync(buffers.dsts(), buffers.srcs(), buffers.sizes(), kNumCopies,
                                  nullptr, nullptr, 0, &fail_idx, stream));
    EmptyKernel<<<1, 1, 0, stream>>>();
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
  } else {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipKernelNodeParams kernel_params = EmptyKernelNodeParams();
    hipGraphNode_t before = nullptr;
    hipGraphNode_t batch = nullptr;
    hipGraphNode_t after = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&before, graph, nullptr, 0, &kernel_params));
    HIP_CHECK(hipGraphAddExtMemcpyBatchNode(&batch, graph, &before, 1, &params));
    HIP_CHECK(hipGraphAddKernelNode(&after, graph, &batch, 1, &kernel_params));
  }

  // Empty kernel, batch, empty kernel: the batch is still a single node no matter how many
  // copies it carries.
  REQUIRE(GetNodeCount(graph) == 3);
  hipGraphNode_t batch_node = FindBatchNode(graph);
  REQUIRE(batch_node != nullptr);

  size_t num_dependencies = 0;
  HIP_CHECK(hipGraphNodeGetDependencies(batch_node, nullptr, &num_dependencies));
  REQUIRE(num_dependencies == 1);
  size_t num_dependents = 0;
  HIP_CHECK(hipGraphNodeGetDependentNodes(batch_node, nullptr, &num_dependents));
  REQUIRE(num_dependents == 1);

  // Building the graph must defer the work rather than perform it.
  buffers.VerifyUntouched();

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  buffers.VerifyCopied();

  // A node sitting between two kernels must still be updatable.
  hipExtMemcpyBatchNodeParams updated = replacement.Params();
  HIP_CHECK(hipGraphExecExtMemcpyBatchNodeSetParams(graph_exec, batch_node, &updated));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  replacement.VerifyCopied();

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Broadcasts one source buffer to a destination on every reachable peer GPU plus one local
 *   device-to-device destination, all within a single batch node, and confirms the source can
 *   then be swapped through hipGraphExecUpdate.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddExtMemcpyBatchNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 *    - Multi-device
 */
HIP_TEST_CASE(Unit_hipGraphExtMemcpyBatchNode_Positive_BroadcastPeerToPeer) {
  const auto method = GENERATE(BuildMethod::kExplicitNode, BuildMethod::kStreamCapture);
  INFO("graph built with " << ToString(method));

  int num_gpus = 0;
  HIP_CHECK(hipGetDeviceCount(&num_gpus));
  if (num_gpus < 2) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);

  HIP_CHECK(hipSetDevice(0));
  std::vector<int> pattern(kElems);
  std::iota(pattern.begin(), pattern.end(), 0x1000);
  std::vector<int> second_pattern(kElems);
  std::iota(second_pattern.begin(), second_pattern.end(), 0x2000);

  int* src = nullptr;
  int* second_src = nullptr;
  int* local_dst = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&src), kBytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&second_src), kBytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&local_dst), kBytes));
  HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(second_src, second_pattern.data(), kBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(local_dst, 0, kBytes));

  // One destination per peer that device 0 can reach directly.
  std::vector<int> peer_devices;
  std::vector<int*> peer_dsts;
  for (int peer = 1; peer < num_gpus; ++peer) {
    int can_access = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access, 0, peer));
    if (!can_access) continue;

    HIP_CHECK(hipSetDevice(peer));
    int* peer_dst = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&peer_dst), kBytes));
    HIP_CHECK(hipMemset(peer_dst, 0, kBytes));
    peer_devices.push_back(peer);
    peer_dsts.push_back(peer_dst);
  }
  HIP_CHECK(hipSetDevice(0));
  if (peer_devices.empty()) {
    HIP_CHECK(hipFree(src));
    HIP_CHECK(hipFree(second_src));
    HIP_CHECK(hipFree(local_dst));
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
    return;
  }
  for (int peer : peer_devices) {
    hipError_t status = hipDeviceEnablePeerAccess(peer, 0);
    if (status != hipErrorPeerAccessAlreadyEnabled) {
      HIP_CHECK(status);
    }
  }

  // Same source for every entry: one local device-to-device copy plus one peer copy each.
  std::vector<void*> dsts{local_dst};
  std::vector<size_t> sizes(1 + peer_dsts.size(), kBytes);
  for (int* peer_dst : peer_dsts) {
    dsts.push_back(peer_dst);
  }
  std::vector<void*> srcs(dsts.size(), src);
  std::vector<void*> second_srcs(dsts.size(), second_src);

  hipExtMemcpyBatchNodeParams params{};
  params.dsts = dsts.data();
  params.srcs = srcs.data();
  params.sizes = sizes.data();
  params.count = dsts.size();

  // A broadcast of any width stays a single node, however the graph was built.
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  BuildBatchGraph(method, params, &graph, &node);

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  auto verify_destination = [](int device, int* ptr, const std::vector<int>& expected) {
    HIP_CHECK(hipSetDevice(device));
    std::vector<int> readback(kElems);
    HIP_CHECK(hipMemcpy(readback.data(), ptr, kBytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));
    INFO("destination on device " << device);
    REQUIRE(readback == expected);
  };

  verify_destination(0, local_dst, pattern);
  for (size_t peer_idx = 0; peer_idx < peer_dsts.size(); ++peer_idx) {
    verify_destination(peer_devices[peer_idx], peer_dsts[peer_idx], pattern);
  }

  // Retarget the broadcast at a different source and push it through hipGraphExecUpdate.
  hipExtMemcpyBatchNodeParams updated = params;
  updated.srcs = second_srcs.data();
  HIP_CHECK(hipGraphExtMemcpyBatchNodeSetParams(node, &updated));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateSuccess;
  HIP_CHECK(hipGraphExecUpdate(graph_exec, graph, &error_node, &update_result));
  REQUIRE(update_result == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(graph_exec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  verify_destination(0, local_dst, second_pattern);
  for (size_t peer_idx = 0; peer_idx < peer_dsts.size(); ++peer_idx) {
    verify_destination(peer_devices[peer_idx], peer_dsts[peer_idx], second_pattern);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(second_src));
  HIP_CHECK(hipFree(local_dst));
  for (size_t peer_idx = 0; peer_idx < peer_dsts.size(); ++peer_idx) {
    HIP_CHECK(hipSetDevice(peer_devices[peer_idx]));
    HIP_CHECK(hipFree(peer_dsts[peer_idx]));
  }
  HIP_CHECK(hipSetDevice(0));
}

/**
* End doxygen group GraphTest.
* @}
*/
