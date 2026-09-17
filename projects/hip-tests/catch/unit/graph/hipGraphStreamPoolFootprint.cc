/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Graph exec instantiate-time device-memory footprint test.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>

#include <cstdlib>
#include <vector>

namespace {

constexpr int kElems = 1024;

struct LargeKernarg {
  int value;
  int padding[255];
};

__global__ void bumpKernel(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += 1;
}

__global__ void writeValueKernel(int* out, LargeKernarg payload) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *out = payload.value;
  }
}

__global__ void waitAndRecordValueKernel(int* out, unsigned int* index,
                                         volatile unsigned int* gate,
                                         LargeKernarg payload) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (*gate == 0) {
    }
    out[atomicAdd(index, 1)] = payload.value;
  }
}

LargeKernarg makePayload(int value) {
  LargeKernarg payload{};
  payload.value = value;
  return payload;
}

class GateGuard {
 public:
  explicit GateGuard(unsigned int* gate) : gate_(gate) {}
  ~GateGuard() { Open(); }
  void Open() {
    if (gate_ != nullptr) {
      __atomic_store_n(gate_, 1u, __ATOMIC_RELEASE);
      gate_ = nullptr;
    }
  }

 private:
  unsigned int* gate_;
};

// Build chains of kernel nodes.
void buildChainedGraph(hipGraph_t* graph, int* buf, int chains, int depth) {
  HIP_CHECK(hipGraphCreate(graph, 0));
  for (int c = 0; c < chains; ++c) {
    hipGraphNode_t prev = nullptr;
    for (int d = 0; d < depth; ++d) {
      hipGraphNode_t node = nullptr;
      int elems = kElems;
      void* args[] = {&buf, &elems};
      hipKernelNodeParams params = {};
      params.func = reinterpret_cast<void*>(bumpKernel);
      params.gridDim = dim3(1);
      params.blockDim = dim3(64);
      params.sharedMemBytes = 0;
      params.kernelParams = args;
      params.extra = nullptr;
      HIP_CHECK(hipGraphAddKernelNode(&node, *graph, prev ? &prev : nullptr, prev ? 1 : 0,
                                      &params));
      prev = node;
    }
  }
}

size_t freeDeviceMemory() {
  size_t free_bytes = 0, total_bytes = 0;
  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
  return free_bytes;
}

void requireDeviceKernargPool() {
  int device = 0;
  hipDeviceProp_t properties = {};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  if (!properties.isLargeBar) {
    HIP_SKIP_TEST("hipMemGetInfo cannot measure host-backed graph kernarg pools.");
  }
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Same-device instantiate footprint must stay below one kernarg pool. Single
 *    chain needs only the launch stream; no eager cross-device slot-0 stream.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphStreamPoolFootprint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphStreamPool_InstantiateFootprint) {
  constexpr int kGraphs = 64;
  constexpr int kChains = 1;
  constexpr int kDepth = 2;
  constexpr size_t kMaxBytesPerGraph = 2 * 1024 * 1024;

  const char* min_overlap = std::getenv("DEBUG_HIP_GRAPH_MIN_OVERLAP");
  INFO("DEBUG_HIP_GRAPH_MIN_OVERLAP=" << (min_overlap ? min_overlap : "(default)"));

  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, kElems * sizeof(int)));

  // Warm up one-off allocations away from the measurement.
  for (int i = 0; i < 4; ++i) {
    hipGraph_t warm_graph = nullptr;
    hipGraphExec_t warm_exec = nullptr;
    buildChainedGraph(&warm_graph, buf, kChains, kDepth);
    HIP_CHECK(hipGraphInstantiate(&warm_exec, warm_graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphExecDestroy(warm_exec));
    HIP_CHECK(hipGraphDestroy(warm_graph));
  }

  const size_t free_before = freeDeviceMemory();

  std::vector<hipGraph_t> graphs(kGraphs, nullptr);
  std::vector<hipGraphExec_t> execs(kGraphs, nullptr);
  for (int i = 0; i < kGraphs; ++i) {
    buildChainedGraph(&graphs[i], buf, kChains, kDepth);
    HIP_CHECK(hipGraphInstantiate(&execs[i], graphs[i], nullptr, nullptr, 0));
  }

  const size_t free_after = freeDeviceMemory();
  REQUIRE(free_after <= free_before);
  const size_t used = free_before - free_after;
  const size_t per_graph = used / kGraphs;
  INFO("device memory per instantiate: " << per_graph << " bytes over " << kGraphs << " graphs");
  REQUIRE(used <= kGraphs * kMaxBytesPerGraph);

  for (int i = 0; i < kGraphs; ++i) {
    HIP_CHECK(hipGraphExecDestroy(execs[i]));
    HIP_CHECK(hipGraphDestroy(graphs[i]));
  }

  // Memory must return after destroy.
  const size_t free_end = freeDeviceMemory();
  const size_t retained = (free_before > free_end) ? (free_before - free_end) : 0;
  INFO("device memory retained after destroy: " << retained << " bytes");
  REQUIRE(retained <= kMaxBytesPerGraph);

  HIP_CHECK(hipFree(buf));
}

/**
 * Test Description
 * ------------------------
 *  - Repeated idle whole-graph updates reuse the installed kernarg slot and
 *    remain memory-bounded.
 *  - Repeated queued launch/update cycles remain bounded by the number of
 *    outstanding launch generations rather than the total number of updates.
 *  - Launches observe the arguments belonging to their own generation.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphStreamPoolFootprint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_KernargFootprint) {
  requireDeviceKernargPool();
  constexpr size_t kMaxUpdateGrowth = 4 * 1024 * 1024;

  SECTION("idle in-place reuse") {
    constexpr int kUpdates = 8192;
    int* out = nullptr;
    HIP_CHECK(hipMalloc(&out, sizeof(int)));

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    LargeKernarg payload = makePayload(0);
    void* args[] = {&out, &payload};
    hipKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(writeValueKernel);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;

    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    auto updateExec = [&](int value) {
      LargeKernarg updated_payload = makePayload(value);
      void* updated_args[] = {&out, &updated_payload};
      hipKernelNodeParams updated_params = params;
      updated_params.kernelParams = updated_args;
      HIP_CHECK(hipGraphKernelNodeSetParams(node, &updated_params));
      hipGraphNode_t error_node = nullptr;
      hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
      HIP_CHECK(hipGraphExecUpdate(exec, graph, &error_node, &update_result));
      REQUIRE(update_result == hipGraphExecUpdateSuccess);
      REQUIRE(error_node == nullptr);
    };

    for (int i = 0; i < 8; ++i) {
      updateExec(i);
    }
    const size_t free_before = freeDeviceMemory();
    for (int i = 1; i <= kUpdates; ++i) {
      updateExec(i);
    }
    const size_t free_after = freeDeviceMemory();
    const size_t update_growth = free_before > free_after ? free_before - free_after : 0;
    INFO("device memory growth over " << kUpdates << " idle graph updates: " << update_growth
                                      << " bytes");
    REQUIRE(update_growth <= kMaxUpdateGrowth);

    HIP_CHECK(hipGraphLaunch(exec, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, out, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(result == kUpdates);

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipFree(out));
  }

  SECTION("pipelined launch generations") {
    constexpr int kBatchSize = 16;
    constexpr int kWarmupRounds = 2;
    constexpr int kMeasuredRounds = 512;

    int* out = nullptr;
    unsigned int* index = nullptr;
    unsigned int* host_gate = nullptr;
    unsigned int* device_gate = nullptr;
    HIP_CHECK(hipMalloc(&out, kBatchSize * sizeof(int)));
    HIP_CHECK(hipMalloc(&index, sizeof(unsigned int)));
    HIP_CHECK(hipHostMalloc(&host_gate, sizeof(unsigned int),
                            hipHostMallocMapped | hipHostMallocCoherent));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_gate), host_gate, 0));

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    LargeKernarg payload = makePayload(0);
    void* args[] = {&out, &index, &device_gate, &payload};
    hipKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(waitAndRecordValueKernel);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;

    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    int next_value = 0;
    auto runRound = [&]() {
      *host_gate = 0;
      GateGuard gate_guard(host_gate);
      HIP_CHECK(hipMemset(index, 0, sizeof(unsigned int)));
      const int first_value = next_value;
      for (int i = 0; i < kBatchSize; ++i) {
        HIP_CHECK(hipGraphLaunch(exec, stream));
        ++next_value;
        LargeKernarg updated_payload = makePayload(next_value);
        void* updated_args[] = {&out, &index, &device_gate, &updated_payload};
        hipKernelNodeParams updated_params = params;
        updated_params.kernelParams = updated_args;
        HIP_CHECK(hipGraphKernelNodeSetParams(node, &updated_params));
        hipGraphNode_t error_node = nullptr;
        hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
        HIP_CHECK(hipGraphExecUpdate(exec, graph, &error_node, &update_result));
        REQUIRE(update_result == hipGraphExecUpdateSuccess);
      }
      gate_guard.Open();
      HIP_CHECK(hipStreamSynchronize(stream));

      int result[kBatchSize] = {};
      HIP_CHECK(hipMemcpy(result, out, sizeof(result), hipMemcpyDeviceToHost));
      for (int i = 0; i < kBatchSize; ++i) {
        REQUIRE(result[i] == first_value + i);
      }
    };

    for (int i = 0; i < kWarmupRounds; ++i) {
      runRound();
    }
    const size_t free_before = freeDeviceMemory();
    for (int i = 0; i < kMeasuredRounds; ++i) {
      runRound();
    }
    const size_t free_after = freeDeviceMemory();
    const size_t update_growth = free_before > free_after ? free_before - free_after : 0;
    INFO("device memory growth over " << kMeasuredRounds * kBatchSize
                                      << " pipelined graph updates: " << update_growth << " bytes");
    REQUIRE(update_growth <= kMaxUpdateGrowth);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipHostFree(host_gate));
    HIP_CHECK(hipFree(index));
    HIP_CHECK(hipFree(out));
  }
}
