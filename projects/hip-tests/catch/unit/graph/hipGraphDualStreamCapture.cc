/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphDualStreamCapture hipGraphDualStreamCapture
 * @{
 * @ingroup GraphTest
 * A deep graph captured from two streams, shaped like a dual-stream MoE layer: each layer forks
 * from the main stream to a side stream, runs a three-kernel branch and a one-kernel branch in
 * parallel, and joins them with an add whose output feeds the next layer. The graph is long
 * enough for the executor to spread it over several hardware queues, and replayed many times so
 * per-queue signal pools wrap around. Kernels spin so the branches really overlap. Every
 * intermediate buffer is poisoned before each replay and the result is compared exactly with a
 * host reference, so a dropped fork or join wait reads poisoned or stale data and fails.
 */

#include <hip_test_common.hh>

#include <cstdint>
#include <vector>

namespace {

constexpr int kLayers = 32;
constexpr size_t kElems = 64 * 1024;
constexpr int kReplays = 20;
constexpr uint32_t kPoison = 0xDEADBEEFu;

// Spin lengths in wall_clock64 ticks. The three-kernel branch is the long one.
constexpr unsigned long long kHeavySpin = 20000;
constexpr unsigned long long kLightSpin = 2000;

__global__ void MulAddSpin(uint32_t* out, const uint32_t* in, uint32_t mul, uint32_t add,
                           unsigned long long spin_ticks, size_t n) {
  const unsigned long long start = wall_clock64();
  while (wall_clock64() - start < spin_ticks) {
  }
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * mul + add;
  }
}

__global__ void Add(uint32_t* out, const uint32_t* a, const uint32_t* b, size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

struct LayerBuffers {
  uint32_t* h1;
  uint32_t* h2;
  uint32_t* routed;
  uint32_t* shared;
  uint32_t* out;  // input of the next layer
};

// Host reference for one element through all layers.
uint32_t Reference(uint32_t x) {
  for (int l = 0; l < kLayers; ++l) {
    const uint32_t h1 = x * 3u + static_cast<uint32_t>(l);
    const uint32_t h2 = h1 * 5u + 7u;
    const uint32_t routed = h2 * 9u + 1u;
    const uint32_t shared = x * 11u + static_cast<uint32_t>(l);
    x = routed + shared;
  }
  return x;
}

void RoutedBranch(const LayerBuffers& b, const uint32_t* in, int layer, hipStream_t s) {
  const dim3 block(256);
  const dim3 grid((kElems + block.x - 1) / block.x);
  MulAddSpin<<<grid, block, 0, s>>>(b.h1, in, 3u, static_cast<uint32_t>(layer), kHeavySpin,
                                    kElems);
  MulAddSpin<<<grid, block, 0, s>>>(b.h2, b.h1, 5u, 7u, kHeavySpin, kElems);
  MulAddSpin<<<grid, block, 0, s>>>(b.routed, b.h2, 9u, 1u, kHeavySpin, kElems);
}

void SharedBranch(const LayerBuffers& b, const uint32_t* in, int layer, hipStream_t s) {
  const dim3 block(256);
  const dim3 grid((kElems + block.x - 1) / block.x);
  MulAddSpin<<<grid, block, 0, s>>>(b.shared, in, 11u, static_cast<uint32_t>(layer), kLightSpin,
                                    kElems);
}

// Captures the layers from |main| and |side|. With |heavy_on_side| the three-kernel branch runs
// on the side stream ("reversed"), otherwise the one-kernel branch does ("original").
hipGraph_t CaptureLayers(const std::vector<LayerBuffers>& layers, const uint32_t* input,
                         bool heavy_on_side, hipStream_t main, hipStream_t side,
                         std::vector<hipEvent_t>& events) {
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamBeginCapture(main, hipStreamCaptureModeGlobal));
  const uint32_t* in = input;
  for (int l = 0; l < kLayers; ++l) {
    const LayerBuffers& b = layers[l];
    hipEvent_t fork = events[2 * l];
    hipEvent_t join = events[2 * l + 1];
    // Fork: the side stream starts after the previous layer's add on the main stream.
    HIP_CHECK(hipEventRecord(fork, main));
    HIP_CHECK(hipStreamWaitEvent(side, fork, 0));
    if (heavy_on_side) {
      RoutedBranch(b, in, l, side);
      SharedBranch(b, in, l, main);
    } else {
      SharedBranch(b, in, l, side);
      RoutedBranch(b, in, l, main);
    }
    // Join: the add waits for the side-stream branch.
    HIP_CHECK(hipEventRecord(join, side));
    HIP_CHECK(hipStreamWaitEvent(main, join, 0));
    const dim3 block(256);
    const dim3 grid((kElems + block.x - 1) / block.x);
    Add<<<grid, block, 0, main>>>(b.out, b.routed, b.shared, kElems);
    in = b.out;
  }
  HIP_CHECK(hipStreamEndCapture(main, &graph));
  return graph;
}

void RunDualStreamGraph(bool heavy_on_side) {
  hipStream_t main = nullptr;
  hipStream_t side = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&main, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));

  const size_t bytes = kElems * sizeof(uint32_t);
  std::vector<uint32_t> host_input(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    host_input[i] = static_cast<uint32_t>(i * 2654435761u);
  }
  uint32_t* input = nullptr;
  HIP_CHECK(hipMalloc(&input, bytes));
  HIP_CHECK(hipMemcpy(input, host_input.data(), bytes, hipMemcpyHostToDevice));

  std::vector<LayerBuffers> layers(kLayers);
  std::vector<uint32_t*> all_buffers;
  for (auto& b : layers) {
    for (uint32_t** p : {&b.h1, &b.h2, &b.routed, &b.shared, &b.out}) {
      HIP_CHECK(hipMalloc(p, bytes));
      all_buffers.push_back(*p);
    }
  }
  std::vector<hipEvent_t> events(2 * kLayers);
  for (auto& e : events) {
    HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  }

  hipGraph_t graph = CaptureLayers(layers, input, heavy_on_side, main, side, events);
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  std::vector<uint32_t> expected(kElems);
  for (size_t i = 0; i < kElems; ++i) {
    expected[i] = Reference(host_input[i]);
  }

  std::vector<uint32_t> result(kElems);
  for (int replay = 0; replay < kReplays; ++replay) {
    // Poison every intermediate so a node that runs before its producer reads garbage.
    for (uint32_t* p : all_buffers) {
      HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(p), kPoison, kElems, main));
    }
    HIP_CHECK(hipGraphLaunch(graph_exec, main));
    HIP_CHECK(hipMemcpyAsync(result.data(), layers.back().out, bytes, hipMemcpyDeviceToHost,
                             main));
    HIP_CHECK(hipStreamSynchronize(main));

    size_t mismatches = 0;
    size_t first_bad = 0;
    for (size_t i = 0; i < kElems; ++i) {
      if (result[i] != expected[i]) {
        if (mismatches++ == 0) first_bad = i;
      }
    }
    INFO("replay " << replay << ": " << mismatches << " mismatches, first at " << first_bad);
    REQUIRE(mismatches == 0);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  for (auto& e : events) {
    HIP_CHECK(hipEventDestroy(e));
  }
  for (uint32_t* p : all_buffers) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(input));
  HIP_CHECK(hipStreamDestroy(side));
  HIP_CHECK(hipStreamDestroy(main));
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Captures a 32-layer dual-stream graph (event fork/join per layer) with the one-kernel
 *    branch on the side stream, replays it 20 times with poisoned intermediates and compares the
 *    result with a host reference.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphDualStreamCapture.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipGraphDualStreamCapture_LightBranchOnSideStream) {
  RunDualStreamGraph(false);
}

/**
 * Test Description
 * ------------------------
 *  - Same as above with the three-kernel branch on the side stream, so the join waits for the
 *    longer branch.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphDualStreamCapture.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipGraphDualStreamCapture_HeavyBranchOnSideStream) {
  RunDualStreamGraph(true);
}

/**
 * End doxygen group GraphTest.
 * @}
 */
