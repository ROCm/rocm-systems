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
 * parallel, and joins them with an add whose output feeds the next layer. The executor spreads
 * the parallel branches of each level over its hardware queues, so every fork and join becomes a
 * cross-queue wait. Every kernel reads its inputs first, then spins, then writes: a consumer that
 * starts before its producer has finished reads poison, whichever branch is the long one. Every
 * intermediate buffer is poisoned before each replay and the result is compared exactly with a
 * host reference, so a dropped fork or join wait fails the test.
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kLayers = 32;
constexpr size_t kElems = 64 * 1024;
constexpr int kReplays = 20;
constexpr uint32_t kPoison = 0xDEADBEEFu;

// Spin lengths in microseconds. The three-kernel branch is the long one.
constexpr unsigned long long kHeavySpinUs = 200;
constexpr unsigned long long kLightSpinUs = 20;
constexpr unsigned long long kAddSpinUs = 20;

unsigned long long WallClockTicksPerUs() {
  int device = 0;
  int wall_clock_khz = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, device));
  return std::max(1ULL, static_cast<unsigned long long>(wall_clock_khz) / 1000);
}

// Reads before the spin, so the read happens as early as the kernel starts; the volatile load and
// the compiler barrier keep it from being moved after the loop.
__device__ uint32_t LoadEarly(const uint32_t* p) {
  const uint32_t v = *static_cast<const volatile uint32_t*>(p);
  asm volatile("" ::: "memory");
  return v;
}

__device__ void Spin(unsigned long long ticks) {
  const unsigned long long start = wall_clock64();
  while (wall_clock64() - start < ticks) {
  }
}

__global__ void MulAddSpin(uint32_t* out, const uint32_t* in, uint32_t mul, uint32_t add,
                           unsigned long long spin_ticks, size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const uint32_t x = i < n ? LoadEarly(&in[i]) : 0;
  Spin(spin_ticks);
  if (i < n) {
    out[i] = x * mul + add;
  }
}

__global__ void AddSpin(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        unsigned long long spin_ticks, size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const uint32_t x = i < n ? LoadEarly(&a[i]) : 0;
  const uint32_t y = i < n ? LoadEarly(&b[i]) : 0;
  Spin(spin_ticks);
  if (i < n) {
    out[i] = x + y;
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
    // 10 keeps the per-layer multiplier odd (135 + 10): with an even one the product over the
    // layers is 0 mod 2^32, every element ends at the same value and early errors are erased.
    const uint32_t shared = x * 10u + static_cast<uint32_t>(l);
    x = routed + shared;
  }
  return x;
}

void RoutedBranch(const LayerBuffers& b, const uint32_t* in, int layer, unsigned long long ticks,
                  hipStream_t s) {
  const dim3 block(256);
  const dim3 grid((kElems + block.x - 1) / block.x);
  MulAddSpin<<<grid, block, 0, s>>>(b.h1, in, 3u, static_cast<uint32_t>(layer), ticks, kElems);
  MulAddSpin<<<grid, block, 0, s>>>(b.h2, b.h1, 5u, 7u, ticks, kElems);
  MulAddSpin<<<grid, block, 0, s>>>(b.routed, b.h2, 9u, 1u, ticks, kElems);
}

void SharedBranch(const LayerBuffers& b, const uint32_t* in, int layer, unsigned long long ticks,
                  hipStream_t s) {
  const dim3 block(256);
  const dim3 grid((kElems + block.x - 1) / block.x);
  MulAddSpin<<<grid, block, 0, s>>>(b.shared, in, 10u, static_cast<uint32_t>(layer), ticks,
                                    kElems);
}

// Captures the layers from |main| and |side|. With |heavy_on_side| the three-kernel branch is
// captured on the side stream ("reversed"), otherwise the one-kernel branch is ("original").
// The executor assigns queues per dependency level, not per capture stream, so the two cases
// differ in the order the branches are captured, not in which stream runs them.
hipGraph_t CaptureLayers(const std::vector<LayerBuffers>& layers, const uint32_t* input,
                         bool heavy_on_side, hipStream_t main, hipStream_t side,
                         std::vector<hipEvent_t>& events) {
  const unsigned long long ticks_per_us = WallClockTicksPerUs();
  const unsigned long long heavy = kHeavySpinUs * ticks_per_us;
  const unsigned long long light = kLightSpinUs * ticks_per_us;
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
      RoutedBranch(b, in, l, heavy, side);
      SharedBranch(b, in, l, light, main);
    } else {
      SharedBranch(b, in, l, light, side);
      RoutedBranch(b, in, l, heavy, main);
    }
    // Join: the add waits for the side-stream branch.
    HIP_CHECK(hipEventRecord(join, side));
    HIP_CHECK(hipStreamWaitEvent(main, join, 0));
    const dim3 block(256);
    const dim3 grid((kElems + block.x - 1) / block.x);
    AddSpin<<<grid, block, 0, main>>>(b.out, b.routed, b.shared, kAddSpinUs * ticks_per_us,
                                      kElems);
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
  // A layer map that isn't one-to-one would collapse the elements to one value and hide errors.
  REQUIRE(expected[0] != expected[1]);

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
 *    branch captured on the side stream, as in the PyTorch dual-stream MoE pattern, replays it 20
 *    times with poisoned intermediates and compares the result with a host reference.
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
 *  - Same as above with the three-kernel branch captured on the side stream, which changes the
 *    order the executor sees the two branches of each layer in.
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
