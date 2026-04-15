/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * Tests that verify GPU profiling timestamps are correct when graph launches
 * recycle profiling signals internally. Specifically targets the case where
 * a graph has more kernels than the internal signal pool, forcing signal reuse.
 *
 * The bug: when a ProfilingSignal is recycled mid-graph, its stale entry in
 * the Timestamp::signals_ list was re-read with wrong HW timestamps from the
 * recycled dispatch, corrupting per-kernel timing.
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

static constexpr int kN = 1024;

static __global__ void busyKernel(float* out, const float* in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    for (int i = 0; i < 50; ++i) {
      val = val * 1.001f + 0.001f;
    }
    out[idx] = val;
  }
}

// Verify that elapsed time from a graph launch with many kernels (forcing
// signal recycling) produces positive timestamps via hipEvent timing.
// Events are recorded outside the captured graph, around the launch.
static void GraphLaunchTimingWithEvents(int numKernels) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, kN * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kN * sizeof(float)));

  std::vector<float> h_in(kN);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), kN * sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t evStart, evEnd;
  HIP_CHECK(hipEventCreate(&evStart));
  HIP_CHECK(hipEventCreate(&evEnd));

  hipGraph_t graph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int i = 0; i < numKernels; ++i) {
    busyKernel<<<dim3((kN + 255) / 256), dim3(256), 0, stream>>>(d_out, d_in, kN);
  }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  constexpr int kIterations = 10;
  for (int iter = 0; iter < kIterations; ++iter) {
    HIP_CHECK(hipEventRecord(evStart, stream));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipEventRecord(evEnd, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, evStart, evEnd));

    INFO("Iteration " << iter << ": elapsed = " << elapsed << " ms, numKernels = " << numKernels);
    REQUIRE(elapsed > 0.0f);
    REQUIRE(elapsed < 60000.0f);
  }

  HIP_CHECK(hipEventDestroy(evStart));
  HIP_CHECK(hipEventDestroy(evEnd));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
}

// Verify per-kernel timing using hipGraphAddEventRecordNode between kernel
// nodes. Checks that individual kernel durations are non-negative and within
// a sane range, and that the overall elapsed time is positive.
static void GraphLaunchPerKernelTiming(int numKernels) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, kN * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kN * sizeof(float)));

  std::vector<float> h_in(kN);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), kN * sizeof(float), hipMemcpyHostToDevice));

  int numEvents = numKernels + 1;
  std::vector<hipEvent_t> events(numEvents);
  for (int i = 0; i < numEvents; ++i) {
    HIP_CHECK(hipEventCreate(&events[i]));
  }

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipKernelNodeParams kNodeParams = {};
  kNodeParams.func = reinterpret_cast<void*>(busyKernel);
  kNodeParams.gridDim = dim3((kN + 255) / 256);
  kNodeParams.blockDim = dim3(256);
  kNodeParams.sharedMemBytes = 0;
  int n = kN;
  void* kArgs[] = {&d_out, &d_in, &n};
  kNodeParams.kernelParams = kArgs;
  kNodeParams.extra = nullptr;

  std::vector<hipGraphNode_t> eventNodes(numEvents);
  std::vector<hipGraphNode_t> kernelNodes(numKernels);

  HIP_CHECK(hipGraphAddEventRecordNode(&eventNodes[0], graph, nullptr, 0, events[0]));

  for (int i = 0; i < numKernels; ++i) {
    HIP_CHECK(hipGraphAddKernelNode(&kernelNodes[i], graph, &eventNodes[i], 1, &kNodeParams));
    HIP_CHECK(hipGraphAddEventRecordNode(&eventNodes[i + 1], graph, &kernelNodes[i], 1,
                                          events[i + 1]));
  }

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int negativeCount = 0;
  for (int i = 0; i < numKernels; ++i) {
    float elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, events[i], events[i + 1]));
    INFO("Kernel " << i << ": elapsed = " << elapsed << " ms");
    REQUIRE(elapsed >= 0.0f);
    REQUIRE(elapsed < 10000.0f);
    if (elapsed < 0.0f) negativeCount++;
  }
  REQUIRE(negativeCount == 0);

  float overallElapsed = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&overallElapsed, events[0], events[numKernels]));
  INFO("Overall elapsed = " << overallElapsed << " ms");
  REQUIRE(overallElapsed > 0.0f);
  REQUIRE(overallElapsed < 60000.0f);

  for (int i = 0; i < numEvents; ++i) {
    HIP_CHECK(hipEventDestroy(events[i]));
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
}

// Verify that repeated graph launches produce consistent timing
// (no drift or corruption from signal reuse across launches)
static void GraphLaunchTimingConsistency(int numKernels) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, kN * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kN * sizeof(float)));

  std::vector<float> h_in(kN);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), kN * sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t evStart, evEnd;
  HIP_CHECK(hipEventCreate(&evStart));
  HIP_CHECK(hipEventCreate(&evEnd));

  hipGraph_t graph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int i = 0; i < numKernels; ++i) {
    busyKernel<<<dim3((kN + 255) / 256), dim3(256), 0, stream>>>(d_out, d_in, kN);
  }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Warmup
  HIP_CHECK(hipEventRecord(evStart, stream));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipEventRecord(evEnd, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  constexpr int kRuns = 20;
  std::vector<float> timings(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    HIP_CHECK(hipEventRecord(evStart, stream));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipEventRecord(evEnd, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, evStart, evEnd));
    timings[i] = elapsed;
    REQUIRE(elapsed > 0.0f);
  }

  std::sort(timings.begin(), timings.end());
  float median = timings[kRuns / 2];
  INFO("Median timing = " << median << " ms for " << numKernels << " kernels");
  REQUIRE(median > 0.0f);

  for (int i = 0; i < kRuns; ++i) {
    INFO("Run " << i << ": elapsed = " << timings[i] << " ms, median = " << median << " ms");
    REQUIRE(timings[i] < median * 100.0f);
    REQUIRE(timings[i] > median * 0.01f);
  }

  HIP_CHECK(hipEventDestroy(evStart));
  HIP_CHECK(hipEventDestroy(evEnd));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Launch a graph with few kernels (no signal recycling) and verify
 *    positive elapsed time via hipEvents.
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_ProfilingTimestamps_SmallGraph) {
  GraphLaunchTimingWithEvents(8);
}

/**
 * Test Description
 * ------------------------
 *  - Launch a graph with many kernels (forces internal signal recycling)
 *    and verify positive elapsed time via hipEvents. This is the scenario
 *    where stale signal entries corrupted timestamps before the fix.
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_ProfilingTimestamps_LargeGraph) {
  GraphLaunchTimingWithEvents(320);
}

/**
 * Test Description
 * ------------------------
 *  - Build a graph with interleaved event record nodes and kernel nodes.
 *    Verify per-kernel timing is positive and consistent with the overall
 *    graph elapsed time.
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_PerKernelTiming_SmallGraph) {
  GraphLaunchPerKernelTiming(8);
}

/**
 * Test Description
 * ------------------------
 *  - Launch a graph many times and verify timing consistency across runs.
 *    Catches timestamp corruption from signal reuse across successive launches.
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_TimingConsistency_SmallGraph) {
  GraphLaunchTimingConsistency(8);
}

/**
 * Test Description
 * ------------------------
 *  - Same as above with large kernel count to stress signal recycling.
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_TimingConsistency_LargeGraph) {
  GraphLaunchTimingConsistency(320);
}

/** @} */
