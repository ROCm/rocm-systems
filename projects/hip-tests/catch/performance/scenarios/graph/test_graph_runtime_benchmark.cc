/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

// Each lane writes its own word so parallel graph nodes have no data race.
__global__ void GraphRuntimeStep(unsigned* value, unsigned ticks) {
  // wall_clock64 is stable for elapsed work on AMD, including gfx11.
#if HT_AMD
  const unsigned long long start = wall_clock64();
  while (wall_clock64() - start < ticks) {
  }
#else
  const unsigned long long start = clock64();
  while (clock64() - start < ticks) {
  }
#endif
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ++*value;
  }
}

struct GraphShape {
  const char* name;
  int lanes;
  int nodes_per_lane;
  unsigned ticks;
  bool diamond;
  bool full_wave;
};

hipGraphNode_t AddStep(hipGraph_t graph, unsigned* value, unsigned ticks,
                       unsigned blocks, const hipGraphNode_t* dependencies,
                       size_t dependency_count) {
  hipKernelNodeParams params{};
  void* args[] = {&value, &ticks};
  params.func = reinterpret_cast<void*>(GraphRuntimeStep);
  params.gridDim = dim3(blocks);
  params.blockDim = dim3(blocks == 1 ? 1 : 256);
  params.kernelParams = args;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, dependencies, dependency_count, &params));
  return node;
}

unsigned BlocksForShape(const GraphShape& shape) {
  if (!shape.full_wave) return 1;
  int device_id = 0;
  hipDeviceProp_t device{};
  HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipGetDeviceProperties(&device, device_id));
  const size_t machine_threads =
      static_cast<size_t>(device.multiProcessorCount) * device.maxThreadsPerMultiProcessor;
  return static_cast<unsigned>((machine_threads + 255) / 256);
}

hipGraph_t MakeGraph(const GraphShape& shape, unsigned* data, unsigned blocks) {
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  if (shape.diamond) {
    const auto root = AddStep(graph, data, shape.ticks, blocks, nullptr, 0);
    const auto left = AddStep(graph, data + 1, shape.ticks, blocks, &root, 1);
    const auto right = AddStep(graph, data + 2, shape.ticks, blocks, &root, 1);
    const std::array<hipGraphNode_t, 2> join{left, right};
    AddStep(graph, data + 3, shape.ticks, blocks, join.data(), join.size());
  } else {
    for (int lane = 0; lane < shape.lanes; ++lane) {
      hipGraphNode_t previous = nullptr;
      for (int node = 0; node < shape.nodes_per_lane; ++node) {
        previous = AddStep(graph, data + lane, shape.ticks, blocks,
                           previous == nullptr ? nullptr : &previous,
                           previous == nullptr ? 0 : 1);
      }
    }
  }
  return graph;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                : values[middle];
}

void MeasureGraph(const GraphShape& shape) {
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  unsigned* data = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipMalloc(&data, 4 * sizeof(unsigned)));

  const unsigned blocks = BlocksForShape(shape);
  const hipGraph_t graph = MakeGraph(shape, data, blocks);
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Keep roughly 1,000 nodes per lane per trial. Report time per graph launch.
  const int launches = shape.full_wave ? 200 : 1000 / shape.nodes_per_lane;
  std::vector<double> event_interval_us;
  std::vector<double> host_us;
  for (int trial = -2; trial < 8; ++trial) {
    HIP_CHECK(hipMemsetAsync(data, 0, 4 * sizeof(unsigned), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventRecord(start, stream));
    const auto host_start = std::chrono::steady_clock::now();
    for (int launch = 0; launch < launches; ++launch) {
      HIP_CHECK(hipGraphLaunch(exec, stream));
    }
    const auto host_stop = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float gpu_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&gpu_ms, start, stop));

    std::array<unsigned, 4> actual{};
    HIP_CHECK(hipMemcpy(actual.data(), data, sizeof(actual), hipMemcpyDeviceToHost));
    for (int lane = 0; lane < 4; ++lane) {
      const unsigned expected = shape.diamond ? static_cast<unsigned>(launches)
                                : lane < shape.lanes
                                    ? static_cast<unsigned>(launches * shape.nodes_per_lane)
                                    : 0u;
      REQUIRE(actual[lane] == expected);
    }
    if (trial >= 0) {
      // This interval includes any GPU idle time while the host submits the
      // batch. It is steady-state throughput, not an isolated kernel latency.
      event_interval_us.push_back(static_cast<double>(gpu_ms) * 1000.0 / launches);
      host_us.push_back(std::chrono::duration<double, std::micro>(host_stop - host_start)
                            .count() / launches);
    }
  }

  CONSOLE_PRINT("graph_runtime_benchmark,%s,%d,%d,%u,%d,%u,%.3f,%.3f", shape.name,
                shape.lanes, shape.nodes_per_lane, shape.ticks, launches,
                blocks, Median(event_interval_us), Median(host_us));

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));
}

}  // namespace

// No performance threshold: run the same cases on each runtime and compare
// medians from independent processes. In particular, tiny diamond and short
// lanes guard against scheduling extra hardware queues for insufficient work.
HIP_TEST_CASE(Performance_GraphRuntimeScheduling) {
  const std::array<GraphShape, 11> shapes{{
      {"serial_tiny", 1, 1, 0, false, false},
      {"two_tiny", 2, 1, 0, false, false},
      {"three_tiny", 3, 1, 0, false, false},
      {"four_tiny", 4, 1, 0, false, false},
      {"diamond_tiny", 2, 1, 0, true, false},
      {"two_group50_tiny", 2, 50, 0, false, false},
      {"two_group2_wave", 2, 2, 2000, false, true},
      {"two_long", 2, 1, 15000, false, false},
      {"four_long", 4, 1, 15000, false, false},
      {"diamond_long", 2, 1, 15000, true, false},
      {"four_group50", 4, 50, 2000, false, false},
  }};
  CONSOLE_PRINT("graph_runtime_benchmark,shape,lanes,nodes_per_lane,ticks,launches,blocks,event_interval_us_per_launch,host_submit_us_per_launch");
  for (const auto& shape : shapes) {
    MeasureGraph(shape);
  }
}
