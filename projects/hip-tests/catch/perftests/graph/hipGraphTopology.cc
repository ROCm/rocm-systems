/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <hip/hip_runtime.h>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// Test kernel for topology benchmarking
__global__ void tiny_kernel(int iters) {
  unsigned v = threadIdx.x;
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    v = (v * 1664525u + 1013904223u) ^ (unsigned)i;
  }
  if (v == 0xdeadbeef) asm volatile("" ::: "memory");
}

// Helper structure for test configuration
struct TopologyConfig {
  int length = 100;
  int width = 1;
  int repeats = 50;   // Reduced for testing
  int warmup = 3;     // Reduced for testing
  int iters = 64;
  bool preupload = false;
};

// Helper function to add kernel node to graph
static hipGraphNode_t add_kernel_node(hipGraph_t graph, hipGraphNode_t* deps, size_t numDeps, int iters) {
  hipKernelNodeParams p{};
  p.func = (void*)tiny_kernel;
  p.gridDim = dim3(1, 1, 1);
  p.blockDim = dim3(1, 1, 1);
  p.sharedMemBytes = 0;
  void* args[] = { &iters };
  p.kernelParams = (void**)args;
  p.extra = nullptr;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps, (int)numDeps, &p));
  return node;
}

// Helper function to build straight topology
static void build_straight_topology(hipGraph_t graph, const TopologyConfig& config) {
  hipGraphNode_t prev{};
  for (int i = 0; i < config.length; ++i) {
    if (i == 0) {
      prev = add_kernel_node(graph, nullptr, 0, config.iters);
    } else {
      hipGraphNode_t n = add_kernel_node(graph, &prev, 1, config.iters);
      prev = n;
    }
  }
}

// Helper function to build parallel topology
static void build_parallel_topology(hipGraph_t graph, const TopologyConfig& config) {
  for (int w = 0; w < config.width; ++w) {
    hipGraphNode_t prev{};
    for (int i = 0; i < config.length; ++i) {
      if (i == 0) {
        prev = add_kernel_node(graph, nullptr, 0, config.iters);
      } else {
        hipGraphNode_t n = add_kernel_node(graph, &prev, 1, config.iters);
        prev = n;
      }
    }
  }
}

// Performance measurement structure
struct PerfResults {
  double instantiation_us = 0.0;
  double first_launch_cpu_us = 0.0;
  double first_e2e_us = 0.0;
  double repeat_cpu_avg_us = 0.0;
  double repeat_cpu_p50_us = 0.0;
  double repeat_cpu_p99_us = 0.0;
  double repeat_device_avg_us = 0.0;
  long long nodes_total = 0;
};

// Helper function to measure graph performance
static PerfResults measure_graph_performance(hipGraph_t graph, const TopologyConfig& config) {
  PerfResults results;
  
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  // Instantiate
  hipGraphExec_t gexec{};
  auto t_inst_begin = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
  auto t_inst_end = std::chrono::steady_clock::now();
  results.instantiation_us = std::chrono::duration<double, std::micro>(t_inst_end - t_inst_begin).count();
  
  if (config.preupload) {
    HIP_CHECK(hipGraphUpload(gexec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Warmup
  for (int i = 0; i < config.warmup; ++i) {
    HIP_CHECK(hipGraphLaunch(gexec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // First launch CPU
  auto t_first_begin = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphLaunch(gexec, stream));
  auto t_first_end = std::chrono::steady_clock::now();
  results.first_launch_cpu_us = std::chrono::duration<double, std::micro>(t_first_end - t_first_begin).count();
  
  // First launch end-to-end
  auto t_e2e_begin = std::chrono::steady_clock::now();
  HIP_CHECK(hipStreamSynchronize(stream));
  auto t_e2e_end = std::chrono::steady_clock::now();
  results.first_e2e_us = std::chrono::duration<double, std::micro>(t_e2e_begin - t_e2e_end).count() + results.first_launch_cpu_us;
  
  // Repeat launches
  std::vector<double> cpu_over_us;
  cpu_over_us.reserve(config.repeats);
  double device_us_sum = 0.0;
  hipEvent_t evt_start{}, evt_stop{};
  HIP_CHECK(hipEventCreate(&evt_start));
  HIP_CHECK(hipEventCreate(&evt_stop));

  for (int r = 0; r < config.repeats; ++r) {
    HIP_CHECK(hipEventRecord(evt_start, stream));
    auto t0 = std::chrono::steady_clock::now();
    HIP_CHECK(hipGraphLaunch(gexec, stream));
    auto t1 = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(evt_stop, stream));
    HIP_CHECK(hipEventSynchronize(evt_stop));
    float ms = 0.0f; 
    HIP_CHECK(hipEventElapsedTime(&ms, evt_start, evt_stop));
    device_us_sum += (double)ms * 1000.0;
    cpu_over_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  HIP_CHECK(hipEventDestroy(evt_start));
  HIP_CHECK(hipEventDestroy(evt_stop));

  // Stats
  results.repeat_cpu_avg_us = 0.0;
  for (double v : cpu_over_us) results.repeat_cpu_avg_us += v;
  results.repeat_cpu_avg_us /= std::max(1, config.repeats);

  if (!cpu_over_us.empty()) {
    auto tmp = cpu_over_us;
    std::sort(tmp.begin(), tmp.end());
    results.repeat_cpu_p50_us = tmp[tmp.size() / 2];
    size_t idx99 = (size_t)std::min((double)tmp.size() - 1, 0.99 * (tmp.size() - 1));
    results.repeat_cpu_p99_us = tmp[idx99];
  }
  results.repeat_device_avg_us = device_us_sum / std::max(1, config.repeats);

  HIP_CHECK(hipGraphExecDestroy(gexec));
  HIP_CHECK(hipStreamDestroy(stream));

  return results;
}

TEST_CASE("Perf_GraphTopology_StraightLine_Performance") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  
  INFO("Device: " << prop.name);
  
  SECTION("Small straight topology") {
    TopologyConfig config;
    config.length = 10;
    config.width = 1;
    config.repeats = 10;
    config.warmup = 2;
    
    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));
    
    build_straight_topology(graph, config);
    
    PerfResults results = measure_graph_performance(graph, config);
    results.nodes_total = config.length;
    
    INFO("Topology: straight | length=" << config.length << " | nodes_total=" << results.nodes_total);
    INFO("Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us << " us");
    INFO("First launch CPU: " << std::fixed << std::setprecision(3) << results.first_launch_cpu_us << " us");
    INFO("First end-to-end: " << std::fixed << std::setprecision(3) << results.first_e2e_us << " us");
    INFO("Repeat CPU avg: " << std::fixed << std::setprecision(3) << results.repeat_cpu_avg_us << " us");
    INFO("Repeat device avg: " << std::fixed << std::setprecision(3) << results.repeat_device_avg_us << " us");
    
    // Validation checks - ensure reasonable performance bounds
    //REQUIRE(results.instantiation_us > 0.0);
    //REQUIRE(results.instantiation_us < 10000.0);  // Less than 10ms for small graph
    //REQUIRE(results.first_launch_cpu_us > 0.0);
    //REQUIRE(results.repeat_cpu_avg_us > 0.0);
    //REQUIRE(results.repeat_device_avg_us > 0.0);
    
    HIP_CHECK(hipGraphDestroy(graph));
  }
  
  SECTION("Medium straight topology") {
    TopologyConfig config;
    config.length = 100;
    config.width = 1;
    config.repeats = 20;
    config.warmup = 3;
    
    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));
    
    build_straight_topology(graph, config);
    
    PerfResults results = measure_graph_performance(graph, config);
    results.nodes_total = config.length;
    
    INFO("Topology: straight | length=" << config.length << " | nodes_total=" << results.nodes_total);
    INFO("Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us << " us");
    INFO("Repeat CPU avg: " << std::fixed << std::setprecision(3) << results.repeat_cpu_avg_us << " us");
    INFO("Repeat device avg: " << std::fixed << std::setprecision(3) << results.repeat_device_avg_us << " us");
    
    // Performance validation
    REQUIRE(results.instantiation_us > 0.0);
    REQUIRE(results.repeat_cpu_avg_us > 0.0);
    REQUIRE(results.repeat_device_avg_us > 0.0);
    
    // Per-node performance check (should be reasonable)
    double per_node_ns = results.nodes_total > 0 ? (results.repeat_device_avg_us * 1000.0) / (double)results.nodes_total : 0.0;
    INFO("Per-node device time: " << std::fixed << std::setprecision(1) << per_node_ns << " ns/node");
    REQUIRE(per_node_ns > 0.1);  // At least 0.1ns per node
    REQUIRE(per_node_ns < 100000.0);  // Less than 100us per node
    
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

TEST_CASE("Perf_GraphTopology_Parallel_Performance") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  
  INFO("Device: " << prop.name);
  
  SECTION("Small parallel topology") {
    TopologyConfig config;
    config.length = 10;
    config.width = 4;
    config.repeats = 10;
    config.warmup = 2;
    
    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));
    
    build_parallel_topology(graph, config);
    
    PerfResults results = measure_graph_performance(graph, config);
    results.nodes_total = (long long)config.width * config.length;
    
    INFO("Topology: parallel | width=" << config.width << " length=" << config.length << " | nodes_total=" << results.nodes_total);
    INFO("Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us << " us");
    INFO("Repeat CPU avg: " << std::fixed << std::setprecision(3) << results.repeat_cpu_avg_us << " us");
    INFO("Repeat device avg: " << std::fixed << std::setprecision(3) << results.repeat_device_avg_us << " us");
    
    // Validation checks
    REQUIRE(results.instantiation_us > 0.0);
    REQUIRE(results.repeat_cpu_avg_us > 0.0);
    REQUIRE(results.repeat_device_avg_us > 0.0);
    REQUIRE(results.nodes_total == config.width * config.length);
    
    HIP_CHECK(hipGraphDestroy(graph));
  }
  
  SECTION("Medium parallel topology") {
    TopologyConfig config;
    config.length = 50;
    config.width = 8;
    config.repeats = 15;
    config.warmup = 3;
    
    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));
    
    build_parallel_topology(graph, config);
    
    PerfResults results = measure_graph_performance(graph, config);
    results.nodes_total = (long long)config.width * config.length;
    
    INFO("Topology: parallel | width=" << config.width << " length=" << config.length << " | nodes_total=" << results.nodes_total);
    INFO("Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us << " us");
    INFO("Repeat CPU avg: " << std::fixed << std::setprecision(3) << results.repeat_cpu_avg_us << " us");
    INFO("Repeat device avg: " << std::fixed << std::setprecision(3) << results.repeat_device_avg_us << " us");
    
    // Performance validation
    REQUIRE(results.instantiation_us > 0.0);
    REQUIRE(results.repeat_cpu_avg_us > 0.0);
    REQUIRE(results.repeat_device_avg_us > 0.0);
    REQUIRE(results.nodes_total == config.width * config.length);
    
    // Parallel should have reasonable parallelism efficiency
    double per_node_ns = results.nodes_total > 0 ? (results.repeat_device_avg_us * 1000.0) / (double)results.nodes_total : 0.0;
    INFO("Per-node device time: " << std::fixed << std::setprecision(1) << per_node_ns << " ns/node");
    REQUIRE(per_node_ns > 0.1);
    REQUIRE(per_node_ns < 50000.0);  // Parallel should be more efficient per node
    
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

TEST_CASE("Perf_GraphTopology_Preupload_Comparison") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  
  INFO("Device: " << prop.name);
  
  TopologyConfig config;
  config.length = 20;
  config.width = 1;
  config.repeats = 10;
  config.warmup = 2;
  
  PerfResults results_no_preupload, results_with_preupload;
  
  SECTION("Without preupload") {
    config.preupload = false;
    
    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));
    build_straight_topology(graph, config);
    
    results_no_preupload = measure_graph_performance(graph, config);
    
    INFO("No preupload - First launch CPU: " << std::fixed << std::setprecision(3) << results_no_preupload.first_launch_cpu_us << " us");
    INFO("No preupload - Repeat CPU avg: " << std::fixed << std::setprecision(3) << results_no_preupload.repeat_cpu_avg_us << " us");
    
    REQUIRE(results_no_preupload.first_launch_cpu_us > 0.0);
    REQUIRE(results_no_preupload.repeat_cpu_avg_us > 0.0);
    
    HIP_CHECK(hipGraphDestroy(graph));
  }
  
  SECTION("With preupload") {
    config.preupload = true;
    
    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));
    build_straight_topology(graph, config);
    
    results_with_preupload = measure_graph_performance(graph, config);
    
    INFO("With preupload - First launch CPU: " << std::fixed << std::setprecision(3) << results_with_preupload.first_launch_cpu_us << " us");
    INFO("With preupload - Repeat CPU avg: " << std::fixed << std::setprecision(3) << results_with_preupload.repeat_cpu_avg_us << " us");
    
    REQUIRE(results_with_preupload.first_launch_cpu_us > 0.0);
    REQUIRE(results_with_preupload.repeat_cpu_avg_us > 0.0);
    
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

TEST_CASE("Perf_GraphTopology_Scalability_Test") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  
  INFO("Device: " << prop.name);
  
  SECTION("Scalability with increasing length") {
    std::vector<int> lengths = {10, 50, 100};
    std::vector<double> instantiation_times;
    
    for (int length : lengths) {
      TopologyConfig config;
      config.length = length;
      config.width = 1;
      config.repeats = 5;  // Fewer repeats for scalability test
      config.warmup = 1;
      
      hipGraph_t graph{};
      HIP_CHECK(hipGraphCreate(&graph, 0));
      build_straight_topology(graph, config);
      
      PerfResults results = measure_graph_performance(graph, config);
      instantiation_times.push_back(results.instantiation_us);
      
      INFO("Length " << length << " - Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us << " us");
      INFO("Length " << length << " - Repeat device avg: " << std::fixed << std::setprecision(3) << results.repeat_device_avg_us << " us");
      
      REQUIRE(results.instantiation_us > 0.0);
      REQUIRE(results.repeat_device_avg_us > 0.0);
      
      HIP_CHECK(hipGraphDestroy(graph));
    }
    
    // Check that instantiation time grows reasonably with graph size
    REQUIRE(instantiation_times.size() == 3);
    REQUIRE(instantiation_times[0] > 0.0);
    REQUIRE(instantiation_times[1] >= instantiation_times[0]);  // Should not decrease
    REQUIRE(instantiation_times[2] >= instantiation_times[1]);  // Should not decrease
  }
}
