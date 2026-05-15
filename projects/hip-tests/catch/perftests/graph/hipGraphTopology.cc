/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <iostream>

/**
 * @addtogroup GraphTopologyPerformance GraphTopologyPerformance
 * @{
 * @ingroup GraphTest
 * Performance tests for various graph topologies including straight, parallel, mixed, and hexagon patterns.
 */
__global__ void timing_kernel(uint64_t count) {
#if HT_AMD
  uint64_t begin_time = wall_clock64();
  uint64_t curr_time = begin_time;
  do {
    curr_time = wall_clock64();
  } while (begin_time + count > curr_time);
#endif
#if HT_NVIDIA
  uint64_t begin_time = clock64();
  uint64_t curr_time = begin_time;
  do {
    curr_time = clock64();
  } while (begin_time + count > curr_time);
#endif
}

struct TestOptions {
  std::string topology = "straight";
  int length = 100;
  int width = 1;
  int repeats = 10;
  int warmup = 5;
  uint32_t kernel_duration_us = 0;
  std::string format = "table";
  bool preupload = false;
  bool pre_graph_warmup = true;
  int straight_nodes = 16;
  int parallel_nodes = 8;
};

static hipGraphNode_t add_kernel_node(hipGraph_t graph, hipGraphNode_t* deps, size_t numDeps, uint32_t kernel_duration_us) {
  hipDevice_t device;
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipGetDevice(&device));
#if HT_AMD
  HIPCHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeWallClockRate, device));
#endif
#if HT_NVIDIA
  HIPCHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, device));
#endif
  uint64_t timer_freq_in_hz = clock_rate * 1000;;
  hipKernelNodeParams p{};
  p.func = (void*)timing_kernel;
  p.gridDim = dim3(1, 1, 1);
  p.blockDim = dim3(1, 1, 1);
  p.sharedMemBytes = 0;
  uint64_t timing_count = timer_freq_in_hz * kernel_duration_us / 1000000;
  void* args[] = {&timing_count};
  p.kernelParams = (void**)args;
  p.extra = nullptr;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps, (int)numDeps, &p));
  return node;
}

static hipGraphNode_t add_memcpy_node(hipGraph_t graph, hipGraphNode_t* deps, size_t numDeps, 
                                      void* dst, void* src, size_t size) {
  hipMemcpy3DParms p{};
  p.srcPtr = make_hipPitchedPtr(src, size, 1, 1);
  p.dstPtr = make_hipPitchedPtr(dst, size, 1, 1);
  p.extent = make_hipExtent(size, 1, 1);
  p.kind = hipMemcpyDeviceToDevice;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, deps, (int)numDeps, &p));
  return node;
}

static hipGraphNode_t add_memset_node(hipGraph_t graph, hipGraphNode_t* deps, size_t numDeps, 
                                      void* ptr, int value, size_t size) {
  hipMemsetParams p{};
  p.dst = ptr;
  p.value = value;
  p.pitch = 0;
  p.elementSize = 1;
  p.width = size;
  p.height = 1;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, deps, (int)numDeps, &p));
  return node;
}

static hipGraphNode_t add_empty_node(hipGraph_t graph, hipGraphNode_t* deps, size_t numDeps) {
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, deps, (int)numDeps));
  return node;
}

static void run_graph_topology_test(const TestOptions& opt) {
  CONSOLE_PRINT("\n=== Running Graph Topology Test ===");
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  hipStream_t stream{};

  if (opt.pre_graph_warmup) {
    HIP_CHECK(hipStreamCreate(&stream));
    hipEvent_t warmup_event1{}, warmup_event2{};
    HIP_CHECK(hipEventCreate(&warmup_event1));
    HIP_CHECK(hipEventCreate(&warmup_event2));

    HIP_CHECK(hipEventRecord(warmup_event1, nullptr));
    HIP_CHECK(hipEventRecord(warmup_event2, stream));

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventDestroy(warmup_event1));
    HIP_CHECK(hipEventDestroy(warmup_event2));
  }

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  void* d_mem1 = nullptr;
  void* d_mem2 = nullptr;
  size_t mem_size = 1024;
  if (opt.topology == "mixed") {
    HIP_CHECK(hipMalloc(&d_mem1, mem_size));
    HIP_CHECK(hipMalloc(&d_mem2, mem_size));
  }

  int width = (opt.topology == "straight" || opt.topology == "hexagon") ? 1 : opt.width;
  const long long nodes_total = (opt.topology == "hexagon") ? 
    opt.straight_nodes + 2 * opt.parallel_nodes : (1LL * width * opt.length);

  if (opt.topology == "straight") {
    hipGraphNode_t prev{};
    for (int i = 0; i < opt.length; ++i) {
      if (i == 0) {
        prev = add_kernel_node(graph, nullptr, 0, opt.kernel_duration_us);
      } else {
        hipGraphNode_t n = add_kernel_node(graph, &prev, 1, opt.kernel_duration_us);
        prev = n;
      }
    }
  } else if (opt.topology == "parallel") {
    for (int w = 0; w < width; ++w) {
      hipGraphNode_t prev{};
      for (int i = 0; i < opt.length; ++i) {
        if (i == 0) {
          prev = add_kernel_node(graph, nullptr, 0, opt.kernel_duration_us);
        } else {
          hipGraphNode_t n = add_kernel_node(graph, &prev, 1, opt.kernel_duration_us);
          prev = n;
        }
      }
    }
  } else if (opt.topology == "mixed") {
    // Pattern: memset -> 3 kernels -> memcpy -> 2 kernels -> empty -> 3 kernels -> memset
    std::vector<hipGraphNode_t> all_nodes;
    std::vector<std::string> node_types;
    std::vector<int> kernel_batches;
    hipGraphNode_t prev{};
    
    int kernel_count = 0;
    int current_batch_size = 0;
    int batch_number = 0;
    
    for (int i = 0; i < opt.length; ++i) {
      std::string node_type;
      if (i == 0) {
        // Start with memset
        prev = add_memset_node(graph, nullptr, 0, d_mem1, 0, mem_size);
        all_nodes.push_back(prev);
        node_type = "memset";
        if (current_batch_size > 0) {
          kernel_batches.push_back(current_batch_size);
          current_batch_size = 0;
          batch_number++;
        }
      } else {
        int step = i % 9; // 9-step pattern
        if (step == 1 || step == 2 || step == 3) {
          // 3 consecutive kernels
          hipGraphNode_t n = add_kernel_node(graph, &prev, 1, opt.kernel_duration_us);
          all_nodes.push_back(n);
          prev = n;
          node_type = "kernel";
          kernel_count++;
          current_batch_size++;
        } else if (step == 4) {
          // memcpy (breaks batching)
          if (current_batch_size > 0) {
            kernel_batches.push_back(current_batch_size);
            current_batch_size = 0;
            batch_number++;
          }
          hipGraphNode_t n = add_memcpy_node(graph, &prev, 1, d_mem2, d_mem1, mem_size);
          all_nodes.push_back(n);
          prev = n;
          node_type = "memcpy";
        } else if (step == 5 || step == 6) {
          // 2 consecutive kernels
          hipGraphNode_t n = add_kernel_node(graph, &prev, 1, opt.kernel_duration_us);
          all_nodes.push_back(n);
          prev = n;
          node_type = "kernel";
          kernel_count++;
          current_batch_size++;
        } else if (step == 7) {
          // empty node
          if (current_batch_size > 0) {
            kernel_batches.push_back(current_batch_size);
            current_batch_size = 0;
            batch_number++;
          }
          hipGraphNode_t n = add_empty_node(graph, &prev, 1);
          all_nodes.push_back(n);
          prev = n;
          node_type = "empty";
        } else if (step == 8 || step == 0) {
          // kernel nodes
          hipGraphNode_t n = add_kernel_node(graph, &prev, 1, opt.kernel_duration_us);
          all_nodes.push_back(n);
          prev = n;
          node_type = "kernel";
          kernel_count++;
          current_batch_size++;
        }
      }
      node_types.push_back(node_type);
    }
    
    if (current_batch_size > 0) {
      kernel_batches.push_back(current_batch_size);
    }
    
    CONSOLE_PRINT("\nMixed topology summary:");
    CONSOLE_PRINT("Total nodes: %d", opt.length);
    CONSOLE_PRINT("Kernel nodes: %d", kernel_count);
    CONSOLE_PRINT("Non-kernel nodes: %d", opt.length - kernel_count);
  } else if (opt.topology == "hexagon") {
    CONSOLE_PRINT("Building hexagon topology: %d straight + %d parallel +  %d straight nodes",
           opt.straight_nodes, opt.parallel_nodes, opt.straight_nodes);
    
    const int parallel_path_length = opt.parallel_nodes; 
    const int parallel_paths = 2;       
    
    int straight_nodes = opt.straight_nodes;
    int before_split = straight_nodes / 2;
    int after_join = straight_nodes - before_split;
    
    std::vector<hipGraphNode_t> nodes;
    
    // Step 1: straight line before split
    hipGraphNode_t last_before_split = {};
    for (int i = 0; i < before_split; ++i) {
      if (i == 0) {
        last_before_split = add_kernel_node(graph, nullptr, 0, opt.kernel_duration_us);
      } else {
        hipGraphNode_t n = add_kernel_node(graph, &last_before_split, 1, opt.kernel_duration_us);
        last_before_split = n;
      }
      nodes.push_back(last_before_split);
    }
    
    // Step 2: parallel paths
    std::vector<hipGraphNode_t> path_ends(parallel_paths);
    for (int path = 0; path < parallel_paths; ++path) {
      hipGraphNode_t prev = last_before_split;
      
      for (int i = 0; i < parallel_path_length; ++i) {
        hipGraphNode_t* deps = (before_split > 0) ? &prev : nullptr;
        size_t numDeps = (before_split > 0) ? 1 : 0;
        
        if (i == 0 && before_split > 0) {
          deps = &last_before_split;
          numDeps = 1;
        }
        
        hipGraphNode_t n = add_kernel_node(graph, deps, numDeps, opt.kernel_duration_us);
        nodes.push_back(n);
        prev = n;
      }
      path_ends[path] = prev;
    }
    
    // Step 3: straight line after join
    if (after_join > 0) {
      hipGraphNode_t join_node = add_kernel_node(graph, path_ends.data(), path_ends.size(), opt.kernel_duration_us);
      nodes.push_back(join_node);
      
      hipGraphNode_t prev = join_node;
      for (int i = 1; i < after_join; ++i) {
        hipGraphNode_t n = add_kernel_node(graph, &prev, 1, opt.kernel_duration_us);
        nodes.push_back(n);
        prev = n;
      }
    }
    
    CONSOLE_PRINT("Hexagon topology created: %d total nodes", (int)nodes.size());
  }

  // Instantiate
  hipGraphExec_t gexec{};
  auto t_inst_begin = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
  auto t_inst_end = std::chrono::steady_clock::now();
  double instantiation_us = std::chrono::duration<double, std::micro>(t_inst_end - t_inst_begin).count();

  if (opt.preupload) {
    HIP_CHECK(hipGraphUpload(gexec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Warmup
  for (int i = 0; i < opt.warmup; ++i) {
    HIP_CHECK(hipGraphLaunch(gexec, stream));
    HIP_CHECK(hipDeviceSynchronize());
  }

  // First launch timing
  auto t_first_begin = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphLaunch(gexec, stream));
  auto t_first_end = std::chrono::steady_clock::now();
  HIP_CHECK(hipStreamSynchronize(stream));
  auto t_e2e_end = std::chrono::steady_clock::now();
  double first_launch_cpu_us = std::chrono::duration<double, std::micro>(t_first_end - t_first_begin).count();
  double first_e2e_us = std::chrono::duration<double, std::micro>(t_e2e_end - t_first_begin).count();

  // Repeat launches
  std::vector<double> cpu_over_us; 
  cpu_over_us.reserve(opt.repeats);
  double device_us_sum = 0.0;
  hipEvent_t evt_start{}, evt_stop{};
  HIP_CHECK(hipEventCreate(&evt_start));
  HIP_CHECK(hipEventCreate(&evt_stop));

  for (int r = 0; r < opt.repeats; ++r) {
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

  // Calculate statistics
  double repeat_cpu_avg_us = 0.0;
  for (double v : cpu_over_us) repeat_cpu_avg_us += v;
  repeat_cpu_avg_us /= opt.repeats;

  double repeat_cpu_p50_us = 0.0, repeat_cpu_p99_us = 0.0;
  if (!cpu_over_us.empty()) {
    std::sort(cpu_over_us.begin(), cpu_over_us.end());
    repeat_cpu_p50_us = cpu_over_us[cpu_over_us.size() * 50 / 100];
    repeat_cpu_p99_us = cpu_over_us[cpu_over_us.size() * 99 / 100];
  }
  double repeat_device_avg_us = device_us_sum / opt.repeats;

  auto per_node_ns = [&](double total_us) -> double {
    return (total_us * 1000.0) / nodes_total;
  };

  // Print results
  CONSOLE_PRINT("\nHIP Graph Performance Results");
  CONSOLE_PRINT("=============================");
  CONSOLE_PRINT("Device      : %s", prop.name);
  CONSOLE_PRINT("Topology    : %s | width=%d length=%d | nodes_total=%lld",
         opt.topology.c_str(), width, opt.length, nodes_total);
  CONSOLE_PRINT("Kernel duration: %u | repeats=%d | warmup=%d\n",
         opt.kernel_duration_us, opt.repeats, opt.warmup);

  CONSOLE_PRINT("%-32s %14s   %s", "Metric", "Total", "Per-node");
  CONSOLE_PRINT("%s", std::string(64, '-').c_str());
  CONSOLE_PRINT("%-32s %11.3f us   (%9.1f ns/node)", "Instantiation", instantiation_us, per_node_ns(instantiation_us));
  CONSOLE_PRINT("%-32s %11.3f us   (%9.1f ns/node)", "First launch CPU", first_launch_cpu_us, per_node_ns(first_launch_cpu_us));
  CONSOLE_PRINT("%-32s %11.3f us   (%9.1f ns/node)", "First launch end-to-end", first_e2e_us, per_node_ns(first_e2e_us));
  CONSOLE_PRINT("%-32s %11.3f us   (%9.1f ns/node)", "Repeat launch CPU avg", repeat_cpu_avg_us, per_node_ns(repeat_cpu_avg_us));
  CONSOLE_PRINT("%-32s %11.3f us   (%9.1f ns/node)", "Repeat launch CPU p50", repeat_cpu_p50_us, per_node_ns(repeat_cpu_p50_us));
  CONSOLE_PRINT("%-32s %11.3f us   (%9.1f ns/node)", "Repeat launch CPU p99", repeat_cpu_p99_us, per_node_ns(repeat_cpu_p99_us));
  CONSOLE_PRINT("%-32s %11.3f us   (%9.1f ns/node)", "Device runtime avg(Events)", repeat_device_avg_us, per_node_ns(repeat_device_avg_us));

  // Cleanup
  if (opt.topology == "mixed") {
    if (d_mem1) HIP_CHECK(hipFree(d_mem1));
    if (d_mem2) HIP_CHECK(hipFree(d_mem2));
  }

  HIP_CHECK(hipGraphExecDestroy(gexec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test straight topology graph performance
 */
TEST_CASE("Perf_GraphTopology_Straight") {
  TestOptions opt;
  opt.topology = "straight";
  opt.length = 50;
  opt.repeats = 5;
  opt.warmup = 2;
  run_graph_topology_test(opt);
}

/**
 * Test parallel topology graph performance
 */
TEST_CASE("Perf_GraphTopology_Parallel") {
  TestOptions opt;
  opt.topology = "parallel";
  opt.length = 25;
  opt.width = 4;
  opt.repeats = 5;
  opt.warmup = 2;
  run_graph_topology_test(opt);
}

/**
 * Test hexagon topology graph performance
 */
TEST_CASE("Perf_GraphTopology_Hexagon") {
  TestOptions opt;
  opt.topology = "hexagon";
  opt.straight_nodes = 20;
  opt.parallel_nodes = 8;
  opt.repeats = 5;
  opt.warmup = 2;
  run_graph_topology_test(opt);
}

/**
 * Test mixed topology graph performance
 */
TEST_CASE("Perf_GraphTopology_Mixed") {
  TestOptions opt;
  opt.topology = "mixed";
  opt.length = 27; // 3 cycles of 9-step pattern  
  opt.repeats = 5;
  opt.warmup = 2;
  run_graph_topology_test(opt);
}