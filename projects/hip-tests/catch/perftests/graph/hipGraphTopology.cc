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

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <hip/hip_runtime.h>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// Test kernel for performance benchmarking
__global__ void tiny_kernel(int iters) {
  unsigned v = threadIdx.x;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    v = (v * 1664525u + 1013904223u) ^ (unsigned)i;
  }
  if (v == 0xdeadbeef) asm volatile("" ::: "memory");
}

struct options {
  std::string topology = "straight";
  int length = 100;
  int width = 1;
  int repeats = 100;
  int warmup = 5;
  int iters = 64;
  bool preupload = false;
};

struct PerfResults {
  double instantiation_us = 0.0;
  double first_launch_cpu_us = 0.0;
  double first_e2e_us = 0.0;
  double repeat_cpu_avg_us = 0.0;
  double repeat_cpu_p50_us = 0.0;
  double repeat_cpu_p99_us = 0.0;
  double repeat_device_avg_us = 0.0;
  long long nodes_total = 0;
  int kernel_count = 0;
  int batches_count = 0;
  std::vector<int> batch_sizes;
};

// Helper function to add kernel node to graph
static hipGraphNode_t add_kernel_node(hipGraph_t graph, hipGraphNode_t* deps, size_t numDeps,
                                      int iters) {
  hipKernelNodeParams p{};
  p.func = (void*)tiny_kernel;
  p.gridDim = dim3(1, 1, 1);
  p.blockDim = dim3(1, 1, 1);
  p.sharedMemBytes = 0;
  void* args[] = {&iters};
  p.kernelParams = (void**)args;
  p.extra = nullptr;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps, (int)numDeps, &p));
  return node;
}

// Helper function to add memcpy node
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

// Helper function to add memset node
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

// Helper function to add empty node
static hipGraphNode_t add_empty_node(hipGraph_t graph, hipGraphNode_t* deps, size_t numDeps) {
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, deps, (int)numDeps));
  return node;
}

// Helper function to build straight topology
static void build_straight_topology(hipGraph_t graph, const options& options) {
  hipGraphNode_t prev{};
  for (int i = 0; i < options.length; ++i) {
    if (i == 0) {
      prev = add_kernel_node(graph, nullptr, 0, options.iters);
    } else {
      hipGraphNode_t n = add_kernel_node(graph, &prev, 1, options.iters);
      prev = n;
    }
  }
}

// Helper function to build parallel topology
static void build_parallel_topology(hipGraph_t graph, const options& options) {
  for (int w = 0; w < options.width; ++w) {
    hipGraphNode_t prev{};
    for (int i = 0; i < options.length; ++i) {
      if (i == 0) {
        prev = add_kernel_node(graph, nullptr, 0, options.iters);
      } else {
        hipGraphNode_t n = add_kernel_node(graph, &prev, 1, options.iters);
        prev = n;
      }
    }
  }
}

// Helper function to build mixed topology
static PerfResults build_mixed_topology(hipGraph_t graph, const options& options, void* d_mem1,
                                        void* d_mem2, size_t mem_size) {
  PerfResults batch_info;
  batch_info.nodes_total = options.length;

  // Pattern: memset -> 3 kernels -> memcpy -> 2 kernels -> empty -> 3 kernels -> memset (repeating)
  std::vector<hipGraphNode_t> all_nodes;
  std::vector<std::string> node_types;
  hipGraphNode_t prev{};

  int kernel_count = 0;
  int current_batch_size = 0;
  std::vector<int> kernel_batches;

  for (int i = 0; i < options.length; ++i) {
    std::string node_type;
    if (i == 0) {
      // Start with memset
      prev = add_memset_node(graph, nullptr, 0, d_mem1, 0, mem_size);
      all_nodes.push_back(prev);
      node_type = "memset";
      if (current_batch_size > 0) {
        kernel_batches.push_back(current_batch_size);
        current_batch_size = 0;
      }
    } else {
      int step = i % 9;  // 9-step pattern
      if (step == 1 || step == 2 || step == 3) {
        // 3 consecutive kernels (batchable)
        hipGraphNode_t n = add_kernel_node(graph, &prev, 1, options.iters);
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
        }
        hipGraphNode_t n = add_memcpy_node(graph, &prev, 1, d_mem2, d_mem1, mem_size);
        all_nodes.push_back(n);
        prev = n;
        node_type = "memcpy";
      } else if (step == 5 || step == 6) {
        // 2 consecutive kernels (batchable)
        hipGraphNode_t n = add_kernel_node(graph, &prev, 1, options.iters);
        all_nodes.push_back(n);
        prev = n;
        node_type = "kernel";
        kernel_count++;
        current_batch_size++;
      } else if (step == 7) {
        // empty node (breaks batching)
        if (current_batch_size > 0) {
          kernel_batches.push_back(current_batch_size);
          current_batch_size = 0;
        }
        hipGraphNode_t n = add_empty_node(graph, &prev, 1);
        all_nodes.push_back(n);
        prev = n;
        node_type = "empty";
      } else if (step == 8 || step == 0) {
        // kernel nodes (can start new batch)
        hipGraphNode_t n = add_kernel_node(graph, &prev, 1, options.iters);
        all_nodes.push_back(n);
        prev = n;
        node_type = "kernel";
        kernel_count++;
        current_batch_size++;
      }
    }
    node_types.push_back(node_type);
  }

  // Add final batch if we have kernels at the end
  if (current_batch_size > 0) {
    kernel_batches.push_back(current_batch_size);
  }

  batch_info.kernel_count = kernel_count;
  batch_info.batches_count = (int)kernel_batches.size();
  batch_info.batch_sizes = kernel_batches;

  return batch_info;
}

// Performance measurement function
static PerfResults measure_graph_performance(hipGraph_t graph, const options& options) {
  PerfResults results;

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  // Instantiate
  hipGraphExec_t gexec{};
  auto t_inst_begin = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
  auto t_inst_end = std::chrono::steady_clock::now();
  results.instantiation_us =
      std::chrono::duration<double, std::micro>(t_inst_end - t_inst_begin).count();

  if (options.preupload) {
    HIP_CHECK(hipGraphUpload(gexec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Warmup
  for (int i = 0; i < options.warmup; ++i) {
    HIP_CHECK(hipGraphLaunch(gexec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  auto t_first_begin = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphLaunch(gexec, stream));
  auto t_first_end = std::chrono::steady_clock::now();
  HIP_CHECK(hipStreamSynchronize(stream));
  auto t_e2e_end = std::chrono::steady_clock::now();

  results.first_launch_cpu_us =
      std::chrono::duration<double, std::micro>(t_first_end - t_first_begin).count();
  results.first_e2e_us =
      std::chrono::duration<double, std::micro>(t_e2e_end - t_first_begin).count();

  // Repeat launches
  std::vector<double> cpu_over_us;
  cpu_over_us.reserve(options.repeats);
  double device_us_sum = 0.0;
  hipEvent_t evt_start{}, evt_stop{};
  HIP_CHECK(hipEventCreate(&evt_start));
  HIP_CHECK(hipEventCreate(&evt_stop));

  for (int r = 0; r < options.repeats; ++r) {
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

  results.repeat_cpu_avg_us = 0.0;
  for (double v : cpu_over_us) results.repeat_cpu_avg_us += v;
  results.repeat_cpu_avg_us /= std::max(1, options.repeats);

  if (!cpu_over_us.empty()) {
    auto tmp = cpu_over_us;
    std::sort(tmp.begin(), tmp.end());
    results.repeat_cpu_p50_us = tmp[tmp.size() / 2];
    size_t idx99 = (size_t)std::min((double)tmp.size() - 1, 0.99 * (tmp.size() - 1));
    results.repeat_cpu_p99_us = tmp[idx99];
  }
  results.repeat_device_avg_us = device_us_sum / std::max(1, options.repeats);

  HIP_CHECK(hipGraphExecDestroy(gexec));
  HIP_CHECK(hipStreamDestroy(stream));

  return results;
}

TEST_CASE("Straight_Topology") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  INFO("Device: " << prop.name);

  SECTION("Straight topology with default options") {
    options options;
    options.topology = "straight";
    options.width = 1;

    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));

    build_straight_topology(graph, options);

    PerfResults results = measure_graph_performance(graph, options);
    results.nodes_total = options.length;

    std::cout << "Topology: " << options.topology << " | length=" << options.length
              << " | nodes_total=" << results.nodes_total << std::endl;
    std::cout << "Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us
              << " us" << std::endl;
    std::cout << "First launch CPU: " << std::fixed << std::setprecision(3)
              << results.first_launch_cpu_us << " us" << std::endl;
    std::cout << "First end-to-end: " << std::fixed << std::setprecision(3) << results.first_e2e_us
              << " us" << std::endl;
    std::cout << "Repeat CPU avg: " << std::fixed << std::setprecision(3)
              << results.repeat_cpu_avg_us << " us" << std::endl;
    std::cout << "Repeat device avg: " << std::fixed << std::setprecision(3)
              << results.repeat_device_avg_us << " us" << std::endl;
    std::cout << '\n';
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

TEST_CASE("Parallel_Topology") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  INFO("Device: " << prop.name);

  SECTION("Parallel topology with default options") {
    options options;
    options.topology = "parallel";
    options.width = 2;

    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));

    build_parallel_topology(graph, options);

    PerfResults results = measure_graph_performance(graph, options);
    results.nodes_total = (long long)options.width * options.length;

    std::cout << "Topology: " << options.topology << " | width=" << options.width
              << " length=" << options.length << " | nodes_total=" << results.nodes_total
              << std::endl;
    std::cout << "Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us
              << " us" << std::endl;
    std::cout << "First launch CPU: " << std::fixed << std::setprecision(3)
              << results.first_launch_cpu_us << " us" << std::endl;
    std::cout << "First end-to-end: " << std::fixed << std::setprecision(3) << results.first_e2e_us
              << " us" << std::endl;
    std::cout << "Repeat CPU avg: " << std::fixed << std::setprecision(3)
              << results.repeat_cpu_avg_us << " us" << std::endl;
    std::cout << "Repeat device avg: " << std::fixed << std::setprecision(3)
              << results.repeat_device_avg_us << " us" << std::endl;
    std::cout << '\n';
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

TEST_CASE("Mixed_Topology") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  INFO("Device: " << prop.name);

  SECTION("Mixed topology with kernel batching and default options") {
    options options;
    options.topology = "mixed";

    void* d_mem1 = nullptr;
    void* d_mem2 = nullptr;
    size_t mem_size = 4096;
    HIP_CHECK(hipMalloc(&d_mem1, mem_size));
    HIP_CHECK(hipMalloc(&d_mem2, mem_size));

    hipGraph_t graph{};
    HIP_CHECK(hipGraphCreate(&graph, 0));

    PerfResults batch_info = build_mixed_topology(graph, options, d_mem1, d_mem2, mem_size);
    PerfResults results = measure_graph_performance(graph, options);

    // Combine results
    results.nodes_total = batch_info.nodes_total;
    results.kernel_count = batch_info.kernel_count;
    results.batches_count = batch_info.batches_count;
    results.batch_sizes = batch_info.batch_sizes;

    std::cout << "Topology: " << options.topology << " | length=" << options.length
              << " | nodes_total=" << results.nodes_total << std::endl;
    std::cout << "Kernel nodes: " << results.kernel_count
              << " | Non-kernel nodes: " << (results.nodes_total - results.kernel_count)
              << std::endl;
    std::cout << "Potential kernel batches: " << results.batches_count << std::endl;

    std::string batch_sizes_str = "";
    for (size_t i = 0; i < results.batch_sizes.size(); ++i) {
      batch_sizes_str += std::to_string(results.batch_sizes[i]);
      if (i < results.batch_sizes.size() - 1) batch_sizes_str += ", ";
    }
    std::cout << "Batch sizes: " << batch_sizes_str << std::endl;

    std::cout << "Instantiation: " << std::fixed << std::setprecision(3) << results.instantiation_us
              << " us" << std::endl;
    std::cout << "First launch CPU: " << std::fixed << std::setprecision(3)
              << results.first_launch_cpu_us << " us" << std::endl;
    std::cout << "First end-to-end: " << std::fixed << std::setprecision(3) << results.first_e2e_us
              << " us" << std::endl;
    std::cout << "Repeat CPU avg: " << std::fixed << std::setprecision(3)
              << results.repeat_cpu_avg_us << " us" << std::endl;
    std::cout << "Repeat device avg: " << std::fixed << std::setprecision(3)
              << results.repeat_device_avg_us << " us" << std::endl;
    std::cout << '\n';

    // Cleanup
    HIP_CHECK(hipFree(d_mem1));
    HIP_CHECK(hipFree(d_mem2));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}
