/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_OK(expr)                                                            \
  do {                                                                          \
    hipError_t status_ = (expr);                                                \
    if (status_ != hipSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,      \
                   #expr, hipGetErrorString(status_));                          \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

using Clock = std::chrono::steady_clock;

__global__ void increment_kernel(unsigned long long* counter) {
  atomicAdd(counter, 1ULL);
}

__global__ void increment_and_continue_kernel(
    unsigned long long* counter, unsigned long long total_work,
    hipGraphConditionalHandle handle) {
  unsigned long long completed = atomicAdd(counter, 1ULL) + 1ULL;
  hipGraphSetConditional(handle, completed < total_work ? 1U : 0U);
}

__global__ void mono_kernel(unsigned long long* counter,
                            unsigned long long count) {
  for (unsigned long long i = 0; i < count; ++i) atomicAdd(counter, 1ULL);
}

struct GraphObjects {
  hipGraph_t graph{};
  hipGraphExec_t exec{};
  hipGraphConditionalHandle handle{};
};

static GraphObjects make_plain_graph(unsigned k, unsigned n,
                                     unsigned long long* counter,
                                     hipGraphConditionalHandle handle) {
  GraphObjects out;
  HIP_OK(hipGraphCreate(&out.graph, 0));
  hipGraphNode_t previous = nullptr;
  unsigned long long total_work = static_cast<unsigned long long>(k) * n;
  for (unsigned i = 0; i < k; ++i) {
    hipKernelNodeParams params{};
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    void* increment_args[] = {&counter};
    void* final_args[] = {&counter, &total_work, &handle};
    params.func = i + 1 == k
                      ? reinterpret_cast<void*>(increment_and_continue_kernel)
                      : reinterpret_cast<void*>(increment_kernel);
    params.kernelParams = i + 1 == k ? final_args : increment_args;
    hipGraphNode_t node;
    HIP_OK(hipGraphAddKernelNode(&node, out.graph,
                                 previous ? &previous : nullptr,
                                 previous ? 1 : 0, &params));
    previous = node;
  }
  HIP_OK(hipGraphInstantiate(&out.exec, out.graph, nullptr, nullptr, 0));
  return out;
}

static GraphObjects make_cond_graph(unsigned k, unsigned n,
                                    unsigned long long* counter) {
  GraphObjects out;
  HIP_OK(hipGraphCreate(&out.graph, 0));
  hipGraphConditionalHandle handle;
  HIP_OK(hipGraphConditionalHandleCreate(&handle, out.graph, 1U, 0));
  out.handle = handle;

  hipGraph_t body = nullptr;
  hipGraphNode_t conditional;
  HIP_OK(hipGraphAddConditionalNode(&conditional, out.graph, nullptr, 0,
                                    handle, hipGraphCondTypeWhile, 1, &body));

  hipGraphNode_t previous = nullptr;
  unsigned long long total_work = static_cast<unsigned long long>(k) * n;
  for (unsigned i = 0; i < k; ++i) {
    hipKernelNodeParams params{};
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    void* increment_args[] = {&counter};
    void* final_args[] = {&counter, &total_work, &handle};
    params.func = i + 1 == k
                      ? reinterpret_cast<void*>(increment_and_continue_kernel)
                      : reinterpret_cast<void*>(increment_kernel);
    params.kernelParams = i + 1 == k ? final_args : increment_args;
    hipGraphNode_t node;
    HIP_OK(hipGraphAddKernelNode(&node, body,
                                 previous ? &previous : nullptr,
                                 previous ? 1 : 0, &params));
    previous = node;
  }
  HIP_OK(hipGraphInstantiate(&out.exec, out.graph, nullptr, nullptr, 0));
  return out;
}

static void destroy_graph(GraphObjects& graph) {
  HIP_OK(hipGraphExecDestroy(graph.exec));
  HIP_OK(hipGraphDestroy(graph.graph));
}

template <typename Submit>
static double measure(Submit submit, hipStream_t stream,
                      unsigned long long* counter,
                      unsigned long long expected,
                      hsa_signal_t condition = {},
                      bool submit_synchronizes = false) {
  constexpr int warmups = 3;
  constexpr int repetitions = 20;
  double total_ms = 0.0;
  for (int rep = -warmups; rep < repetitions; ++rep) {
    HIP_OK(hipMemsetAsync(counter, 0, sizeof(*counter), stream));
    HIP_OK(hipStreamSynchronize(stream));
    if (condition.handle != 0) {
      hsa_signal_store_screlease(condition, 1);
    }
    auto start = Clock::now();
    submit();
    if (!submit_synchronizes) {
      HIP_OK(hipStreamSynchronize(stream));
    }
    auto stop = Clock::now();

    unsigned long long result = 0;
    HIP_OK(hipMemcpy(&result, counter, sizeof(result), hipMemcpyDeviceToHost));
    if (result != expected) {
      std::fprintf(stderr, "incorrect result: got %llu, expected %llu\n",
                   result, expected);
      std::exit(2);
    }
    if (condition.handle != 0 && hsa_signal_load_scacquire(condition) != 0) {
      std::fprintf(stderr, "condition did not become false\n");
      std::exit(3);
    }
    if (rep >= 0) {
      total_ms += std::chrono::duration<double, std::milli>(stop - start).count();
    }
  }
  return total_ms / repetitions;
}

int main() {
  hipDeviceProp_t props{};
  HIP_OK(hipGetDeviceProperties(&props, 0));
  std::printf("device,%s\n", props.name);
  std::printf(
      "K,N,host_fixed_ms,host_cond_ms,graph_fixed_ms,graph_host_cond_ms,"
      "cond_while_ms,mono_kernel_ms\n");
  std::fflush(stdout);

  hipStream_t stream;
  HIP_OK(hipStreamCreate(&stream));
  unsigned long long* counter;
  HIP_OK(hipMalloc(&counter, sizeof(*counter)));

  const std::vector<unsigned> ks{1, 10, 100, 1000};
  const std::vector<unsigned> ns{1, 10, 100};
  for (unsigned k : ks) {
    for (unsigned n : ns) {
      if (static_cast<unsigned long long>(k) * n > 10000) continue;
      const unsigned long long work = static_cast<unsigned long long>(k) * n;
      GraphObjects conditional = make_cond_graph(k, n, counter);
      GraphObjects plain = make_plain_graph(k, n, counter, conditional.handle);
      hsa_signal_t condition_signal{};
      condition_signal.handle = conditional.handle.signal_handle;

      auto launch_body = [&] {
        for (unsigned i = 0; i < k; ++i) {
          if (i + 1 == k) {
            hipLaunchKernelGGL(increment_and_continue_kernel, dim3(1), dim3(1),
                               0, stream, counter, work, conditional.handle);
          } else {
            hipLaunchKernelGGL(increment_kernel, dim3(1), dim3(1), 0, stream,
                               counter);
          }
        }
      };

      double host_fixed = measure([&] {
        for (unsigned j = 0; j < n; ++j) launch_body();
      }, stream, counter, work, condition_signal);

      double host_cond = measure([&] {
        unsigned iterations = 0;
        do {
          launch_body();
          HIP_OK(hipStreamSynchronize(stream));
          ++iterations;
        } while (hsa_signal_load_scacquire(condition_signal) != 0);
        if (iterations != n) {
          std::fprintf(stderr, "host_cond ran %u iterations, expected %u\n",
                       iterations, n);
          std::exit(4);
        }
      }, stream, counter, work, condition_signal, true);

      double graph_fixed = measure([&] {
        for (unsigned j = 0; j < n; ++j)
          HIP_OK(hipGraphLaunch(plain.exec, stream));
      }, stream, counter, work, condition_signal);

      double graph_host_cond = measure([&] {
        unsigned iterations = 0;
        do {
          HIP_OK(hipGraphLaunch(plain.exec, stream));
          HIP_OK(hipStreamSynchronize(stream));
          ++iterations;
        } while (hsa_signal_load_scacquire(condition_signal) != 0);
        if (iterations != n) {
          std::fprintf(stderr,
                       "graph_host_cond ran %u iterations, expected %u\n",
                       iterations, n);
          std::exit(5);
        }
      }, stream, counter, work, condition_signal, true);

      double cond = measure([&] {
        HIP_OK(hipGraphLaunch(conditional.exec, stream));
      }, stream, counter, work, condition_signal);

      double mono = measure([&] {
        hipLaunchKernelGGL(mono_kernel, dim3(1), dim3(1), 0, stream, counter,
                           work);
      }, stream, counter, work);

      std::printf("%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", k, n,
                  host_fixed, host_cond, graph_fixed, graph_host_cond, cond,
                  mono);
      std::fflush(stdout);
      destroy_graph(plain);
      destroy_graph(conditional);
    }
  }

  HIP_OK(hipFree(counter));
  HIP_OK(hipStreamDestroy(stream));
  return 0;
}
