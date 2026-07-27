/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Compares thread_block::sync() against barrier_arrive()/barrier_wait() over
// the same imbalanced workload, where the split version overlaps independent
// work with the barrier bubble, and reports the speedup. Archs without hardware
// split barriers should stay near 1x rather than regress.

#include <hip/hip_cooperative_groups.h>
#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

// compute_a is imbalanced and consumed across the barrier; compute_c is uniform
// and independent, so it can run in the arrive/wait gap.
static __device__ __forceinline__ float compute_a(float v, int iters) {
  for (int k = 0; k < iters; ++k) v = v * 1.0000001f + 1.0f;
  return v;
}

static __device__ __forceinline__ float compute_c(float v, int iters) {
  for (int k = 0; k < iters; ++k) v = v * 0.9999999f + 0.5f;
  return v;
}

static __global__ void imbalanced_full_barrier(float* out, const float* in,
                                               int heavy_iters, int light_iters,
                                               int indep_iters) {
  extern __shared__ float sh_full[];
  auto tb = cooperative_groups::this_thread_block();
  size_t i = threadIdx.x;

  const int iters = (i == 0) ? heavy_iters : light_iters;
  sh_full[i] = compute_a(in[i], iters);

  tb.sync();

  float c = compute_c(in[i], indep_iters);
  out[i] = sh_full[(i + 1) % blockDim.x] + c;
}

static __global__ void imbalanced_split_barrier(float* out, const float* in,
                                                int heavy_iters,
                                                int light_iters,
                                                int indep_iters) {
  extern __shared__ float sh_split[];
  auto tb = cooperative_groups::this_thread_block();
  size_t i = threadIdx.x;

  const int iters = (i == 0) ? heavy_iters : light_iters;
  sh_split[i] = compute_a(in[i], iters);  // publish before arrive

  auto tok = tb.barrier_arrive();
  float c = compute_c(in[i], indep_iters);  // independent work in the gap
  tb.barrier_wait(std::move(tok));

  out[i] = sh_split[(i + 1) % blockDim.x] + c;
}

struct KernelParams {
  int heavy;
  int light;
  int indep;
};

class FullBarrierBenchmark : public Benchmark<FullBarrierBenchmark> {
 public:
  void operator()(float* out, const float* in, unsigned size, KernelParams p) {
    TIMED_SECTION(kTimerTypeEvent) {
      imbalanced_full_barrier<<<1, size, sizeof(float) * size>>>(
          out, in, p.heavy, p.light, p.indep);
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipGetLastError());
  }
};

class SplitBarrierBenchmark : public Benchmark<SplitBarrierBenchmark> {
 public:
  void operator()(float* out, const float* in, unsigned size, KernelParams p) {
    TIMED_SECTION(kTimerTypeEvent) {
      imbalanced_split_barrier<<<1, size, sizeof(float) * size>>>(
          out, in, p.heavy, p.light, p.indep);
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipGetLastError());
  }
};

HIP_TEST_CASE(Performance_SplitBarrier_ImbalancedWorkgroup) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::cout << "[split_barrier][perf] device gcnArchName: " << prop.gcnArchName
            << std::endl;

  const unsigned size =
      std::min(1024u, static_cast<unsigned>(prop.maxThreadsPerBlock));
  const KernelParams params{/*heavy=*/16000, /*light=*/1000, /*indep=*/4000};

  LinearAllocGuard<float> d_in(LinearAllocs::hipMalloc, sizeof(float) * size);
  LinearAllocGuard<float> d_out_full(LinearAllocs::hipMalloc,
                                     sizeof(float) * size);
  LinearAllocGuard<float> d_out_split(LinearAllocs::hipMalloc,
                                      sizeof(float) * size);

  std::vector<float> in(size), out_full(size), out_split(size);
  for (unsigned i = 0; i < size; i++) in[i] = 1.0f + static_cast<float>(i);
  HIP_CHECK(hipMemcpy(d_in.ptr(), in.data(), sizeof(float) * size,
                      hipMemcpyHostToDevice));

  FullBarrierBenchmark full;
  const auto full_stats = full.Run(d_out_full.ptr(), d_in.ptr(), size, params);

  SplitBarrierBenchmark split;
  const auto split_stats =
      split.Run(d_out_split.ptr(), d_in.ptr(), size, params);

  // Both kernels are mathematically identical.
  HIP_CHECK(hipMemcpy(out_full.data(), d_out_full.ptr(), sizeof(float) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_split.data(), d_out_split.ptr(), sizeof(float) * size,
                      hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < size; i++) {
    INFO("idx " << i);
    REQUIRE(out_split[i] == Catch::Approx(out_full[i]));
  }

  const float full_best = std::get<2>(full_stats);
  const float split_best = std::get<2>(split_stats);
  const float speedup = (split_best > 0.0f) ? (full_best / split_best) : 0.0f;

  std::cout << "[split_barrier][perf] block=" << size
            << " full-barrier best=" << full_best << " ms, split-barrier best="
            << split_best << " ms, speedup=" << speedup << "x" << std::endl;
}
