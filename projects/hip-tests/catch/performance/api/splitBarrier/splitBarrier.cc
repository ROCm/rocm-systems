/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Measures how much of a barrier bubble barrier_arrive()/barrier_wait() can
// recover versus thread_block::sync(). Gap work is issued only by the threads
// that arrive early, so it overlaps the straggler instead of extending the
// critical path. Scenarios sweep the ratio of bubble to available gap work.
//
// Three kernels run per scenario, because the gap work is by construction
// independent of the barrier and so a full-barrier kernel can also issue it
// before arriving:
//
//   full     sync() then gap work        - the naive placement
//   hoisted  gap work then sync()        - the fair sync()-only baseline
//   split    arrive(), gap work, wait()
//
// vs_naive is what the split barrier saves over the naive placement, vs_hoisted
// is what it saves over the best a plain barrier can do. The second is the one
// that says whether arrive/wait bought anything.

#include <hip/hip_cooperative_groups.h>
#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// compute_a feeds shared memory that is consumed across the barrier, so it has
// to finish before arriving. compute_c depends on nothing shared.
static __device__ __forceinline__ float compute_a(float v, int iters) {
  for (int k = 0; k < iters; ++k) v = v * 1.0000001f + 1.0f;
  return v;
}

static __device__ __forceinline__ float compute_c(float v, int iters) {
  for (int k = 0; k < iters; ++k) v = v * 0.9999999f + 0.5f;
  return v;
}

// Stragglers are whole waves: a slow lane already stalls its wave, and keeping
// the predicate wave-uniform avoids divergence inside the gap work.
static __global__ void imbalanced_full_barrier(float* out, const float* in,
                                               int heavy, int light, int indep,
                                               unsigned slow_waves) {
  extern __shared__ float sh_full[];
  auto tb = cooperative_groups::this_thread_block();
  const size_t i = threadIdx.x;
  const bool slow =
      (threadIdx.x / static_cast<unsigned>(warpSize)) < slow_waves;

  sh_full[i] = compute_a(in[i], slow ? heavy : light);

  tb.sync();

  const float c = slow ? 0.0f : compute_c(in[i], indep);
  out[i] = sh_full[(i + 1) % blockDim.x] + c;
}

// Same barrier, but the gap work is issued before arriving instead of after
// being released. Nothing here reads what the barrier protects, so this is
// legal, and it recovers the same bubble without a split barrier. This is the
// baseline the split version has to beat.
static __global__ void imbalanced_hoisted_barrier(float* out, const float* in,
                                                  int heavy, int light,
                                                  int indep,
                                                  unsigned slow_waves) {
  extern __shared__ float sh_hoist[];
  auto tb = cooperative_groups::this_thread_block();
  const size_t i = threadIdx.x;
  const bool slow =
      (threadIdx.x / static_cast<unsigned>(warpSize)) < slow_waves;

  sh_hoist[i] = compute_a(in[i], slow ? heavy : light);
  const float c = slow ? 0.0f : compute_c(in[i], indep);

  tb.sync();

  out[i] = sh_hoist[(i + 1) % blockDim.x] + c;
}

static __global__ void imbalanced_split_barrier(float* out, const float* in,
                                                int heavy, int light, int indep,
                                                unsigned slow_waves) {
  extern __shared__ float sh_split[];
  auto tb = cooperative_groups::this_thread_block();
  const size_t i = threadIdx.x;
  const bool slow =
      (threadIdx.x / static_cast<unsigned>(warpSize)) < slow_waves;

  sh_split[i] = compute_a(in[i], slow ? heavy : light);  // publish before arrive

  auto tok = tb.barrier_arrive();
  // Only the early arrivers have work here, so it fills the bubble rather than
  // adding to the straggler's critical path.
  const float c = slow ? 0.0f : compute_c(in[i], indep);
  tb.barrier_wait(std::move(tok));

  out[i] = sh_split[(i + 1) % blockDim.x] + c;
}

struct Scenario {
  const char* name;
  int heavy;
  int light;
  int indep;
  unsigned slow_waves;  // 0 selects half the block
};

class FullBarrierBenchmark : public Benchmark<FullBarrierBenchmark> {
 public:
  void operator()(float* out, const float* in, unsigned size, int heavy,
                  int light, int indep, unsigned slow_waves) {
    TIMED_SECTION(kTimerTypeEvent) {
      imbalanced_full_barrier<<<1, size, sizeof(float) * size>>>(
          out, in, heavy, light, indep, slow_waves);
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipGetLastError());
  }
};

class HoistedBarrierBenchmark : public Benchmark<HoistedBarrierBenchmark> {
 public:
  void operator()(float* out, const float* in, unsigned size, int heavy,
                  int light, int indep, unsigned slow_waves) {
    TIMED_SECTION(kTimerTypeEvent) {
      imbalanced_hoisted_barrier<<<1, size, sizeof(float) * size>>>(
          out, in, heavy, light, indep, slow_waves);
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipGetLastError());
  }
};

class SplitBarrierBenchmark : public Benchmark<SplitBarrierBenchmark> {
 public:
  void operator()(float* out, const float* in, unsigned size, int heavy,
                  int light, int indep, unsigned slow_waves) {
    TIMED_SECTION(kTimerTypeEvent) {
      imbalanced_split_barrier<<<1, size, sizeof(float) * size>>>(
          out, in, heavy, light, indep, slow_waves);
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipGetLastError());
  }
};

// In compute_a/compute_c iteration units the block cannot finish before the
// straggler, and the split version only saves the gap work that fits inside the
// bubble; anything beyond it spills past the wait.
//
// This models vs_naive only, and only bubble recovery. It is not a hard bound:
// freeing the waiting waves earlier also lets them fill otherwise idle issue
// slots, so a measured speedup can legitimately come out above it. The same
// arithmetic gives max(heavy, light + indep) for the hoisted baseline, i.e. the
// model for vs_hoisted is 1.0.
static double BubbleRecoveryModel(int heavy, int light, int indep) {
  const double bubble = static_cast<double>(heavy) - light;
  const double spill = std::max(0.0, static_cast<double>(indep) - bubble);
  const double full = static_cast<double>(heavy) + indep;
  const double split = static_cast<double>(heavy) + spill;
  return (split > 0.0) ? (full / split) : 1.0;
}

HIP_TEST_CASE(Performance_SplitBarrier_ImbalancedWorkgroup) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const unsigned wave = static_cast<unsigned>(prop.warpSize);
  const unsigned size =
      std::min(1024u, static_cast<unsigned>(prop.maxThreadsPerBlock));
  const unsigned waves = std::max(1u, size / wave);

  std::cout << "[split_barrier][perf] arch=" << prop.gcnArchName
            << " block=" << size << " wave=" << wave << " waves=" << waves
            << std::endl;

  const Scenario scenarios[] = {
      {"balanced", 8000, 8000, 4000, 1},
      {"gap_matches_bubble", 16000, 8000, 8000, 1},
      {"gap_limited", 16000, 2000, 2000, 1},
      {"bubble_limited", 10000, 8000, 16000, 1},
      {"no_gap_work", 16000, 2000, 0, 1},
      {"half_block_slow", 16000, 4000, 12000, 0},
  };

  LinearAllocGuard<float> d_in(LinearAllocs::hipMalloc, sizeof(float) * size);
  LinearAllocGuard<float> d_out_full(LinearAllocs::hipMalloc,
                                     sizeof(float) * size);
  LinearAllocGuard<float> d_out_hoist(LinearAllocs::hipMalloc,
                                      sizeof(float) * size);
  LinearAllocGuard<float> d_out_split(LinearAllocs::hipMalloc,
                                      sizeof(float) * size);

  std::vector<float> in(size), out_full(size), out_hoist(size), out_split(size);
  for (unsigned i = 0; i < size; i++) in[i] = 1.0f + static_cast<float>(i);
  HIP_CHECK(hipMemcpy(d_in.ptr(), in.data(), sizeof(float) * size,
                      hipMemcpyHostToDevice));

  std::vector<std::string> summary;

  for (const Scenario& s : scenarios) {
    const unsigned slow_waves =
        (s.slow_waves == 0) ? std::max(1u, waves / 2) : s.slow_waves;

    FullBarrierBenchmark full;
    full.AddSectionName(s.name);
    full.SetDisplayOutput(false);
    const auto full_stats = full.Run(d_out_full.ptr(), d_in.ptr(), size,
                                     s.heavy, s.light, s.indep, slow_waves);

    HoistedBarrierBenchmark hoist;
    hoist.AddSectionName(s.name);
    hoist.SetDisplayOutput(false);
    const auto hoist_stats = hoist.Run(d_out_hoist.ptr(), d_in.ptr(), size,
                                       s.heavy, s.light, s.indep, slow_waves);

    SplitBarrierBenchmark split;
    split.AddSectionName(s.name);
    split.SetDisplayOutput(false);
    const auto split_stats = split.Run(d_out_split.ptr(), d_in.ptr(), size,
                                       s.heavy, s.light, s.indep, slow_waves);

    // The three kernels are mathematically identical.
    HIP_CHECK(hipMemcpy(out_full.data(), d_out_full.ptr(), sizeof(float) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(out_hoist.data(), d_out_hoist.ptr(),
                        sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(out_split.data(), d_out_split.ptr(),
                        sizeof(float) * size, hipMemcpyDeviceToHost));
    for (unsigned i = 0; i < size; i++) {
      INFO("scenario " << s.name << " idx " << i);
      REQUIRE(out_hoist[i] == Catch::Approx(out_full[i]));
      REQUIRE(out_split[i] == Catch::Approx(out_full[i]));
    }

    const float full_best = std::get<2>(full_stats);
    const float hoist_best = std::get<2>(hoist_stats);
    const float split_best = std::get<2>(split_stats);
    const double measured =
        (split_best > 0.0f) ? (full_best / split_best) : 0.0;
    const double vs_hoisted =
        (split_best > 0.0f) ? (hoist_best / split_best) : 0.0;
    const double model = BubbleRecoveryModel(s.heavy, s.light, s.indep);

    std::ostringstream line;
    line << std::left << std::setw(19) << s.name << std::right
         << "  heavy=" << std::setw(5) << s.heavy << " light=" << std::setw(5)
         << s.light << " indep=" << std::setw(5) << s.indep
         << " slow_waves=" << std::setw(2) << slow_waves << "  |  full="
         << std::fixed << std::setprecision(4) << std::setw(8) << full_best
         << "ms hoisted=" << std::setw(8) << hoist_best << "ms split="
         << std::setw(8) << split_best << "ms  |  vs_naive="
         << std::setprecision(3) << std::setw(5) << measured
         << "x model=" << std::setw(5) << model << "x vs_hoisted="
         << std::setw(5) << vs_hoisted << "x";
    if (model > 1.0) {
      const double recovered = 100.0 * (measured - 1.0) / (model - 1.0);
      if (measured > model) {
        line << " above model (+" << std::setprecision(1)
             << 100.0 * (measured - model) / model << "%)";
      } else {
        line << " recovered=" << std::setprecision(1) << std::setw(6)
             << recovered << "%";
      }
    }
    summary.push_back(line.str());

    // Performance is reported, not gated; only flag an outright regression.
    INFO("scenario " << s.name << " vs_naive " << measured << "x vs_hoisted "
                     << vs_hoisted << "x");
    CHECK(measured > 0.9);
    CHECK(vs_hoisted > 0.9);
  }

  std::cout << "[split_barrier][perf] ---- headroom summary ----" << std::endl;
  for (const std::string& line : summary) {
    std::cout << "[split_barrier][perf] " << line << std::endl;
  }
  std::cout << "[split_barrier][perf] vs_naive = versus sync() with the gap "
               "work left after the barrier; model covers bubble recovery only, "
               "so measured may sit above it. vs_hoisted = versus the same gap "
               "work issued before sync(), which is what a plain barrier can "
               "already achieve; ~1.0x there means arrive/wait costs nothing "
               "but buys nothing on this workload."
            << std::endl;
}
