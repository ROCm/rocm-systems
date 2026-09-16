/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Reproducer for ROCM-25757: PyFR couette-flow hipGraphLaunch overhead.
 *
 * The PyFR 2D couette-flow case (40-element Navier-Stokes p2) creates exactly
 * three small HIP graphs per RK4 step.  Per-step time is dominated by
 * hipGraphLaunch (~75 us/launch on MI300X).  This test builds the three graph
 * topologies with lightweight stand-in kernels and measures launch overhead
 * in a tight loop, reproducing the bottleneck without PyFR.
 *
 * Graph topologies (from profiling):
 *
 *   graph_3 (17 nodes, 4 levels):
 *     L0: two 3-kernel chains  (gimmik_mm -> gimmik_mm -> tflux)
 *     L1: six independent kernels  (gimmik_mm variants)
 *     L2: two EMPTY nodes (fan-in)
 *     L3: three kernels  (bceflux x2, inteflux)
 *
 *   graph_5 (4 nodes):
 *     two independent 2-kernel chains  (gimmik_mm -> negdivconf)
 *
 *   graph_7 (7 nodes):
 *     L0: two chains  (gimmik_mm -> memcpy_DtoD)
 *     L1: three kernels  (bcconu x2, intconu)  depend on both L0 chains
 */

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

__global__ void pyfr_stub_kernel(uint64_t count) {
#if HT_AMD
  uint64_t t0 = wall_clock64();
  while (wall_clock64() - t0 < count) {}
#endif
#if HT_NVIDIA
  uint64_t t0 = clock64();
  while (clock64() - t0 < count) {}
#endif
}

static uint64_t us_to_ticks(uint32_t us) {
  hipDevice_t dev;
  int clock_khz = 0;
  HIP_CHECK(hipGetDevice(&dev));
#if HT_AMD
  HIPCHECK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate, dev));
#endif
#if HT_NVIDIA
  HIPCHECK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeClockRate, dev));
#endif
  return static_cast<uint64_t>(clock_khz) * 1000ULL * us / 1000000ULL;
}

static hipGraphNode_t add_kern(hipGraph_t g, hipGraphNode_t* deps, size_t ndeps,
                               uint64_t ticks) {
  hipKernelNodeParams p{};
  p.func = reinterpret_cast<void*>(pyfr_stub_kernel);
  p.gridDim = dim3(1);
  p.blockDim = dim3(64);
  p.sharedMemBytes = 0;
  void* args[] = {&ticks};
  p.kernelParams = args;
  p.extra = nullptr;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddKernelNode(&node, g, deps, ndeps, &p));
  return node;
}

static hipGraphNode_t add_mcpy(hipGraph_t g, hipGraphNode_t* deps, size_t ndeps,
                               void* dst, void* src, size_t bytes) {
  hipMemcpy3DParms p{};
  p.srcPtr = make_hipPitchedPtr(src, bytes, 1, 1);
  p.dstPtr = make_hipPitchedPtr(dst, bytes, 1, 1);
  p.extent = make_hipExtent(bytes, 1, 1);
  p.kind   = hipMemcpyDeviceToDevice;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddMemcpyNode(&node, g, deps, ndeps, &p));
  return node;
}

static hipGraphNode_t add_empty(hipGraph_t g, hipGraphNode_t* deps, size_t ndeps) {
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddEmptyNode(&node, g, deps, ndeps));
  return node;
}

// ── graph_3: 17 nodes, 4 levels ────────────────────────────────────────────
static hipGraph_t build_graph_3(uint64_t ticks) {
  hipGraph_t g{};
  HIP_CHECK(hipGraphCreate(&g, 0));

  // Level 0 — two 3-kernel chains
  hipGraphNode_t s8_0 = add_kern(g, nullptr, 0, ticks);       // gimmik_mm_12x6
  hipGraphNode_t s8_1 = add_kern(g, &s8_0,  1, ticks);        // gimmik_mm_12x9
  hipGraphNode_t s8_2 = add_kern(g, &s8_1,  1, ticks);        // tflux

  hipGraphNode_t s0_0 = add_kern(g, nullptr, 0, ticks);       // gimmik_mm_18x9
  hipGraphNode_t s0_1 = add_kern(g, &s0_0,  1, ticks);        // gimmik_mm_18x12
  hipGraphNode_t s0_2 = add_kern(g, &s0_1,  1, ticks);        // tflux

  // Level 1 — six kernels, each depends on both L0 chain tails
  hipGraphNode_t l0_tails[] = {s8_2, s0_2};
  hipGraphNode_t l1[6];
  for (int i = 0; i < 6; ++i)
    l1[i] = add_kern(g, l0_tails, 2, ticks);                  // gimmik_mm variants

  // Level 2 — two EMPTY fan-in nodes
  // empty_10 depends on l1[0..2], empty_2 depends on l1[3..5]
  hipGraphNode_t e10_deps[] = {l1[0], l1[1], l1[2]};
  hipGraphNode_t empty_10 = add_empty(g, e10_deps, 3);

  hipGraphNode_t e2_deps[] = {l1[3], l1[4], l1[5]};
  hipGraphNode_t empty_2 = add_empty(g, e2_deps, 3);

  // Level 3 — three kernels depend on both empties
  hipGraphNode_t l2_tails[] = {empty_10, empty_2};
  add_kern(g, l2_tails, 2, ticks);  // bceflux
  add_kern(g, l2_tails, 2, ticks);  // bceflux
  add_kern(g, l2_tails, 2, ticks);  // inteflux

  return g;
}

// ── graph_5: 4 nodes, two independent 2-kernel chains ──────────────────────
static hipGraph_t build_graph_5(uint64_t ticks) {
  hipGraph_t g{};
  HIP_CHECK(hipGraphCreate(&g, 0));

  hipGraphNode_t a0 = add_kern(g, nullptr, 0, ticks);  // gimmik_mm_6x9
  add_kern(g, &a0, 1, ticks);                          // negdivconf

  hipGraphNode_t b0 = add_kern(g, nullptr, 0, ticks);  // gimmik_mm_9x12
  add_kern(g, &b0, 1, ticks);                          // negdivconf

  return g;
}

// ── graph_7: 7 nodes — 2 kernel+memcpy chains → 3 kernels ─────────────────
static hipGraph_t build_graph_7(uint64_t ticks, void* d_buf1, void* d_buf2) {
  hipGraph_t g{};
  HIP_CHECK(hipGraphCreate(&g, 0));

  // Chain A: gimmik_mm_9x6 → memcpy DtoD 18432
  hipGraphNode_t a0 = add_kern(g, nullptr, 0, ticks);
  hipGraphNode_t a1 = add_mcpy(g, &a0, 1, d_buf1, d_buf2, 18432);

  // Chain B: gimmik_mm_12x9 → memcpy DtoD 24576
  hipGraphNode_t b0 = add_kern(g, nullptr, 0, ticks);
  hipGraphNode_t b1 = add_mcpy(g, &b0, 1, d_buf2, d_buf1, 24576);

  // Three kernels depend on both chain tails
  hipGraphNode_t tails[] = {a1, b1};
  add_kern(g, tails, 2, ticks);   // bcconu
  add_kern(g, tails, 2, ticks);   // bcconu
  add_kern(g, tails, 2, ticks);   // intconu

  return g;
}

// ── Benchmark driver ───────────────────────────────────────────────────────
struct PyFRGraphResults {
  double instantiate_us;
  double first_launch_us;
  double avg_launch_us;
  double p50_launch_us;
  double p99_launch_us;
  double device_avg_us;
};

static PyFRGraphResults bench_graph(hipGraph_t graph, hipStream_t stream,
                                    int warmup, int repeats) {
  PyFRGraphResults res{};

  hipGraphExec_t exec{};
  auto t0 = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  auto t1 = std::chrono::steady_clock::now();
  res.instantiate_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

  for (int i = 0; i < warmup; ++i) {
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // First launch
  auto tf0 = std::chrono::steady_clock::now();
  HIP_CHECK(hipGraphLaunch(exec, stream));
  auto tf1 = std::chrono::steady_clock::now();
  HIP_CHECK(hipStreamSynchronize(stream));
  res.first_launch_us = std::chrono::duration<double, std::micro>(tf1 - tf0).count();

  // ---- CPU-side launch overhead ----
  // Time only the hipGraphLaunch call; synchronize each iteration so queue
  // back-pressure doesn't leak into the next sample.
  std::vector<double> cpu_us(repeats);
  for (int r = 0; r < repeats; ++r) {
    auto ta = std::chrono::steady_clock::now();
    HIP_CHECK(hipGraphLaunch(exec, stream));
    auto tb = std::chrono::steady_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream));
    cpu_us[r] = std::chrono::duration<double, std::micro>(tb - ta).count();
  }

  // ---- Device-side execution time ----
  // Bracket a whole batch of back-to-back launches with a single event pair so
  // per-sample event-record/sync granularity is amortized out (it is the same
  // ~1-2 us order as a barrier packet and otherwise swamps the signal). Launches
  // serialize on the in-order launch stream, so elapsed = repeats x per-graph
  // device time.
  hipEvent_t ev0{}, ev1{};
  HIP_CHECK(hipEventCreate(&ev0));
  HIP_CHECK(hipEventCreate(&ev1));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipEventRecord(ev0, stream));
  for (int r = 0; r < repeats; ++r) {
    HIP_CHECK(hipGraphLaunch(exec, stream));
  }
  HIP_CHECK(hipEventRecord(ev1, stream));
  HIP_CHECK(hipEventSynchronize(ev1));
  float batch_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&batch_ms, ev0, ev1));
  res.device_avg_us = (static_cast<double>(batch_ms) * 1000.0) / repeats;
  HIP_CHECK(hipEventDestroy(ev0));
  HIP_CHECK(hipEventDestroy(ev1));

  double sum = 0;
  for (double v : cpu_us) sum += v;
  res.avg_launch_us = sum / repeats;

  std::sort(cpu_us.begin(), cpu_us.end());
  // Nearest-rank percentile, indexed off (repeats - 1) so the result always
  // stays within the sample range (e.g. p99 of 1000 samples is index 989, not 990).
  res.p50_launch_us = cpu_us[(repeats - 1) * 50 / 100];
  res.p99_launch_us = cpu_us[(repeats - 1) * 99 / 100];

  HIP_CHECK(hipGraphExecDestroy(exec));
  return res;
}

static void print_results(const char* name, int nodes, const PyFRGraphResults& r) {
  CONSOLE_PRINT("  %-10s (%2d nodes)  inst=%7.1f us  first=%7.1f us  "
                "avg=%7.1f us  p50=%7.1f us  p99=%7.1f us  dev_avg=%7.1f us",
                name, nodes,
                r.instantiate_us, r.first_launch_us,
                r.avg_launch_us, r.p50_launch_us, r.p99_launch_us,
                r.device_avg_us);
}

// ── PyFR couette-flow step simulation ──────────────────────────────────────
// Each RK4 step launches graph_3, graph_5, graph_7 four times (one per stage).
static void run_pyfr_couette_bench(uint32_t kernel_us, int steps, int warmup_steps) {
  uint64_t ticks = us_to_ticks(kernel_us);

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  void* d_buf1 = nullptr;
  void* d_buf2 = nullptr;
  HIP_CHECK(hipMalloc(&d_buf1, 32768));
  HIP_CHECK(hipMalloc(&d_buf2, 32768));

  hipGraph_t g3 = build_graph_3(ticks);
  hipGraph_t g5 = build_graph_5(ticks);
  hipGraph_t g7 = build_graph_7(ticks, d_buf1, d_buf2);

  hipGraphExec_t e3{}, e5{}, e7{};
  HIP_CHECK(hipGraphInstantiate(&e3, g3, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&e5, g5, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&e7, g7, nullptr, nullptr, 0));

  // Warmup
  for (int s = 0; s < warmup_steps; ++s) {
    for (int stage = 0; stage < 4; ++stage) {
      HIP_CHECK(hipGraphLaunch(e3, stream));
      HIP_CHECK(hipGraphLaunch(e5, stream));
      HIP_CHECK(hipGraphLaunch(e7, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Timed loop — mirrors PyFR's RK4 stepping
  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s) {
    for (int stage = 0; stage < 4; ++stage) {
      HIP_CHECK(hipGraphLaunch(e3, stream));
      HIP_CHECK(hipGraphLaunch(e5, stream));
      HIP_CHECK(hipGraphLaunch(e7, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  auto t1 = std::chrono::steady_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double per_step_ms = total_ms / steps;
  int launches_per_step = 4 * 3;  // 4 RK stages × 3 graphs
  double per_launch_us = (total_ms * 1000.0) / (steps * launches_per_step);

  CONSOLE_PRINT("\n=== PyFR Couette-Flow Step Simulation ===");
  CONSOLE_PRINT("Kernel duration : %u us", kernel_us);
  CONSOLE_PRINT("Steps           : %d  (RK4, 4 stages × 3 graphs = %d launches/step)",
                steps, launches_per_step);
  CONSOLE_PRINT("Total time      : %.1f ms", total_ms);
  CONSOLE_PRINT("Per-step        : %.3f ms", per_step_ms);
  CONSOLE_PRINT("Per-launch      : %.1f us  (target: < 75 us)", per_launch_us);

  HIP_CHECK(hipGraphExecDestroy(e3));
  HIP_CHECK(hipGraphExecDestroy(e5));
  HIP_CHECK(hipGraphExecDestroy(e7));
  HIP_CHECK(hipGraphDestroy(g3));
  HIP_CHECK(hipGraphDestroy(g5));
  HIP_CHECK(hipGraphDestroy(g7));
  HIP_CHECK(hipFree(d_buf1));
  HIP_CHECK(hipFree(d_buf2));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - ROCM-25757: Benchmark hipGraphLaunch overhead for each of the three
 *      PyFR couette-flow graphs individually.  Reports instantiation and
 *      launch latency (CPU-side and device-side).
 * Test source
 * ------------------------
 *    - performance/scenarios/graph/hipGraphPyFRCouette.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Performance_PyFR_Couette_IndividualGraphs) {
  const uint32_t kernel_us = 4;  // ~4 us matches PyFR kernel durations
  uint64_t ticks = us_to_ticks(kernel_us);

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  void* d_buf1 = nullptr;
  void* d_buf2 = nullptr;
  HIP_CHECK(hipMalloc(&d_buf1, 32768));
  HIP_CHECK(hipMalloc(&d_buf2, 32768));

  constexpr int warmup = 10;
  constexpr int repeats = 1000;

  CONSOLE_PRINT("\n=== PyFR Couette-Flow Graph Launch Overhead (ROCM-25757) ===");
  CONSOLE_PRINT("Kernel stub: %u us | warmup: %d | repeats: %d\n", kernel_us, warmup, repeats);

  hipGraph_t g3 = build_graph_3(ticks);
  auto r3 = bench_graph(g3, stream, warmup, repeats);
  print_results("graph_3", 17, r3);

  hipGraph_t g5 = build_graph_5(ticks);
  auto r5 = bench_graph(g5, stream, warmup, repeats);
  print_results("graph_5", 4, r5);

  hipGraph_t g7 = build_graph_7(ticks, d_buf1, d_buf2);
  auto r7 = bench_graph(g7, stream, warmup, repeats);
  print_results("graph_7", 7, r7);

  HIP_CHECK(hipGraphDestroy(g3));
  HIP_CHECK(hipGraphDestroy(g5));
  HIP_CHECK(hipGraphDestroy(g7));
  HIP_CHECK(hipFree(d_buf1));
  HIP_CHECK(hipFree(d_buf2));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - ROCM-25757: End-to-end PyFR couette-flow RK4 step simulation.
 *      Launches all three graphs in sequence, four times per step (one per
 *      RK4 stage), measuring per-step and per-launch overhead.  The 75 us
 *      per-launch target comes from profiling the real PyFR workload.
 * Test source
 * ------------------------
 *    - performance/scenarios/graph/hipGraphPyFRCouette.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Performance_PyFR_Couette_StepSimulation) {
  const int steps = isQuickLevel() ? 100 : 2000;
  run_pyfr_couette_bench(4, steps, 20);
}

/**
 * Test Description
 * ------------------------
 *    - ROCM-25757: Minimal single-iteration test for logging/tracing.
 *      Instantiates each graph, does 1 warmup launch + 1 timed launch,
 *      then destroys.  Use with AMD_LOG_LEVEL=5 or DEBUG_HIP_GRAPH_DOT_PRINT.
 * Test source
 * ------------------------
 *    - performance/scenarios/graph/hipGraphPyFRCouette.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Performance_PyFR_Couette_SingleIteration) {
  const uint32_t kernel_us = 4;
  uint64_t ticks = us_to_ticks(kernel_us);

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  void* d_buf1 = nullptr;
  void* d_buf2 = nullptr;
  HIP_CHECK(hipMalloc(&d_buf1, 32768));
  HIP_CHECK(hipMalloc(&d_buf2, 32768));

  CONSOLE_PRINT("\n=== PyFR Couette-Flow Single Iteration (for tracing) ===\n");

  hipGraph_t g3 = build_graph_3(ticks);
  auto r3 = bench_graph(g3, stream, 1, 1);
  print_results("graph_3", 17, r3);

  hipGraph_t g5 = build_graph_5(ticks);
  auto r5 = bench_graph(g5, stream, 1, 1);
  print_results("graph_5", 4, r5);

  hipGraph_t g7 = build_graph_7(ticks, d_buf1, d_buf2);
  auto r7 = bench_graph(g7, stream, 1, 1);
  print_results("graph_7", 7, r7);

  HIP_CHECK(hipGraphDestroy(g3));
  HIP_CHECK(hipGraphDestroy(g5));
  HIP_CHECK(hipGraphDestroy(g7));
  HIP_CHECK(hipFree(d_buf1));
  HIP_CHECK(hipFree(d_buf2));
  HIP_CHECK(hipStreamDestroy(stream));
}
