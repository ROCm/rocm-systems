/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipPerfSharedQueueAnyOrderOverlap hipPerfSharedQueueAnyOrderOverlap
 * @{
 * @ingroup PerformanceTestStream
 * Characterizes the any-order dispatch overlap optimization
 * (DEBUG_CLR_AQL_BARRIER_OPT) for plain HIP streams.
 *
 * When more streams are created than the HW queue pool (GPU_MAX_HW_QUEUES,
 * default 4), the extra streams share HW queues. With the flag OFF the first
 * kernel of each oversubscribed stream carries a head barrier and serializes
 * behind the prior tenant on the ring; with the flag ON that barrier is cleared
 * so independent streams overlap.
 *
 * This test sweeps the stream count and reports per-iteration launch time. Run it
 * twice to see the effect (the flag is read once at process init):
 *   DEBUG_CLR_AQL_BARRIER_OPT=0 <exe> "Performance_hipPerfSharedQueueAnyOrderOverlap"
 *   DEBUG_CLR_AQL_BARRIER_OPT=1 <exe> "Performance_hipPerfSharedQueueAnyOrderOverlap"
 * At/below the pool size the two must match (no regression); above it the ON run
 * should stay roughly flat while the OFF run grows with the stream count.
 */

#include <hip_test_common.hh>

#include <vector>

namespace {

__global__ void BusyKernel(int* out, int slot, int busy_iters) {
  float acc = 0.0f;
  for (int i = 0; i < busy_iters; ++i) {
    acc += __sinf(static_cast<float>(i) * 0.001f);
  }
  out[slot] += (acc > 3.0e38f) ? 2 : 1;  // branch never taken; keeps acc (and the loop) live
}

__global__ void BusyRmwKernel(int* buf, int busy_iters) {
  float acc = 0.0f;
  for (int i = 0; i < busy_iters; ++i) acc += __sinf(static_cast<float>(i) * 0.001f);
  buf[0] += (acc > 3.0e38f) ? 2 : 1;
}

double TimeEventDependencyPairs(int num_pairs, int busy_iters, int iters) {
  std::vector<hipStream_t> prod(num_pairs), cons(num_pairs);
  std::vector<hipEvent_t> ev(num_pairs);
  std::vector<int*> bp(num_pairs), bc(num_pairs);
  for (int p = 0; p < num_pairs; ++p) {
    HIP_CHECK(hipStreamCreate(&prod[p]));
    HIP_CHECK(hipStreamCreate(&cons[p]));
    HIP_CHECK(hipEventCreateWithFlags(&ev[p], hipEventDisableTiming));
    HIP_CHECK(hipMalloc(&bp[p], sizeof(int)));
    HIP_CHECK(hipMalloc(&bc[p], sizeof(int)));
    HIP_CHECK(hipMemset(bp[p], 0, sizeof(int)));
    HIP_CHECK(hipMemset(bc[p], 0, sizeof(int)));
  }

  auto launch_all = [&]() {
    for (int p = 0; p < num_pairs; ++p) {
      BusyRmwKernel<<<dim3(1), dim3(1), 0, prod[p]>>>(bp[p], busy_iters);
      HIP_CHECK(hipEventRecord(ev[p], prod[p]));
      HIP_CHECK(hipStreamWaitEvent(cons[p], ev[p], 0));
      BusyRmwKernel<<<dim3(1), dim3(1), 0, cons[p]>>>(bc[p], busy_iters);
    }
    HIP_CHECK(hipDeviceSynchronize());
  };

  for (int w = 0; w < 5; ++w) launch_all();  // warm-up + reach steady oversubscription

  hipEvent_t beg, end;
  HIP_CHECK(hipEventCreate(&beg));
  HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipEventRecord(beg));
  HIP_CHECK(hipEventSynchronize(beg));  // ensure start timestamp is taken before timed work
  for (int it = 0; it < iters; ++it) launch_all();
  HIP_CHECK(hipEventRecord(end));
  HIP_CHECK(hipEventSynchronize(end));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, beg, end));

  HIP_CHECK(hipEventDestroy(beg));
  HIP_CHECK(hipEventDestroy(end));
  for (int p = 0; p < num_pairs; ++p) {
    HIP_CHECK(hipFree(bp[p]));
    HIP_CHECK(hipFree(bc[p]));
    HIP_CHECK(hipEventDestroy(ev[p]));
    HIP_CHECK(hipStreamDestroy(prod[p]));
    HIP_CHECK(hipStreamDestroy(cons[p]));
  }
  return (static_cast<double>(ms) * 1000.0) / iters;  // microseconds per iteration
}

double TimeStreamSweep(int num_streams, int busy_iters, int iters) {
  std::vector<hipStream_t> streams(num_streams);
  for (auto& s : streams) HIP_CHECK(hipStreamCreate(&s));

  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, num_streams * sizeof(int)));
  HIP_CHECK(hipMemset(d_out, 0, num_streams * sizeof(int)));

  auto launch_all = [&]() {
    for (int k = 0; k < num_streams; ++k) {
      BusyKernel<<<dim3(1), dim3(1), 0, streams[k]>>>(d_out, k, busy_iters);
    }
    for (auto& s : streams) HIP_CHECK(hipStreamSynchronize(s));
  };

  for (int w = 0; w < 5; ++w) launch_all();  // warm-up + reach steady oversubscription

  hipEvent_t beg, end;
  HIP_CHECK(hipEventCreate(&beg));
  HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipEventRecord(beg));
  HIP_CHECK(hipEventSynchronize(beg));  // ensure start timestamp is taken before timed work
  for (int it = 0; it < iters; ++it) launch_all();
  HIP_CHECK(hipEventRecord(end));
  HIP_CHECK(hipEventSynchronize(end));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, beg, end));

  HIP_CHECK(hipEventDestroy(beg));
  HIP_CHECK(hipEventDestroy(end));
  HIP_CHECK(hipFree(d_out));
  for (auto& s : streams) HIP_CHECK(hipStreamDestroy(s));
  return (static_cast<double>(ms) * 1000.0) / iters;  // microseconds per iteration
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Sweep the stream count from below to well above the HW queue pool size and
 *    report per-iteration multi-stream launch time, characterizing the overlap
 *    optimization's effect under oversubscription.
 * Test source
 * ------------------------
 *  - performance/scenarios/stream/hipPerfSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Performance_hipPerfSharedQueueAnyOrderOverlap") {
  constexpr int kBusyIters = 20000;
  constexpr int kIters = 50;
  const int stream_counts[] = {2, 4, 8, 16, 32};

  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  CONSOLE_PRINT("device=%s busy_iters=%d iters=%d", props.gcnArchName, kBusyIters, kIters);

  for (int n : stream_counts) {
    const double us_per_iter = TimeStreamSweep(n, kBusyIters, kIters);
    CONSOLE_PRINT("streams=%d us_per_iter=%.3f", n, us_per_iter);
    REQUIRE(us_per_iter > 0.0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Characterizes overlap when streams carry explicit cross-stream dependencies.
 *    Each pair is producer -> hipEventRecord -> hipStreamWaitEvent -> consumer, so a
 *    consumer waits its own producer, but producers across pairs are independent and
 *    can overlap on the shared ring. Sweeps the pair count and reports per-iteration
 *    time. Run twice (flag OFF then ON): correctness of the waits is unchanged, and
 *    the ON run should improve as independent producers overlap under oversubscription.
 * Test source
 * ------------------------
 *  - performance/scenarios/stream/hipPerfSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Performance_hipPerfSharedQueueAnyOrderOverlap_EventDependency") {
  constexpr int kBusyIters = 20000;
  constexpr int kIters = 50;
  const int pair_counts[] = {2, 4, 8};

  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  CONSOLE_PRINT("device=%s busy_iters=%d iters=%d", props.gcnArchName, kBusyIters, kIters);

  for (int n : pair_counts) {
    const double us_per_iter = TimeEventDependencyPairs(n, kBusyIters, kIters);
    CONSOLE_PRINT("pairs=%d us_per_iter=%.3f", n, us_per_iter);
    REQUIRE(us_per_iter > 0.0);
  }
}

/**
 * End doxygen group PerformanceTestStream.
 * @}
 */
