/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipPerfSharedQueueAnyOrderOverlap hipPerfSharedQueueAnyOrderOverlap
 * @{
 * @ingroup PerformanceTestStream
 * Characterizes the shared-queue any-order overlap optimization
 * (DEBUG_HIP_SHARED_QUEUE_ANYORDER) for plain HIP streams.
 *
 * When more streams are created than the HW queue pool (GPU_MAX_HW_QUEUES,
 * default 4), the extra streams share HW queues. With the flag OFF the first
 * kernel of each oversubscribed stream carries a head barrier and serializes
 * behind the prior tenant on the ring; with the flag ON that barrier is cleared
 * so independent streams overlap.
 *
 * This test sweeps the stream count and reports per-iteration launch time. Run it
 * twice to see the effect (the flag is read once at process init):
 *   DEBUG_HIP_SHARED_QUEUE_ANYORDER=0 <exe> "Performance_hipPerfSharedQueueAnyOrderOverlap"
 *   DEBUG_HIP_SHARED_QUEUE_ANYORDER=1 <exe> "Performance_hipPerfSharedQueueAnyOrderOverlap"
 * At/below the pool size the two must match (no regression); above it the ON run
 * should stay roughly flat while the OFF run grows with the stream count.
 */

#include <hip_test_common.hh>

#include <vector>

namespace {

__global__ void BusyKernel(int* out, int slot, int busy_iters) {
  volatile float acc = 0.0f;
  for (int i = 0; i < busy_iters; ++i) {
    acc += __sinf(static_cast<float>(i) * 0.001f);
  }
  if (acc == 123456.789f) {  // never true; defeats dead-code elimination
    out[slot] += 1;
  }
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
  constexpr int kIters = 100;
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
 * End doxygen group PerformanceTestStream.
 * @}
 */
