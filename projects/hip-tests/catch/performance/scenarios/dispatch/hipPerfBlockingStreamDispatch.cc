/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipPerfBlockingStreamDispatch hipPerfBlockingStreamDispatch
 * @{
 * @ingroup PerformanceTestDispatch
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <chrono>
#include <vector>

/**
 * Test Description
 * ------------------------
 *  - Measures the time per launch of a short timing kernel launched back to back on a
 *    blocking stream, after a synchronous hipMemcpy on the null stream, and on a
 *    non-blocking stream. Each launch to a blocking stream first checks whether the null
 *    stream's last command has finished, so the difference between the two streams is the
 *    cost of that check. Run it with DEBUG_CLR_DEV_SIGNALS=0 and 1 to compare signals in
 *    system and in device memory.
 *  - Timing follows hipblaslt-bench --adaptive with its host timer: each sample is a
 *    hipStreamSynchronize, a batch of launches sized to take about 1 ms, and another
 *    hipStreamSynchronize. The reported time is the median per launch over the samples.
 * Test source
 * ------------------------
 * - performance/scenarios/dispatch/hipPerfBlockingStreamDispatch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kWarmupUs = 50000.0;
constexpr double kSampleUs = 1000.0;
constexpr double kMeasureUs = 500000.0;
constexpr int kMaxWarmupChunk = 4096;
constexpr uint64_t kKernelDurationsUs[] = {0, 1, 2, 4};

__global__ void BlockingStreamTimingKernel(uint64_t duration_ticks) {
  const uint64_t start_ticks = wall_clock64();
  while (wall_clock64() - start_ticks < duration_ticks) {
  }
}

struct DispatchTiming {
  double median_us;
  double enqueue_us;
  int batch;
};

double ElapsedUs(Clock::time_point start, Clock::time_point stop) {
  return std::chrono::duration<double, std::micro>(stop - start).count();
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

DispatchTiming MeasureDispatch(hipStream_t stream, uint64_t duration_ticks) {
  auto launch = [&]() { BlockingStreamTimingKernel<<<1, 1, 0, stream>>>(duration_ticks); };

  launch();
  HIP_CHECK(hipStreamSynchronize(stream));
  double warmup_us = 0.0;
  double per_launch_us = 0.0;
  for (int chunk = 1; warmup_us < kWarmupUs; chunk = std::min(chunk * 2, kMaxWarmupChunk)) {
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto start = Clock::now();
    for (int i = 0; i < chunk; ++i) launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    const double chunk_us = ElapsedUs(start, Clock::now());
    warmup_us += chunk_us;
    per_launch_us = chunk_us / chunk;
  }

  const int batch = std::max(1, static_cast<int>(kSampleUs / per_launch_us));
  std::vector<double> sample_us;
  std::vector<double> sample_enqueue_us;
  double measured_us = 0.0;
  while (measured_us < kMeasureUs) {
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto start = Clock::now();
    for (int i = 0; i < batch; ++i) launch();
    const auto enqueued = Clock::now();
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto stop = Clock::now();
    sample_us.push_back(ElapsedUs(start, stop) / batch);
    sample_enqueue_us.push_back(ElapsedUs(start, enqueued) / batch);
    measured_us += ElapsedUs(start, stop);
  }
  HIP_CHECK(hipGetLastError());
  return {Median(sample_us), Median(sample_enqueue_us), batch};
}

}  // namespace

HIP_TEST_CASE(Performance_hipPerfBlockingStreamDispatch) {
  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));
  int wall_clock_khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, device_id));
  if (wall_clock_khz == 0) {
    HIP_SKIP_TEST("The timing kernel needs the device wall clock");
  }

  // A synchronous copy waits for its command but, unlike a stream or device synchronize,
  // leaves it as the null stream's last command, which later launches then check.
  const int host_value = 0;
  int* null_stream_buffer = nullptr;
  HIP_CHECK(hipMalloc(&null_stream_buffer, sizeof(int)));
  HIP_CHECK(hipMemcpy(null_stream_buffer, &host_value, sizeof(int), hipMemcpyHostToDevice));

  hipStream_t blocking_stream;
  hipStream_t non_blocking_stream;
  HIP_CHECK(hipStreamCreate(&blocking_stream));
  HIP_CHECK(hipStreamCreateWithFlags(&non_blocking_stream, hipStreamNonBlocking));

  CONSOLE_PRINT("Back-to-back launches, median us per launch (enqueue only in parentheses)");
  CONSOLE_PRINT(
      "kernel_us  non-blocking stream    blocking stream        blocking vs non-blocking");
  for (const uint64_t duration_us : kKernelDurationsUs) {
    const uint64_t duration_ticks = static_cast<uint64_t>(wall_clock_khz) * duration_us / 1000;
    const DispatchTiming non_blocking = MeasureDispatch(non_blocking_stream, duration_ticks);
    const DispatchTiming blocking = MeasureDispatch(blocking_stream, duration_ticks);
    CONSOLE_PRINT("%9llu  %7.3f (%7.3f)      %7.3f (%7.3f)      %+6.1f%%",
                  static_cast<unsigned long long>(duration_us), non_blocking.median_us,
                  non_blocking.enqueue_us, blocking.median_us, blocking.enqueue_us,
                  100.0 * (blocking.median_us / non_blocking.median_us - 1.0));
  }

  HIP_CHECK(hipStreamDestroy(non_blocking_stream));
  HIP_CHECK(hipStreamDestroy(blocking_stream));
  HIP_CHECK(hipFree(null_stream_buffer));
}

/**
 * End doxygen group hipPerfBlockingStreamDispatch.
 * @}
 */
