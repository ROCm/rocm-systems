/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup memset memset
 * @{
 * @ingroup PerformanceTestMemory
 */

class Memset2DAsyncBenchmark : public Benchmark<Memset2DAsyncBenchmark> {
 public:
  Memset2DAsyncBenchmark(size_t width, size_t height)
      : dst_(width, height), stream_(Streams::created) {}

  void operator()(size_t width, size_t height) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream_.stream()) {
      HIP_CHECK(hipMemset2DAsync(dst_.ptr(), dst_.pitch(), 17, dst_.width(), dst_.height(),
                                 stream_.stream()));
    }
    HIP_CHECK(hipStreamSynchronize(stream_.stream()));
  }

 private:
  LinearAllocGuard2D<char> dst_;
  StreamGuard stream_;
};

static void RunBenchmark(size_t width, size_t height) {
  Memset2DAsyncBenchmark benchmark(width, height);
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");
  benchmark.Run(width, height);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset2DAsync`:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 * - performance/api/memset/hipMemset2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemset2DAsync) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32);
}

// Fills rows padded to a pitch 128 bytes wider than the row, starting dst_offset bytes into the
// allocation
class Memset2DAsyncPaddedBenchmark : public Benchmark<Memset2DAsyncPaddedBenchmark> {
 public:
  Memset2DAsyncPaddedBenchmark(size_t width, size_t height, size_t dst_offset)
      : dst_offset_(dst_offset),
        pitch_(width + 128),
        dst_(LinearAllocs::hipMalloc, dst_offset + pitch_ * height),
        stream_(Streams::created) {}

  void operator()(size_t width, size_t height) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream_.stream()) {
      HIP_CHECK(hipMemset2DAsync(dst_.ptr() + dst_offset_, pitch_, 17, width, height,
                                 stream_.stream()));
    }
    HIP_CHECK(hipStreamSynchronize(stream_.stream()));
  }

 private:
  size_t dst_offset_;
  size_t pitch_;
  LinearAllocGuard<char> dst_;
  StreamGuard stream_;
};

static void RunPaddedBenchmark(size_t width, size_t height, size_t dst_offset) {
  size_t free_memory = 0, total_memory = 0;
  HIP_CHECK(hipMemGetInfo(&free_memory, &total_memory));
  if (free_memory < dst_offset + (width + 128) * height + 1_MB) {
    HIP_SKIP_TEST("Not enough free device memory");
  }

  Memset2DAsyncPaddedBenchmark benchmark(width, height, dst_offset);
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");
  benchmark.AddSectionName("offset " + std::to_string(dst_offset));
  benchmark.RegisterBandwidth(width * height);
  benchmark.Run(width, height);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset2DAsync` on rows padded to a pitch wider than the row, that start 16-byte
 *    aligned and 11 bytes past a 16-byte boundary:
 *    -# Fill size
 *      - Small: 64 B x 32
 *      - Medium: 4 KB x 128, 4 KB x 1024
 *      - Large: 16 KB x 16384
 * Test source
 * ------------------------
 * - performance/api/memset/hipMemset2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemset2DAsync_Padded) {
  const auto [width, height] =
      GENERATE(table<size_t, size_t>({{64, 32}, {4_KB, 128}, {4_KB, 1024}, {16_KB, 16384}}));
  const auto dst_offset = GENERATE(0, 11);
  RunPaddedBenchmark(width, height, dst_offset);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
