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

class Memset3DAsyncBenchmark : public Benchmark<Memset3DAsyncBenchmark> {
 public:
  Memset3DAsyncBenchmark(size_t width, size_t height, size_t depth)
      : dst_(width, height, depth), stream_(Streams::created) {}

  void operator()() {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream_.stream()) {
      HIP_CHECK(hipMemset3DAsync(dst_.pitched_ptr(), 17, dst_.extent(), stream_.stream()));
    }
    HIP_CHECK(hipStreamSynchronize(stream_.stream()));
  }

 private:
  LinearAllocGuard3D<char> dst_;
  StreamGuard stream_;
};

static void RunBenchmark(size_t width, size_t height, size_t depth) {
  Memset3DAsyncBenchmark benchmark(width, height, depth);
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ", " +
                           std::to_string(depth) + ")");
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset3DAsync`:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 * - performance/api/memset/hipMemset3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemset3DAsync) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 16, 4);
}

// Fills rows padded to a pitch 128 bytes wider than the row, in slices padded by a row, starting
// dst_offset bytes into the allocation
class Memset3DAsyncPaddedBenchmark : public Benchmark<Memset3DAsyncPaddedBenchmark> {
 public:
  Memset3DAsyncPaddedBenchmark(const hipExtent extent, size_t dst_offset)
      : extent_(extent),
        pitch_(extent.width + 128),
        rows_per_slice_(extent.height + 1),
        dst_(LinearAllocs::hipMalloc, dst_offset + pitch_ * rows_per_slice_ * extent.depth),
        dst_ptr_(make_hipPitchedPtr(dst_.ptr() + dst_offset, pitch_, extent.width,
                                    rows_per_slice_)),
        stream_(Streams::created) {}

  void operator()() {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream_.stream()) {
      HIP_CHECK(hipMemset3DAsync(dst_ptr_, 17, extent_, stream_.stream()));
    }
    HIP_CHECK(hipStreamSynchronize(stream_.stream()));
  }

 private:
  hipExtent extent_;
  size_t pitch_;
  size_t rows_per_slice_;
  LinearAllocGuard<char> dst_;
  hipPitchedPtr dst_ptr_;
  StreamGuard stream_;
};

static void RunPaddedBenchmark(const hipExtent extent, size_t dst_offset) {
  size_t free_memory = 0, total_memory = 0;
  HIP_CHECK(hipMemGetInfo(&free_memory, &total_memory));
  if (free_memory <
      dst_offset + (extent.width + 128) * (extent.height + 1) * extent.depth + 1_MB) {
    HIP_SKIP_TEST("Not enough free device memory");
  }

  Memset3DAsyncPaddedBenchmark benchmark(extent, dst_offset);
  benchmark.AddSectionName("(" + std::to_string(extent.width) + ", " +
                           std::to_string(extent.height) + ", " + std::to_string(extent.depth) +
                           ")");
  benchmark.AddSectionName("offset " + std::to_string(dst_offset));
  benchmark.RegisterBandwidth(extent.width * extent.height * extent.depth);
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset3DAsync` on rows padded to a pitch wider than the row and slices padded by
 *    a row, with rows that start 16-byte aligned and 11 bytes past a 16-byte boundary:
 *    -# Fill size
 *      - Small: 257 B x 17 x 3
 *      - Medium: 1 KB x 1024 x 8
 *      - Large: 4 KB x 4096 x 16
 * Test source
 * ------------------------
 * - performance/api/memset/hipMemset3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemset3DAsync_Padded) {
  const auto [width, height, depth] = GENERATE(
      table<size_t, size_t, size_t>({{257, 17, 3}, {1_KB, 1024, 8}, {4_KB, 4096, 16}}));
  const auto dst_offset = GENERATE(0, 11);
  RunPaddedBenchmark(make_hipExtent(width, height, depth), dst_offset);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
