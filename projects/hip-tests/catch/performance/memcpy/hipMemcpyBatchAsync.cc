/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
 */

#if HT_AMD

namespace {

constexpr size_t kBatchCount = 64;
constexpr unsigned char kPattern = 0x5a;

std::string GetSizeSectionName(size_t size) {
  if (size < 1_MB) {
    return std::to_string(size / 1_KB) + " KB";
  }
  return std::to_string(size / 1_MB) + " MB";
}

size_t AlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

double GetGbPerSecond(size_t bytes, float time_ms) {
  constexpr double kBytesPerGb = 1'000'000'000.0;
  return (static_cast<double>(bytes) / kBytesPerGb) /
         (static_cast<double>(time_ms) / 1000.0);
}

void PrintBandwidthStats(size_t bytes, float mean_ms, float best_ms,
                         float worst_ms) {
  const auto flags = std::cout.flags();
  const auto precision = std::cout.precision();

  std::cout << std::fixed << std::setprecision(2)
            << "Bandwidth: Average: " << GetGbPerSecond(bytes, mean_ms)
            << " GB/s, Best: " << GetGbPerSecond(bytes, best_ms)
            << " GB/s, Worst: " << GetGbPerSecond(bytes, worst_ms) << " GB/s\n";

  std::cout.flags(flags);
  std::cout.precision(precision);
}

class MemcpyBatchAsyncDtoDBenchmark
    : public Benchmark<MemcpyBatchAsyncDtoDBenchmark> {
public:
  void operator()(void **dsts, void **srcs, size_t *sizes, size_t count,
                  const hipStream_t &stream) {
    constexpr size_t kNumAttrs = 0;
    size_t attrs_idxs[1] = {0};
    size_t fail_idx = 0;

    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, count, nullptr,
                                    attrs_idxs, kNumAttrs, &fail_idx, stream));
    }
  }
};

void ValidateCopy(void *dst, size_t size) {
  std::array<unsigned char, 16> prefix = {};
  unsigned char suffix = 0;

  HIP_CHECK(
      hipMemcpy(prefix.data(), dst, prefix.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&suffix, static_cast<unsigned char *>(dst) + size - 1,
                      sizeof(suffix), hipMemcpyDeviceToHost));

  for (const auto value : prefix) {
    REQUIRE(value == kPattern);
  }
  REQUIRE(suffix == kPattern);
}

void RunBenchmark(size_t copy_size, size_t offset) {
  MemcpyBatchAsyncDtoDBenchmark benchmark;
  benchmark.AddSectionName(GetSizeSectionName(copy_size));
  benchmark.AddSectionName(offset == 0 ? "aligned" : "4-byte offset");
  benchmark.AddSectionName(std::to_string(kBatchCount) + " copies");

  constexpr size_t kAllocationAlignment = 256;
  const size_t stride = AlignUp(copy_size + offset, kAllocationAlignment);
  const size_t allocation_size = stride * kBatchCount;

  void *src_allocation = nullptr;
  void *dst_allocation = nullptr;
  HIP_CHECK(hipMalloc(&src_allocation, allocation_size));
  HIP_CHECK(hipMalloc(&dst_allocation, allocation_size));
  HIP_CHECK(hipMemset(src_allocation, kPattern, allocation_size));
  HIP_CHECK(hipMemset(dst_allocation, 0, allocation_size));

  std::vector<void *> srcs(kBatchCount);
  std::vector<void *> dsts(kBatchCount);
  std::vector<size_t> sizes(kBatchCount, copy_size);
  for (size_t i = 0; i < kBatchCount; ++i) {
    srcs[i] =
        static_cast<unsigned char *>(src_allocation) + (i * stride) + offset;
    dsts[i] =
        static_cast<unsigned char *>(dst_allocation) + (i * stride) + offset;
  }

  const StreamGuard stream_guard(Streams::created);
  const auto [mean_ms, deviation_ms, best_ms, worst_ms] =
      benchmark.Run(dsts.data(), srcs.data(), sizes.data(), sizes.size(),
                    stream_guard.stream());
  (void)deviation_ms;
  PrintBandwidthStats(copy_size * kBatchCount, mean_ms, best_ms, worst_ms);

  for (size_t i = 0; i < kBatchCount; ++i) {
    ValidateCopy(dsts[i], copy_size);
  }

  HIP_CHECK(hipFree(src_allocation));
  HIP_CHECK(hipFree(dst_allocation));
}

} // namespace

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyBatchAsync` with a same-device D2D batch so the ROCclr
 * optimized intra-device batch path is selected:
 *    -# Copy sizes: 4 KB, 256 KB, 1 MB, 4 MB, 16 MB, 256 MB
 *    -# Batch shape: four D2D copies of the same size
 *    -# Allocation type: hipMalloc source and destination buffers on one device
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - AMD
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_D2D_OptimizedPath_Aligned) {
  const auto copy_size = GENERATE(4_KB, 64_KB, 128_KB, 256_KB, 1_MB, 4_MB,
                                  16_MB, 64_MB, 128_MB, 256_MB, 1024_MB);
  RunBenchmark(copy_size, 0);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyBatchAsync` with same-device D2D copies whose source
 * and destination pointers are 4-byte aligned but not 16-byte aligned. Test
 * source
 * ------------------------
 *  - performance/memcpy/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - AMD
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_D2D_OptimizedPath_4ByteOffset) {
  const auto copy_size = GENERATE(4_KB, 64_KB, 128_KB, 256_KB, 1_MB, 4_MB,
                                  16_MB, 64_MB, 128_MB, 256_MB, 1024_MB);
  RunBenchmark(copy_size, sizeof(uint32_t));
}

#endif

/**
 * End doxygen group memcpy.
 * @}
 */
