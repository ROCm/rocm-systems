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

double GetGbPerSecond(size_t bytes, float timeMs) {
  constexpr double kBytesPerGb = 1'000'000'000.0;
  return (static_cast<double>(bytes) / kBytesPerGb) /
         (static_cast<double>(timeMs) / 1000.0);
}

void PrintBandwidthStats(size_t bytes, float meanMs, float bestMs,
                         float worstMs) {
  const auto flags = std::cout.flags();
  const auto precision = std::cout.precision();

  std::cout << std::fixed << std::setprecision(2)
            << "Bandwidth: Average: " << GetGbPerSecond(bytes, meanMs)
            << " GB/s, Best: " << GetGbPerSecond(bytes, bestMs)
            << " GB/s, Worst: " << GetGbPerSecond(bytes, worstMs) << " GB/s\n";

  std::cout.flags(flags);
  std::cout.precision(precision);
}

class MemcpyBatchAsyncDtoDBenchmark
    : public Benchmark<MemcpyBatchAsyncDtoDBenchmark> {
public:
  void operator()(void **dsts, void **srcs, size_t *sizes, size_t count,
                  const hipStream_t &stream) {
    constexpr size_t numAttrs = 0;
    size_t attrsIdxs[1] = {0};
    size_t failIdx = 0;

    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, count, nullptr,
                                    attrsIdxs, numAttrs, &failIdx, stream));
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

void RunBenchmark(size_t copySize, size_t offset) {
  MemcpyBatchAsyncDtoDBenchmark benchmark;
  benchmark.AddSectionName(GetSizeSectionName(copySize));
  benchmark.AddSectionName(offset == 0 ? "aligned" : "4-byte offset");
  benchmark.AddSectionName(std::to_string(kBatchCount) + " copies");

  constexpr size_t kAllocationAlignment = 256;
  const size_t stride = AlignUp(copySize + offset, kAllocationAlignment);
  const size_t allocationSize = stride * kBatchCount;

  void *srcAllocation = nullptr;
  void *dstAllocation = nullptr;
  HIP_CHECK(hipMalloc(&srcAllocation, allocationSize));
  HIP_CHECK(hipMalloc(&dstAllocation, allocationSize));
  HIP_CHECK(hipMemset(srcAllocation, kPattern, allocationSize));
  HIP_CHECK(hipMemset(dstAllocation, 0, allocationSize));

  std::vector<void *> srcs(kBatchCount);
  std::vector<void *> dsts(kBatchCount);
  std::vector<size_t> sizes(kBatchCount, copySize);
  for (size_t i = 0; i < kBatchCount; ++i) {
    srcs[i] =
        static_cast<unsigned char *>(srcAllocation) + (i * stride) + offset;
    dsts[i] =
        static_cast<unsigned char *>(dstAllocation) + (i * stride) + offset;
  }

  const StreamGuard streamGuard(Streams::created);
  const auto [meanMs, deviationMs, bestMs, worstMs] =
      benchmark.Run(dsts.data(), srcs.data(), sizes.data(), sizes.size(),
                    streamGuard.stream());
  (void)deviationMs;
  PrintBandwidthStats(copySize * kBatchCount, meanMs, bestMs, worstMs);

  for (size_t i = 0; i < kBatchCount; ++i) {
    ValidateCopy(dsts[i], copySize);
  }

  HIP_CHECK(hipFree(srcAllocation));
  HIP_CHECK(hipFree(dstAllocation));
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
  const auto copySize = GENERATE(4_KB, 64_KB, 128_KB, 256_KB, 1_MB, 4_MB, 16_MB,
                                 64_MB, 128_MB, 256_MB, 1024_MB);
  RunBenchmark(copySize, 0);
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
  const auto copySize = GENERATE(4_KB, 64_KB, 128_KB, 256_KB, 1_MB, 4_MB, 16_MB,
                                 64_MB, 128_MB, 256_MB, 1024_MB);
  RunBenchmark(copySize, sizeof(uint32_t));
}

#endif

/**
 * End doxygen group memcpy.
 * @}
 */
