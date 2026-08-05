/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTestMemory
 */

namespace {

constexpr unsigned char kPattern = 0x5a;

enum class PointerPattern {
  kBasePointers,
  kBroadcastSource,
};

std::string GetSizeSectionName(size_t size) {
  if (size < 1_MB) {
    return std::to_string(size / 1_KB) + " KB";
  }
  return std::to_string(size / 1_MB) + " MB";
}

std::string GetPointerPatternSectionName(PointerPattern pointer_pattern) {
  switch (pointer_pattern) {
    case PointerPattern::kBasePointers:
      return "base pointers";
    case PointerPattern::kBroadcastSource:
      return "broadcast source";
  }
  return "unknown pointer pattern";
}

class MemcpyBatchAsync : public Benchmark<MemcpyBatchAsync> {
 public:
  void operator()(void** dsts, void** srcs, size_t* sizes, size_t count, hipStream_t stream) {
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(
          hipMemcpyBatchAsync(dsts, srcs, sizes, count, nullptr, nullptr, 0, nullptr, stream));
    }
  }
};

class MemcpySequentialAsync : public Benchmark<MemcpySequentialAsync> {
 public:
  void operator()(void** dsts, void** srcs, size_t* sizes, size_t count, hipMemcpyKind kind,
                  hipStream_t stream) {
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      for (size_t i = 0; i < count; ++i) {
        HIP_CHECK(hipMemcpyAsync(dsts[i], srcs[i], sizes[i], kind, stream));
      }
    }
  }
};

template <typename BenchmarkType>
void AddCommonSectionNames(BenchmarkType& benchmark, size_t copy_size, size_t batch_copy_count,
                           PointerPattern pointer_pattern) {
  benchmark.AddSectionName(GetPointerPatternSectionName(pointer_pattern));
  benchmark.AddSectionName(GetSizeSectionName(copy_size));
  benchmark.AddSectionName(std::to_string(batch_copy_count) + " copies");
  benchmark.RegisterBandwidth(copy_size * batch_copy_count);
}

std::string FormatSpeedupRatio(float batch_mean, float sequential_mean) {
  std::ostringstream ratio;
  ratio << std::fixed << std::setprecision(2) << sequential_mean / batch_mean;
  return ratio.str();
}

void RunComparison(void** dsts, void** srcs, size_t* sizes, size_t count, size_t copy_size,
                   size_t batch_copy_count, PointerPattern pointer_pattern, hipMemcpyKind kind) {
  const StreamGuard stream_guard{Streams::created};
  const hipStream_t stream = stream_guard.stream();

  MemcpySequentialAsync sequential_benchmark;
  sequential_benchmark.SetDisplayOutput(false);
  const auto sequential_stats = sequential_benchmark.Run(dsts, srcs, sizes, count, kind, stream);
  const float sequential_mean = std::get<0>(sequential_stats);

  MemcpyBatchAsync batch_benchmark;
  AddCommonSectionNames(batch_benchmark, copy_size, batch_copy_count, pointer_pattern);
  batch_benchmark.RegisterStatsSuffix([sequential_mean](float batch_mean) {
    return " Ratio " + FormatSpeedupRatio(batch_mean, sequential_mean) + "x";
  });
  batch_benchmark.Run(dsts, srcs, sizes, count, stream);
}

void RunDeviceToDeviceBenchmark(size_t copy_size, size_t batch_copy_count,
                                PointerPattern pointer_pattern) {
  const size_t allocation_size = copy_size * batch_copy_count;

  LinearAllocGuard<unsigned char> src_allocation(LinearAllocs::hipMalloc, allocation_size);
  LinearAllocGuard<unsigned char> dst_allocation(LinearAllocs::hipMalloc, allocation_size);
  HIP_CHECK(hipMemset(src_allocation.ptr(), kPattern, allocation_size));

  std::vector<void*> srcs(batch_copy_count);
  std::vector<void*> dsts(batch_copy_count);
  std::vector<size_t> sizes(batch_copy_count, copy_size);
  for (size_t i = 0; i < batch_copy_count; ++i) {
    srcs[i] = pointer_pattern == PointerPattern::kBroadcastSource
                  ? src_allocation.ptr()
                  : src_allocation.ptr() + (i * copy_size);
    dsts[i] = dst_allocation.ptr() + (i * copy_size);
  }

  RunComparison(dsts.data(), srcs.data(), sizes.data(), sizes.size(), copy_size, batch_copy_count,
                pointer_pattern, hipMemcpyDeviceToDevice);
}

void RunPeerToPeerBenchmark(size_t copy_size, size_t batch_copy_count,
                            PointerPattern pointer_pattern) {
  const size_t allocation_size = copy_size * batch_copy_count;
  HIP_CHECK(hipSetDevice(0));
  auto [src_device, dst_device] = GetDeviceIds(true);

  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocGuard<unsigned char> src_allocation(LinearAllocs::hipMalloc, allocation_size);
  HIP_CHECK(hipMemset(src_allocation.ptr(), kPattern, allocation_size));

  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocGuard<unsigned char> dst_allocation(LinearAllocs::hipMalloc, allocation_size);

  std::vector<void*> srcs(batch_copy_count);
  std::vector<void*> dsts(batch_copy_count);
  std::vector<size_t> sizes(batch_copy_count, copy_size);
  for (size_t i = 0; i < batch_copy_count; ++i) {
    srcs[i] = pointer_pattern == PointerPattern::kBroadcastSource
                  ? src_allocation.ptr()
                  : src_allocation.ptr() + (i * copy_size);
    dsts[i] = dst_allocation.ptr() + (i * copy_size);
  }

  HIP_CHECK(hipSetDevice(src_device));
  RunComparison(dsts.data(), srcs.data(), sizes.data(), sizes.size(), copy_size, batch_copy_count,
                pointer_pattern, hipMemcpyDeviceToDevice);
}

void RunHostDeviceBenchmark(size_t copy_size, size_t batch_copy_count,
                            LinearAllocs src_allocation_type, LinearAllocs dst_allocation_type,
                            PointerPattern pointer_pattern) {
  const size_t allocation_size = copy_size * batch_copy_count;

  LinearAllocGuard<unsigned char> src_allocation(src_allocation_type, allocation_size);
  LinearAllocGuard<unsigned char> dst_allocation(dst_allocation_type, allocation_size);
  if (src_allocation_type == LinearAllocs::hipMalloc) {
    HIP_CHECK(hipMemset(src_allocation.ptr(), kPattern, allocation_size));
  } else {
    std::memset(src_allocation.host_ptr(), kPattern, allocation_size);
  }

  std::vector<void*> srcs(batch_copy_count);
  std::vector<void*> dsts(batch_copy_count);
  std::vector<size_t> sizes(batch_copy_count, copy_size);
  for (size_t i = 0; i < batch_copy_count; ++i) {
    srcs[i] = pointer_pattern == PointerPattern::kBroadcastSource
                  ? src_allocation.ptr()
                  : src_allocation.ptr() + (i * copy_size);
    dsts[i] = dst_allocation.ptr() + (i * copy_size);
  }

  const hipMemcpyKind kind = src_allocation_type == LinearAllocs::hipMalloc ? hipMemcpyDeviceToHost
                                                                            : hipMemcpyHostToDevice;
  RunComparison(dsts.data(), srcs.data(), sizes.data(), sizes.size(), copy_size, batch_copy_count,
                pointer_pattern, kind);
}

}  // namespace

HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_D2D) {
  const PointerPattern pointer_pattern =
      GENERATE(PointerPattern::kBasePointers, PointerPattern::kBroadcastSource);
  const auto [copy_size, batch_copy_count] = GENERATE(table<size_t, size_t>(
      {{4_KB, 4096}, {16_KB, 1024}, {64_KB, 256}, {256_KB, 64}, {1024_KB, 16}}));
  RunDeviceToDeviceBenchmark(copy_size, batch_copy_count, pointer_pattern);
}

HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_P2P) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  const PointerPattern pointer_pattern =
      GENERATE(PointerPattern::kBasePointers, PointerPattern::kBroadcastSource);
  const auto [copy_size, batch_copy_count] = GENERATE(table<size_t, size_t>(
      {{4_KB, 4096}, {16_KB, 1024}, {64_KB, 256}, {256_KB, 64}, {1024_KB, 16}}));
  RunPeerToPeerBenchmark(copy_size, batch_copy_count, pointer_pattern);
}

HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_H2D_Pageable) {
  const PointerPattern pointer_pattern =
      GENERATE(PointerPattern::kBasePointers, PointerPattern::kBroadcastSource);
  const auto [copy_size, batch_copy_count] = GENERATE(table<size_t, size_t>(
      {{4_KB, 4096}, {16_KB, 1024}, {64_KB, 256}, {256_KB, 64}, {1024_KB, 16}}));
  RunHostDeviceBenchmark(copy_size, batch_copy_count, LinearAllocs::malloc, LinearAllocs::hipMalloc,
                         pointer_pattern);
}

HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_H2D_Pinned) {
  const PointerPattern pointer_pattern =
      GENERATE(PointerPattern::kBasePointers, PointerPattern::kBroadcastSource);
  const auto [copy_size, batch_copy_count] = GENERATE(table<size_t, size_t>(
      {{4_KB, 4096}, {16_KB, 1024}, {64_KB, 256}, {256_KB, 64}, {1024_KB, 16}}));
  RunHostDeviceBenchmark(copy_size, batch_copy_count, LinearAllocs::hipHostMalloc,
                         LinearAllocs::hipMalloc, pointer_pattern);
}

HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_D2H_Pageable) {
  const PointerPattern pointer_pattern =
      GENERATE(PointerPattern::kBasePointers, PointerPattern::kBroadcastSource);
  const auto [copy_size, batch_copy_count] = GENERATE(table<size_t, size_t>(
      {{4_KB, 4096}, {16_KB, 1024}, {64_KB, 256}, {256_KB, 64}, {1024_KB, 16}}));
  RunHostDeviceBenchmark(copy_size, batch_copy_count, LinearAllocs::hipMalloc, LinearAllocs::malloc,
                         pointer_pattern);
}

HIP_TEST_CASE(Performance_hipMemcpyBatchAsync_D2H_Pinned) {
  const PointerPattern pointer_pattern =
      GENERATE(PointerPattern::kBasePointers, PointerPattern::kBroadcastSource);
  const auto [copy_size, batch_copy_count] = GENERATE(table<size_t, size_t>(
      {{4_KB, 4096}, {16_KB, 1024}, {64_KB, 256}, {256_KB, 64}, {1024_KB, 16}}));
  RunHostDeviceBenchmark(copy_size, batch_copy_count, LinearAllocs::hipMalloc,
                         LinearAllocs::hipHostMalloc, pointer_pattern);
}

/**
 * End doxygen group memcpy.
 * @}
 */
