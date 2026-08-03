/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemcpyAsync
 * @{
 * @ingroup PerformanceTestMemory
 * `hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
 *                            hipMemcpyKind kind, hipStream_t stream)` -
 * Measures aggregate copy bandwidth as the number of concurrent same-direction
 * streams scales from 1 to N. Each stream owns independent src/dst buffers, so the
 * only shared resource under contention is the SDMA copy-engine pool. This isolates
 * what CLR's SDMA engine selection logic buys: compare aggregate GB/s across
 * ROC_SDMA_ENGINE_SELECT modes (0=default/exclusivity, 1=round-robin, 2=disabled).
 *
 * Notes:
 *  - When stream count > 4, set GPU_MAX_HW_QUEUES >= stream count so streams are not
 *    serialized onto a small HW-queue pool before reaching the SDMA engines.
 *  - HIP_VISIBLE_DEVICES selects the device under test.
 *  - --iterations / --warmups control the measured/warmup batch counts.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <performance_common.hh>
#include <cmd_options.hh>

#include <chrono>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultIterations = 100;
constexpr int kDefaultWarmups = 10;

struct Direction {
  hipMemcpyKind kind;
  LinearAllocs src_alloc;
  LinearAllocs dst_alloc;
  const char* name;
};

// Pick RAII alloc types so each direction exercises the intended SDMA path.
Direction MakeDirection(hipMemcpyKind kind) {
  switch (kind) {
    case hipMemcpyHostToDevice:
      return {kind, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, "H2D"};
    case hipMemcpyDeviceToHost:
      return {kind, LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc, "D2H"};
    case hipMemcpyDeviceToDevice:
    default:
      return {kind, LinearAllocs::hipMalloc, LinearAllocs::hipMalloc, "D2D"};
  }
}

void RunConcurrentStreamCopy(hipMemcpyKind kind, size_t bytes_per_stream,
                             unsigned int num_streams) {
  const Direction dir = MakeDirection(kind);

  // Scale iterations down for large transfers so the total sweep stays reasonable.
  int default_iters = kDefaultIterations;
  if (bytes_per_stream >= (256u << 20)) default_iters = 10;        // >=256MB
  else if (bytes_per_stream >= (64u << 20)) default_iters = 30;    // >=64MB
  const int iterations =
      cmd_options.iterations == 5 ? default_iters : cmd_options.iterations;
  const int warmups = cmd_options.warmups == 5 ? kDefaultWarmups : cmd_options.warmups;

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<char>> src_bufs;
  std::vector<LinearAllocGuard<char>> dst_bufs;
  streams.reserve(num_streams);
  src_bufs.reserve(num_streams);
  dst_bufs.reserve(num_streams);

  for (unsigned int i = 0; i < num_streams; ++i) {
    streams.emplace_back(Streams::withFlags, hipStreamNonBlocking);
    src_bufs.emplace_back(dir.src_alloc, bytes_per_stream);
    dst_bufs.emplace_back(dir.dst_alloc, bytes_per_stream);
    HIP_CHECK(hipMemset(src_bufs[i].ptr(), i & 0xff, bytes_per_stream));
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Launch all N async copies, then sync all — the concurrency model that forces
  // the runtime to spread work across engines.
  auto issue_batch = [&](int iters) {
    for (int it = 0; it < iters; ++it) {
      for (unsigned int i = 0; i < num_streams; ++i) {
        HIP_CHECK(hipMemcpyAsync(dst_bufs[i].ptr(), src_bufs[i].ptr(), bytes_per_stream,
                                 dir.kind, streams[i].stream()));
      }
    }
    for (unsigned int i = 0; i < num_streams; ++i) {
      HIP_CHECK(hipStreamSynchronize(streams[i].stream()));
    }
  };

  issue_batch(warmups);

  const auto start = std::chrono::steady_clock::now();
  issue_batch(iterations);
  const std::chrono::duration<float, std::milli> elapsed_ms =
      std::chrono::steady_clock::now() - start;

  const size_t aggregate_bytes =
      static_cast<size_t>(num_streams) * bytes_per_stream * static_cast<size_t>(iterations);
  const double gbps = GetGigabytesPerSecond(aggregate_bytes, elapsed_ms.count());
  const double per_iter_ms = elapsed_ms.count() / static_cast<double>(iterations);

  CONSOLE_PRINT("ConcurrentStreamCopy %s streams=%2u size=%10zu bytes  "
                "aggregate %8.2f GB/s (%d iters, %.3f ms)",
                dir.name, num_streams, bytes_per_stream, gbps, iterations,
                elapsed_ms.count());
  // Machine-parseable line: direction,streams,bytes,iters,total_ms,per_iter_ms,aggregate_gbps
  CONSOLE_PRINT("SDMACSV,%s,%u,%zu,%d,%.4f,%.4f,%.4f", dir.name, num_streams,
                bytes_per_stream, iterations, elapsed_ms.count(), per_iter_ms, gbps);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Concurrent same-direction memcpy bandwidth vs. stream count (1..16).
 * Test source
 * ------------------------
 *  - performance/scenarios/memory/hipPerfConcurrentStreamCopy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Performance_hipPerfConcurrentStreamCopy) {
  const auto kind = GENERATE(hipMemcpyHostToDevice, hipMemcpyDeviceToHost);
  const auto bytes_per_stream =
      GENERATE(16_KB, 64_KB, 1_MB, 2_MB, 128_MB, 1_GB);
  const auto num_streams = GENERATE(4u, 8u, 16u);

  RunConcurrentStreamCopy(kind, bytes_per_stream, num_streams);
}
