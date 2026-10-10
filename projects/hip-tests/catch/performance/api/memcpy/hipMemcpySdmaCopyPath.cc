/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTestMemory
 */

namespace {

const char* PathName() {
  const char* env = std::getenv("HSA_ENABLE_SDMA_USER_QUEUE");
  if (env == nullptr) return "native-user-queue(default)";
  return (std::atoi(env) != 0) ? "native-user-queue" : "legacy-sws-queue";
}

constexpr size_t kMaxBytes = 4 * 1024 * 1024;
constexpr int kMinIters = 200;
constexpr int kMaxIters = 200000;

// Roughly fixed total bytes per configuration so every size costs about the same wall time.
size_t TargetBytesPerConfig() {
  return isQuickLevel() ? (512ull * 1024 * 1024) : (2ull * 1024 * 1024 * 1024);
}

void Measure(const char* dir, hipMemcpyKind kind, void* dst, const void* src, size_t bytes,
             hipStream_t stream) {
  int iters = static_cast<int>(TargetBytesPerConfig() / bytes);
  if (iters < kMinIters) iters = kMinIters;
  if (iters > kMaxIters) iters = kMaxIters;

  // Warm up outside the measurement: queue creation and lazy blit setup would swamp the A/B.
  const int warmup = (iters / 20 < 100) ? 100 : iters / 20;
  for (int i = 0; i < warmup; ++i) HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  const double total = static_cast<double>(iters) * static_cast<double>(bytes);
  const double gib_s = (secs > 0.0) ? total / secs / (1024.0 * 1024.0 * 1024.0) : 0.0;
  const double us = (iters > 0) ? secs * 1e6 / iters : 0.0;
  std::printf("[SDMA-PERF] path=%-26s dir=%-8s size=%7zuKiB iters=%6d time=%6.3fs "
              "bw=%7.2f GiB/s per_copy=%9.2fus\n",
              PathName(), dir, bytes / 1024, iters, secs, gib_s, us);
  std::fflush(stdout);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Times hipMemcpyAsync in every direction that can reach an SDMA engine and prints one
 * [SDMA-PERF] line each, for comparing the native SDMA user queue against the legacy SWS
 * queue by running twice with HSA_ENABLE_SDMA_USER_QUEUE=1 and 0.
 *
 * H2D, D2H and D2D do not share a path inside ROCr: H2D and D2H use different SDMA engines,
 * while same-device D2D goes to the blit kernel unless hipMemcpyDeviceToDeviceNoCU forces it
 * onto SDMA, so D2D is listed under both kinds. A direction whose numbers do not move between
 * the two runs is not on the SDMA user-queue path at all.
 *
 * The sizes straddle ROCr's force_sdma_size (1 MiB), below which D2D goes to SDMA and above
 * which it goes to the blit kernel, so a step there is a dispatch change rather than a
 * regression. Use a section filter for one size per run:
 *
 *     MemcpyPerformance.exe Performance_hipMemcpy_SdmaCopyPath -c "512 KiB"
 *
 * These copies are latency-bound from ~128 KiB to ~1 MiB, so compare per-copy microseconds,
 * not GiB/s. ctest hides stdout for passing tests; run the exe directly or use ctest -V.
 * ------------------------
 *    - catch\performance\api\memcpy\hipMemcpySdmaCopyPath.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Performance_hipMemcpy_SdmaCopyPath) {
  size_t bytes = 0;
  SECTION("64 KiB") { bytes = 64 * 1024; }
  SECTION("128 KiB") { bytes = 128 * 1024; }
  SECTION("256 KiB") { bytes = 256 * 1024; }
  SECTION("512 KiB") { bytes = 512 * 1024; }
  SECTION("1 MiB") { bytes = 1024 * 1024; }
  SECTION("2 MiB") { bytes = 2048 * 1024; }
  SECTION("4 MiB") { bytes = 4096 * 1024; }

  unsigned char* h_src = nullptr;
  unsigned char* h_dst = nullptr;
  void* d_src = nullptr;
  void* d_dst = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_src), kMaxBytes));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_dst), kMaxBytes));
  HIP_CHECK(hipMalloc(&d_src, kMaxBytes));
  HIP_CHECK(hipMalloc(&d_dst, kMaxBytes));
  std::memset(h_src, 0xa5, kMaxBytes);
  std::memset(h_dst, 0, kMaxBytes);
  HIP_CHECK(hipMemcpy(d_src, h_src, kMaxBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_dst, 0, kMaxBytes));
  HIP_CHECK(hipDeviceSynchronize());

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::printf("SDMA copy-path performance  (path=%s)\n", PathName());
  std::fflush(stdout);

  Measure("H2D", hipMemcpyHostToDevice, d_src, h_src, bytes, stream);
  Measure("D2H", hipMemcpyDeviceToHost, h_dst, d_src, bytes, stream);
  Measure("D2D", hipMemcpyDeviceToDevice, d_dst, d_src, bytes, stream);
  Measure("D2D-NoCU", hipMemcpyDeviceToDeviceNoCU, d_dst, d_src, bytes, stream);

  // Not redundant with the timing: a wrong ring read pointer lets the producer overwrite
  // packets the engine has not consumed, which corrupts data rather than stalling.
  std::memset(h_dst, 0, bytes);
  HIP_CHECK(hipMemcpy(h_dst, d_dst, bytes, hipMemcpyDeviceToHost));
  REQUIRE(std::memcmp(h_dst, h_src, bytes) == 0);
  std::memset(h_dst, 0, bytes);
  HIP_CHECK(hipMemcpy(h_dst, d_src, bytes, hipMemcpyDeviceToHost));
  REQUIRE(std::memcmp(h_dst, h_src, bytes) == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipHostFree(h_src));
  HIP_CHECK(hipHostFree(h_dst));
}

/**
 * End doxygen group PerformanceTestMemory.
 * @}
 */
