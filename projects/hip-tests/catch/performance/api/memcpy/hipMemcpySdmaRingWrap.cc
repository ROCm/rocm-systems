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

constexpr size_t kRingWrapBufBytes = 64 * 1024;

// Issued before the clock starts, so queue creation and lazy blit setup stay out of the timing.
constexpr int kRingWrapWarmupCopies = 2000;

// Greppable summary for diffing the two paths across an HSA_ENABLE_SDMA_USER_QUEUE flip.
void PrintRingWrapTiming(long long copies, double copy_secs) {
  const char* env = std::getenv("HSA_ENABLE_SDMA_USER_QUEUE");
  const char* path = (env == nullptr)        ? "native-user-queue(default)"
                     : (std::atoi(env) != 0) ? "native-user-queue"
                                             : "legacy-sws-queue";
  const double bytes = static_cast<double>(copies) * static_cast<double>(kRingWrapBufBytes);
  const double rate = (copy_secs > 0.0) ? static_cast<double>(copies) / copy_secs : 0.0;
  const double gib_s = (copy_secs > 0.0) ? bytes / copy_secs / (1024.0 * 1024.0 * 1024.0) : 0.0;
  const double us_per_copy = (copies > 0) ? copy_secs * 1e6 / static_cast<double>(copies) : 0.0;

  std::printf(
      "[SDMA-PERF] path=%-26s copies=%lld size=%zuKiB time=%.3fs rate=%.0f copies/s "
      "bw=%.2f GiB/s per_copy=%.2fus\n",
      path, copies, kRingWrapBufBytes / 1024, copy_secs, rate, gib_s, us_per_copy);
  std::fflush(stdout);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Drives far more device-to-device NoCU traffic through one stream than the SDMA blit
 * ring holds, forcing the ring to wrap several times, and reports the achieved rate. It lives
 * in the performance suite because it runs for seconds by design.
 *
 * NoCU is required: plain hipMemcpyDeviceToDevice is serviced by the blit kernel and pinned
 * host<->device copies are satisfied above ROCr, so neither reaches this ring.
 *
 * BlitSdma's producer only consults the ring read pointer in CanWriteUpto():
 *     (upto_index - *queue_rptr_) < kQueueSize
 * Both are monotonic byte counts, so below one ring's worth of traffic that check passes
 * whatever the read pointer holds -- a stale or dead read pointer is invisible, and no other
 * test gets far enough to notice. Past that point the producer spins forever in
 * AcquireWriteAddress waiting for space the engine already freed, so the failure is a HANG
 * reported by ctest's timeout, with the progress lines showing how far it got.
 *
 * Prints a [SDMA-PERF] line for comparing the two paths (HSA_ENABLE_SDMA_USER_QUEUE=1 vs 0).
 * ctest hides stdout for passing tests, so run the exe directly or use ctest -V.
 * ------------------------
 *    - catch\performance\api\memcpy\hipMemcpySdmaRingWrap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Performance_hipMemcpy_SdmaRingWrap) {
  // Both settings move several times the ring's capacity, so the ring is reused repeatedly.
  const int iters = isQuickLevel() ? 120000 : 250000;

  unsigned char* h_pattern = nullptr;
  unsigned char* h_check = nullptr;
  void* d_src = nullptr;
  void* d_dst = nullptr;

  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_pattern), kRingWrapBufBytes));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_check), kRingWrapBufBytes));
  HIP_CHECK(hipMalloc(&d_src, kRingWrapBufBytes));
  HIP_CHECK(hipMalloc(&d_dst, kRingWrapBufBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  for (size_t i = 0; i < kRingWrapBufBytes; ++i)
    h_pattern[i] = static_cast<unsigned char>((i * 7) & 0xff);
  HIP_CHECK(hipMemcpy(d_src, h_pattern, kRingWrapBufBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_dst, 0, kRingWrapBufBytes));
  HIP_CHECK(hipDeviceSynchronize());

  // Warm up outside the measurement: queue creation and lazy blit setup would skew the A/B.
  for (int i = 0; i < kRingWrapWarmupCopies; ++i) {
    HIP_CHECK(
        hipMemcpyAsync(d_dst, d_src, kRingWrapBufBytes, hipMemcpyDeviceToDeviceNoCU, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  double copy_secs = 0.0;
  auto span_start = std::chrono::steady_clock::now();
  long long copies = 0;
  bool mismatch = false;

  for (int i = 0; i < iters; ++i) {
    HIP_CHECK(
        hipMemcpyAsync(d_dst, d_src, kRingWrapBufBytes, hipMemcpyDeviceToDeviceNoCU, stream));

    // Drain periodically, or the stream's queue grows unboundedly and we measure backpressure
    // instead of ring reuse. Also verifies data, since a wrong read pointer can corrupt
    // in-flight packets rather than merely stall.
    if ((i % 1024) == 1023) {
      // The synchronize and readback are bookkeeping, not the copy path being measured.
      HIP_CHECK(hipStreamSynchronize(stream));
      copy_secs +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - span_start).count();

      std::memset(h_check, 0, kRingWrapBufBytes);
      HIP_CHECK(hipMemcpy(h_check, d_dst, kRingWrapBufBytes, hipMemcpyDeviceToHost));
      if (std::memcmp(h_check, h_pattern, kRingWrapBufBytes) != 0) {
        mismatch = true;
        break;
      }
      span_start = std::chrono::steady_clock::now();
    }

    ++copies;
    // Breadcrumb: on a wedge, the last line printed says how many copies got through.
    if ((copies % 50000) == 0) {
      std::printf("  %lld copies\n", copies);
      std::fflush(stdout);
    }
  }

  HIP_CHECK(hipStreamSynchronize(stream));
  copy_secs +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - span_start).count();

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipHostFree(h_pattern));
  HIP_CHECK(hipHostFree(h_check));

  PrintRingWrapTiming(copies, copy_secs);

  REQUIRE_FALSE(mismatch);
  REQUIRE(copies == iters);
}

/**
 * End doxygen group PerformanceTestMemory.
 * @}
 */
