/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <iostream>
#include <vector>

#include "hip_test_common.hh"

/**
 * Stream creation that reports and gives up instead of failing, for the multi-stream perf
 * scenarios.
 *
 * These sweeps ask for up to 1000+ concurrent streams, which not every part can provide. How
 * many fit is a device property, not a defect in the scenario, so report the memory state and
 * let the caller skip.
 *
 * The count has to be checked against free memory *before* creating, not simply attempted,
 * because on the PAL backend running out is not recoverable. Measured on gfx1201: PAL spends
 * ~14.75 MiB of device memory per stream, and when free memory reaches zero the new stream's
 * trap handler cannot be allocated ("KMD failed to setup the trap handler", paldevice.cpp), so
 * its command buffer fails to submit, VirtualGPU::flush() fails, and palvirtual.cpp latches
 * CL_INVALID_OPERATION into the process-global amd::Device::gpu_error_. That latch is static
 * and never cleared, and HIP_UPDATE_ERROR_STATE substitutes it for the real return value of
 * every subsequent HIP call -- so hipStreamCreate reports hipErrorLaunchFailure rather than the
 * hipErrorOutOfMemory it actually produced, and every later call in the process, including the
 * hipStreamDestroy that would free the memory, fails the same way. Nothing can be recovered
 * once that happens, hence the budget check.
 *
 * The ROCr backend behaves differently: ~16 KiB per stream, so memory is not the limiting
 * resource there, and it stops at a stream-count ceiling with a clean hipErrorOutOfMemory that
 * leaves the device usable. Both are handled below.
 */
namespace stream_test {

/** Fraction of free device memory the stream pool may occupy, leaving room for the allocations
 *  the scenario itself makes. */
constexpr double kMemoryBudgetFraction = 0.8;

/** Streams per probe batch. Two batches are measured and the slope taken, so the one-time cost
 *  paid by the first stream (context and pool setup, 2.5 GB on the ROCr path) cancels out. */
constexpr size_t kProbeBatch = 8;

/** Marginal device bytes consumed per stream, measured once per process.
 *
 *  Zero means streams do not measurably consume device memory, i.e. memory is not what limits
 *  the stream count on this backend -- do not derive a cap from it. The probe is only honest
 *  while the runtime is cold: once streams have been created and destroyed the runtime recycles
 *  them from a pool and a later probe sees no change at all. Catch2 re-runs the test body once
 *  per SECTION, so that is the normal case; measure on the first call and reuse the result.
 */
inline size_t MarginalBytesPerStream() {
  static const size_t kMeasured = [] {
    size_t totalMem = 0, freeAfterFirst = 0, freeAfterSecond = 0;
    std::vector<hipStream_t> probe;
    probe.reserve(2 * kProbeBatch);

    for (size_t batch = 0; batch < 2; ++batch) {
      for (size_t i = 0; i < kProbeBatch; ++i) {
        hipStream_t s = nullptr;
        if (hipStreamCreate(&s) != hipSuccess) break;
        probe.push_back(s);
      }
      if (hipMemGetInfo(batch == 0 ? &freeAfterFirst : &freeAfterSecond, &totalMem) !=
          hipSuccess) {
        break;
      }
    }

    for (hipStream_t s : probe) {
      static_cast<void>(hipStreamDestroy(s));
    }

    if (freeAfterFirst <= freeAfterSecond) return size_t{0};
    return (freeAfterFirst - freeAfterSecond) / kProbeBatch;
  }();
  return kMeasured;
}

/** Creates `count` streams, appending them to `streams`.
 *
 *  Returns false when the device cannot provide them, having printed total and free device
 *  memory together with the number of streams that were created. Two ways that happens:
 *  free memory cannot hold `count` at the measured per-stream cost (checked first, because on
 *  PAL actually running out is unrecoverable), or creation failed outright.
 *
 *  Errors other than the two exhaustion codes fail the test as usual.
 */
inline bool CreateStreamsOrReport(size_t count, bool nonBlocking,
                                  std::vector<hipStream_t>& streams) {
  auto report = [count](size_t created, const char* why) {
    size_t freeMem = 0, totalMem = 0;
    // The query itself fails on a context that has already been lost; say so rather than
    // printing a misleading "0 MiB free of 0 MiB total".
    if (hipMemGetInfo(&freeMem, &totalMem) == hipSuccess) {
      std::cout << "Cannot provide " << count << " streams (" << why << "): created " << created
                << "; device memory " << (freeMem >> 20) << " MiB free of " << (totalMem >> 20)
                << " MiB total" << std::endl;
    } else {
      std::cout << "Cannot provide " << count << " streams (" << why << "): created " << created
                << "; device memory unavailable (the device context has been lost)" << std::endl;
    }
  };

  // Predict, so that the unrecoverable PAL case is never reached.
  const size_t perStream = MarginalBytesPerStream();
  if (perStream != 0) {
    size_t freeMem = 0, totalMem = 0;
    HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
    const size_t affordable =
        static_cast<size_t>(static_cast<double>(freeMem) * kMemoryBudgetFraction) / perStream;
    if (count > affordable) {
      report(0, "exceeds the free device memory budget");
      std::cout << "  " << (perStream >> 20) << " MiB per stream, budget allows " << affordable
                << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < count; ++i) {
    hipStream_t s = nullptr;
    const hipError_t err = nonBlocking ? hipStreamCreateWithFlags(&s, hipStreamNonBlocking)
                                       : hipStreamCreate(&s);

    // Backstop for a limit that free memory does not predict: the ROCr path stops at a
    // stream-count ceiling with memory still available. hipErrorOutOfMemory is what
    // ihipStreamCreate really returns; hipErrorLaunchFailure is that same condition seen
    // through the global GPU-error latch described above, which the budget check should have
    // prevented, so it is purely defensive.
    if (err == hipErrorOutOfMemory || err == hipErrorLaunchFailure) {
      report(i, "hipStreamCreate failed");
      // Best effort: on a lost context these fail too, and HIP_CHECK here would turn a skip
      // into a failure.
      for (hipStream_t created : streams) {
        static_cast<void>(hipStreamDestroy(created));
      }
      streams.clear();
      return false;
    }

    HIP_CHECK(err);
    streams.push_back(s);
  }
  return true;
}

}  // namespace stream_test
