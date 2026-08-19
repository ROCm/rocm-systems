/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Per-Thread-Default-Stream Workload
 * @{
 * @ingroup HRRTest
 * Direct GPU workload for the per-thread-default-stream (`_spt`) memset entry
 * points, hidden with the Catch2 [.] tag and driven from hrr_roundtrip_test.cc
 * exactly like the other _Direct workloads.
 *
 * `_spt` is not an API surface applications call by name: it is reached by
 * building with `-fgpu-default-stream=per-thread`, which defines
 * HIP_API_PER_THREAD_DEFAULT_STREAM so amd_hip_runtime_pt_api.h redirects the
 * ordinary names (hipMemset -> hipMemset_spt, and so on).  This workload
 * therefore calls the ordinary names and lets the compiler select the entry
 * points, which is why it needs its own translation unit: the option is set per
 * source in CMakeLists.txt, and the other HrrTest workloads must keep resolving
 * to the plain entry points.
 *
 * Unit_HRR_MemsetSptRoundtrip asserts the recorded API ids, so if the per-source
 * option is ever dropped this stops testing nothing and fails instead.
 */

#include <hip_test_common.hh>

// The compile option is what makes this file meaningful; without it every call
// below resolves to the plain entry point already covered elsewhere.
#if !defined(HIP_API_PER_THREAD_DEFAULT_STREAM) && \
    !defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#error "hrr_spt_workload_test.cc requires the per-thread default stream compile option"
#endif

// hipMemcpy must keep resolving to the plain entry point: only its hand-written
// capture shim writes the data blob a D2H read is validated against, and
// hipMemcpy_spt has a NOOP playback handler.  Leaving it redirected would strip
// this workload of its replay oracle.  The same holds for every other redirected
// memcpy name, so undo the redirect for any of them this workload starts using.
#undef hipMemcpy

// ===========================================================================
// Workload: memset on the per-thread default stream
//
// Exercises hipMemset_spt / hipMemsetAsync_spt / hipMemset2D_spt /
// hipMemset2DAsync_spt through their ordinary names.  Each API fills its OWN
// buffer with its OWN byte pattern and each buffer is read back by its OWN D2H,
// so every API has an independent oracle: a NOOP playback handler for any single
// one of them leaves that buffer at its replay zero-init value and fails that
// D2H.  Pointing all four at one buffer would leave only the last writer
// observable.
//
// Pattern choice.  When the replayed and captured bytes are not identical the
// D2H validator falls back to float tolerance (atol=rtol=1e-3), trying
// f32/bf16/f16/f64 and accepting the first encoding with no out-of-tolerance
// element.  So an expected value has to decode above atol/(1-rtol) = 1.001e-3
// in EVERY candidate encoding before a zero-initialised replay buffer is
// reported as a mismatch.  As a repeated byte, the smallest decode is always
// the f16 one and the largest the f64 one:
//   0x41 -> f16 2.63  f32 1.21e1  bf16 1.21e1  f64 2.26e6
//   0x42 -> f16 3.13  f32 4.86e1  bf16 4.85e1  f64 1.57e11
//   0x44 -> f16 4.27  f32 7.85e2  bf16 7.84e2  f64 7.48e20
//   0x4C -> f16 17.2  f32 5.36e7  bf16 5.35e7  f64 3.55e59
// A small-magnitude pattern would not do: 0x2A, for instance, decodes to
// 1.51e-13 as f32, well inside the tolerance, so an all-zero replay buffer
// would be accepted.  Unit_HRR_MemsetSptRoundtrip additionally pins
// HIP_HRR_D2H_EXACT=1; the patterns keep the workload falsifiable without it.
// Distinct per-API patterns also make a swapped destination pointer visible.
// Final blobs: 0x41414141 / 0x42424242 / 0x44444444 / 0x4C4C4C4C.
// ===========================================================================
TEST_CASE("Unit_HRR_MemsetSpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 256;
  constexpr size_t SZ = N * sizeof(int);  // 1024 bytes per buffer

  // 2-D geometry for the pitched variants: pitch == row width, so the memset
  // covers the buffer contiguously and every validated byte is written by it.
  // Leaving part of a validated buffer unwritten would capture whatever the
  // allocation happened to hold, which replay cannot reproduce.
  constexpr size_t PITCH = 128;         // bytes per row
  constexpr size_t ROWS  = SZ / PITCH;  // 8 rows
  static_assert(PITCH * ROWS == SZ, "2-D memset must cover the whole buffer");

  int* d_set         = nullptr;
  int* d_set_async   = nullptr;
  int* d_set2d       = nullptr;
  int* d_set2d_async = nullptr;
  HIP_CHECK(hipMalloc(&d_set, SZ));
  HIP_CHECK(hipMalloc(&d_set_async, SZ));
  HIP_CHECK(hipMalloc(&d_set2d, SZ));
  HIP_CHECK(hipMalloc(&d_set2d_async, SZ));

  HIP_CHECK(hipMemset(d_set, 0x41, SZ));
  // The async variants need a null stream or they are indistinguishable from
  // the plain ones: PER_THREAD_DEFAULT_STREAM substitutes only a null or legacy
  // stream and passes an explicitly created one straight through.
  //
  // Omitting the stream instead, and so taking the _spt prototype's
  // hipStreamPerThread default, reaches the same stream but is deliberately not
  // exercised here: it records the sentinel handle, and replay only resolves
  // that because translate_stream returns nullptr for handles it cannot map.
  // AIRUNTIME-2556 turns that fallback into an explicit failure, which would
  // break this test as though the memset handlers had regressed.
  HIP_CHECK(hipMemsetAsync(d_set_async, 0x42, SZ, nullptr));
  HIP_CHECK(hipMemset2D(d_set2d, PITCH, 0x44, PITCH, ROWS));
  HIP_CHECK(hipMemset2DAsync(d_set2d_async, PITCH, 0x4C, PITCH, ROWS, nullptr));
  // The per-thread default stream does not implicitly synchronise with the
  // legacy stream the readbacks below run on.
  HIP_CHECK(hipDeviceSynchronize());

  // One D2H blob per API under test.
  int* h = new int[N]();
  HIP_CHECK(hipMemcpy(h, d_set, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x41414141);
  HIP_CHECK(hipMemcpy(h, d_set_async, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x42424242);
  HIP_CHECK(hipMemcpy(h, d_set2d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x44444444);
  HIP_CHECK(hipMemcpy(h, d_set2d_async, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x4C4C4C4C);

  HIP_CHECK(hipFree(d_set));
  HIP_CHECK(hipFree(d_set_async));
  HIP_CHECK(hipFree(d_set2d));
  HIP_CHECK(hipFree(d_set2d_async));
  delete[] h;
}

/**
 * End doxygen group HRRTest.
 * @}
 */
