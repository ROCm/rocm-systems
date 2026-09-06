/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Per-Thread-Default-Stream Workloads
 * @{
 * @ingroup HRRTest
 * Direct GPU workloads for the per-thread-default-stream (`_spt`) entry points,
 * hidden with the Catch2 [.] tag and driven from hrr_roundtrip_test.cc exactly
 * like the other _Direct workloads.
 *
 * `_spt` is not an API surface applications call by name: it is reached by
 * building with `-fgpu-default-stream=per-thread`, which defines
 * HIP_API_PER_THREAD_DEFAULT_STREAM so amd_hip_runtime_pt_api.h redirects the
 * ordinary names (hipMemset -> hipMemset_spt, and so on).  These workloads
 * therefore call the ordinary names and let the compiler select the entry
 * points, which is why they need their own translation unit: the option is set
 * per source in CMakeLists.txt, and the other HrrTest workloads must keep
 * resolving to the plain entry points.
 *
 * Because the source no longer names the API under test, the archive is the only
 * evidence that the redirect took effect.  Every roundtrip driver asserts the
 * recorded `_spt` API ids before replaying, so if the per-source option is ever
 * dropped these stop testing nothing and fail instead.
 *
 * Two conventions hold throughout, both of them load-bearing:
 *
 *  - Oracle policy.  A replay-validated buffer must be written and read by APIs
 *    that both snapshot a data blob at capture and reproduce their effect at
 *    replay.  Several `_spt` handlers are deliberate no-ops, so where the API
 *    under test is one of them it operates on a scratch buffer and the validated
 *    buffer is written separately.  Those workloads cover the capture path; the
 *    recorded-id assertions are what makes that coverage meaningful.  Where the
 *    handler is faithful the workload validates the effect directly, one D2H per
 *    API so a single regressed handler fails its own buffer.
 *
 *  - Macro ordering.  Workloads disagree about which names must stay redirected:
 *    one tests hipMemcpy_spt, the rest need the plain hipMemcpy as their oracle,
 *    and the graph frame needs the plain stream-capture pair.  The redirects are
 *    therefore undone in stages, and each stage is followed by a preprocessor
 *    check rather than a comment asking the next reader to be careful, so
 *    reordering the file breaks the build instead of quietly testing the plain
 *    entry point.
 */

#include <hip_test_common.hh>

// The compile option is what makes this file meaningful; without it every call
// below resolves to the plain entry point already covered elsewhere.
#if !defined(HIP_API_PER_THREAD_DEFAULT_STREAM) && \
    !defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#error "hrr_spt_workload_test.cc requires the per-thread default stream compile option"
#endif

// Driver-style entry points are absent from amd_hip_runtime_pt_api.h, so they
// are never redirected and stay usable as an oracle whatever the macro state
// below is.  hipMemsetD32 / hipMemsetD32Async / hipMemcpyDtoDAsync have faithful
// replay handlers, and hipMemcpyHtoD / hipMemcpyDtoH / hipMemcpyDtoHAsync have
// hand-written capture shims that snapshot the blob a D2H read is validated
// against.  The workloads whose own API under test is a redirected memcpy use
// these instead of the plain hipMemcpy the rest of the file relies on.

// Device symbol for the symbol-memcpy variants, sized to hold a whole transfer.
__device__ int g_hrr_spt_symbol[256];

#if !defined(hipMemcpy) || !defined(hipMemcpyAsync) || !defined(hipMemcpy2D) || \
    !defined(hipMemcpy2DAsync) || !defined(hipMemcpyToSymbol) ||               \
    !defined(hipMemcpyToSymbolAsync)
#error "Unit_HRR_MemcpySpt_Direct must run before the memcpy redirects are undone"
#endif

// ===========================================================================
// Workload: memcpy on the per-thread default stream
//
// Exercises hipMemcpy_spt, hipMemcpyAsync_spt, hipMemcpy2D_spt,
// hipMemcpy2DAsync_spt, hipMemcpyToSymbol_spt and hipMemcpyToSymbolAsync_spt
// through their ordinary names.  All six have no-op playback handlers, so they
// run against a scratch buffer that is freed before the validated one is
// written: routing the oracle through them would leave the replayed buffer at
// its zero-init value and fail the D2H for a reason that has nothing to do with
// whether the redirect took effect.  This is capture-path coverage, and
// Unit_HRR_MemcpySptRoundtrip asserts each recorded id.
//
// The oracle is hipMemcpyHtoD / hipMemcpyDtoH rather than the plain hipMemcpy
// the later workloads use, because hipMemcpy has to stay redirected here to
// reach hipMemcpy_spt at all.
// Final blob: 0x41414141.
// ===========================================================================
TEST_CASE("Unit_HRR_MemcpySpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x41414141;

  // 2-D geometry covering the buffer contiguously, so no unwritten padding can
  // reach a validated blob.
  constexpr size_t PITCH = 128;
  constexpr size_t ROWS  = SZ / PITCH;
  static_assert(PITCH * ROWS == SZ, "2-D copy must cover the whole buffer");

  int* h_src = new int[N];
  for (int i = 0; i < N; ++i) h_src[i] = VAL;
  int* h_chk = new int[N]();

  // ---- APIs under test, on a scratch buffer -------------------------------
  int* d_spt = nullptr;
  HIP_CHECK(hipMalloc(&d_spt, SZ));

  HIP_CHECK(hipMemcpy(d_spt, h_src, SZ, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(h_chk, d_spt, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == VAL);

  // The async variants need a null stream or they are indistinguishable from
  // the plain ones: PER_THREAD_DEFAULT_STREAM substitutes only a null or legacy
  // stream and passes an explicitly created one straight through.
  HIP_CHECK(hipMemcpyAsync(d_spt, h_src, SZ, hipMemcpyHostToDevice, nullptr));
  HIP_CHECK(hipMemcpyAsync(h_chk, d_spt, SZ, hipMemcpyDeviceToHost, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == VAL);

  HIP_CHECK(hipMemcpy2D(d_spt, PITCH, h_src, PITCH, PITCH, ROWS,
                        hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(h_chk, PITCH, d_spt, PITCH, PITCH, ROWS,
                        hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == VAL);

  HIP_CHECK(hipMemcpy2DAsync(d_spt, PITCH, h_src, PITCH, PITCH, ROWS,
                             hipMemcpyHostToDevice, nullptr));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hrr_spt_symbol), h_src, SZ, 0,
                              hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_hrr_spt_symbol), h_src, SZ, 0,
                                   hipMemcpyHostToDevice, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(d_spt));

  // ---- Replay oracle, on driver-style entry points -------------------------
  int* d_ok = nullptr;
  HIP_CHECK(hipMalloc(&d_ok, SZ));
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(d_ok), h_src, SZ));
  HIP_CHECK(hipDeviceSynchronize());

  int* h_out = new int[N]();
  HIP_CHECK(hipMemcpyDtoH(h_out, reinterpret_cast<hipDeviceptr_t>(d_ok), SZ));
  for (int i = 0; i < N; ++i) REQUIRE(h_out[i] == VAL);

  HIP_CHECK(hipFree(d_ok));
  delete[] h_src;
  delete[] h_chk;
  delete[] h_out;
}

// hipMemcpy must keep resolving to the plain entry point: only its hand-written
// capture shim writes the data blob a D2H read is validated against, and
// hipMemcpy_spt has a NOOP playback handler.  Leaving it redirected would strip
// the workloads below of their replay oracle.  The same holds for every other
// redirected memcpy name, so undo the redirect for any of them they start using.
#undef hipMemcpy

#if defined(hipMemcpy)
#error "the workloads below need the plain hipMemcpy as their replay oracle"
#endif

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
// Every workload in this file follows the same rule for the same reason.
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

// ===========================================================================
// Workload: hipMemset3D_spt / hipMemset3DAsync_spt
//
// The 3-D memsets are no-op on replay, for the plain entry points as much as the
// per-thread ones, so this is capture-path coverage on a scratch pitched
// allocation with the validated buffer written separately.
// Final blob: 0x45454545.
// ===========================================================================
TEST_CASE("Unit_HRR_Memset3DSpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x45454545;

  // ---- APIs under test, on a scratch allocation ---------------------------
  {
    hipPitchedPtr pp3{};
    hipExtent     ext3 = make_hipExtent(16 * sizeof(int), 4, 2);
    HIP_CHECK(hipMalloc3D(&pp3, ext3));
    HIP_CHECK(hipMemset3D(pp3, 0x45, ext3));
    HIP_CHECK(hipMemset3DAsync(pp3, 0x00, ext3, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(pp3.ptr));
  }

  // ---- Replay oracle -----------------------------------------------------
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, SZ));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), VAL, N));
  HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HIP_CHECK(hipMemcpy(h, d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HIP_CHECK(hipFree(d));
  delete[] h;
}

// ===========================================================================
// Workload: hipMemcpy3D_spt / hipMemcpy3DAsync_spt
//
// Unlike the 1-D and 2-D per-thread copies these two have faithful replay
// handlers, so the effect is validated directly: each copies a filled source
// over a destination pre-filled with a different pattern, and each destination
// has its own D2H.  A regressed handler leaves its destination holding the
// replayed pre-fill instead of the copied pattern and fails that buffer.
// Final blobs: 0x46464646 in both destinations, over a 0x4A4A4A4A pre-fill.
// ===========================================================================
TEST_CASE("Unit_HRR_Memcpy3DSpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N       = 256;
  constexpr size_t SZ      = N * sizeof(int);
  constexpr int    VAL     = 0x46464646;
  constexpr int    PREFILL = 0x4A4A4A4A;

  // 3-D geometry covering the whole linear buffer: one slice of ROWS rows.
  constexpr size_t PITCH = 128;
  constexpr size_t ROWS  = SZ / PITCH;
  static_assert(PITCH * ROWS == SZ, "3-D copy must cover the whole buffer");

  int* d_src       = nullptr;
  int* d_dst       = nullptr;
  int* d_dst_async = nullptr;
  HIP_CHECK(hipMalloc(&d_src, SZ));
  HIP_CHECK(hipMalloc(&d_dst, SZ));
  HIP_CHECK(hipMalloc(&d_dst_async, SZ));

  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_src), VAL, N));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_dst), PREFILL, N));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_dst_async), PREFILL, N));
  HIP_CHECK(hipDeviceSynchronize());

  // ---- APIs under test ---------------------------------------------------
  {
    hipMemcpy3DParms p{};
    p.srcPtr = make_hipPitchedPtr(d_src, PITCH, PITCH, ROWS);
    p.dstPtr = make_hipPitchedPtr(d_dst, PITCH, PITCH, ROWS);
    p.extent = make_hipExtent(PITCH, ROWS, 1);
    p.kind   = hipMemcpyDeviceToDevice;
    HIP_CHECK(hipMemcpy3D(&p));

    p.dstPtr = make_hipPitchedPtr(d_dst_async, PITCH, PITCH, ROWS);
    HIP_CHECK(hipMemcpy3DAsync(&p, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  // One D2H blob per API under test.
  int* h = new int[N]();
  HIP_CHECK(hipMemcpy(h, d_dst, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);
  HIP_CHECK(hipMemcpy(h, d_dst_async, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipFree(d_dst_async));
  delete[] h;
}

// ===========================================================================
// Workload: per-thread-default-stream stream and event queries
//
// Exercises hipStreamIsCapturing_spt, hipStreamQuery_spt,
// hipStreamSynchronize_spt, hipStreamGetPriority_spt, hipStreamGetFlags_spt and
// hipEventRecord_spt.  None of these writes device memory, so there is nothing
// for a D2H to prove about them beyond the recorded ids and a replay that runs
// to completion; the validated buffer exists to give the roundtrip an oracle at
// all.  The tolerant checks mirror the plain variants elsewhere in the suite,
// where an unsupported query is a legitimate answer rather than a failure.
// Final blob: 0x4C4C4C4C.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamQuerySpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x4C4C4C4C;

  // ---- APIs under test ---------------------------------------------------
  // A null stream keeps the redirect observable: an explicitly created stream is
  // passed straight through and would make these identical to the plain calls.
  {
    hipError_t             e;
    hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
    e = hipStreamIsCapturing(nullptr, &st);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    e = hipStreamQuery(nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorNotReady || e == hipErrorNotSupported));
    e = hipStreamSynchronize(nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    int prio = 0;
    e = hipStreamGetPriority(nullptr, &prio);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    unsigned int fl = 0;
    e = hipStreamGetFlags(nullptr, &fl);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
  }

  {
    hipEvent_t ev;
    HIP_CHECK(hipEventCreate(&ev));
    hipError_t e = hipEventRecord(ev, nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    HIP_CHECK(hipEventSynchronize(ev));
    HIP_CHECK(hipEventDestroy(ev));
  }

  HIP_CHECK(hipDeviceSynchronize());

  // ---- Replay oracle -----------------------------------------------------
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, SZ));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), VAL, N));
  HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HIP_CHECK(hipMemcpy(h, d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HIP_CHECK(hipFree(d));
  delete[] h;
}

// ===========================================================================
// Workload: hipStreamWaitEvent_spt cross-stream ordering
//
// The handler translates both the stream and the event handle, so the effect is
// validated directly: stream s1 waiting on an event recorded on s0 after a
// memset has to be reproduced for the copy chain to carry the pattern through to
// the readback.  Both streams are explicit here because the point is ordering
// between two of them; the redirect still applies, and the recorded id is what
// the roundtrip checks.
// Final blob: 0x42424242.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamWaitEventSpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x42424242;

  hipStream_t s0, s1;
  HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  hipEvent_t ev;
  HIP_CHECK(hipEventCreate(&ev));

  int *d0, *d1;
  HIP_CHECK(hipMalloc(&d0, SZ));
  HIP_CHECK(hipMalloc(&d1, SZ));

  HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d0), VAL, N, s0));
  HIP_CHECK(hipEventRecord(ev, s0));

  // API under test.
  HIP_CHECK(hipStreamWaitEvent(s1, ev, 0));

  HIP_CHECK(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(d1),
                               reinterpret_cast<hipDeviceptr_t>(d0), SZ, s1));
  int* h = new int[N]();
  HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d1), SZ, s1));
  HIP_CHECK(hipStreamSynchronize(s1));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HIP_CHECK(hipFree(d0));
  HIP_CHECK(hipFree(d1));
  HIP_CHECK(hipEventDestroy(ev));
  HIP_CHECK(hipStreamDestroy(s0));
  HIP_CHECK(hipStreamDestroy(s1));
  delete[] h;
}

// ===========================================================================
// Graph workloads: why the capture frame keeps the plain entry points
//
// HRR replays a graph only when it can find it in PlaybackContext::graph_map,
// and one place populates that map: the hand-written hipStreamEndCapture
// handler, which records the graph against the captured handle.  The generated
// hipStreamEndCapture_spt handler discards the graph it produces, so a frame
// closed through the redirected name leaves hipGraphInstantiate with nothing to
// instantiate and replay aborts.
//
// The frame therefore has to reach the plain entry points, which means undoing
// the redirect for those two names.  The order is load-bearing:
//
//   1. hipStreamEndCapture first, leaving hipStreamBeginCapture redirected so
//      Unit_HRR_StreamCaptureBeginSpt_Direct can still reach
//      hipStreamBeginCapture_spt, the API it exists to test.
//   2. hipStreamBeginCapture afterwards, for the two workloads whose APIs under
//      test sit inside the frame rather than opening it.
//
// hipStreamBeginCapture_spt's own handler does not set ctx.in_graph_capture,
// which gates timing, sync-after-launch and the zero-init memsets.  None of
// those apply to a memset-only frame with no allocation or kernel launch in it,
// so step 1 leaves a correct replay.
// ===========================================================================
#undef hipStreamEndCapture

#if !defined(hipStreamBeginCapture)
#error "Unit_HRR_StreamCaptureBeginSpt_Direct needs hipStreamBeginCapture still redirected"
#endif
#if defined(hipStreamEndCapture)
#error "the graph capture frame needs the plain hipStreamEndCapture"
#endif

// ===========================================================================
// Workload: hipStreamBeginCapture_spt
//
// The API under test opens the frame and the plain end capture closes it, so a
// regressed handler shows up as a graph that never replays rather than as a
// tolerated no-op.
// Final blob: 0x4A4A4A4A.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamCaptureBeginSpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x4A4A4A4A;

  hipStream_t s;
  HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, SZ));

  HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));
  hipGraph_t g = nullptr;
  HIP_CHECK(hipStreamEndCapture(s, &g));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec, s));
  HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d), SZ, s));
  HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(g));
  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

#undef hipStreamBeginCapture

#if defined(hipStreamBeginCapture) || defined(hipStreamEndCapture)
#error "the workloads below need the plain stream-capture frame"
#endif
#if !defined(hipGraphLaunch) || !defined(hipStreamIsCapturing) || \
    !defined(hipStreamGetCaptureInfo)
#error "the workloads below still test hipGraphLaunch_spt and the _spt capture queries"
#endif

// ===========================================================================
// Workload: hipGraphLaunch_spt
//
// The graph is built through the supported stream-capture path and launched via
// the redirected name, whose handler translates the graph-exec handle recorded
// by the hand-written hipGraphInstantiate handler.
// Final blob: 0x44444444.
// ===========================================================================
TEST_CASE("Unit_HRR_GraphLaunchSpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x44444444;

  hipStream_t s;
  HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, SZ));

  HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));
  hipGraph_t g = nullptr;
  HIP_CHECK(hipStreamEndCapture(s, &g));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));

  // API under test.
  HIP_CHECK(hipGraphLaunch(exec, s));
  HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d), SZ, s));
  HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(g));
  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload: hipStreamIsCapturing_spt / hipStreamGetCaptureInfo_spt
//
// The per-thread capture queries run inside a plain begin/end-capture frame.
// Their handlers only translate the stream and write to replay-local outputs, so
// the frame around them is what makes the captured memset replayable as a graph
// and the readback below meaningful; the live REQUIREs on the reported status are
// what prove the queries observed the frame they were called in.
// Final blob: 0x5A5A5A5A.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamCaptureQuerySpt_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x5A5A5A5A;

  hipStream_t s;
  HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, SZ));

  HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));

  // APIs under test.
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  HIP_CHECK(hipStreamIsCapturing(s, &st));
  REQUIRE(st == hipStreamCaptureStatusActive);

  hipStreamCaptureStatus st2   = hipStreamCaptureStatusNone;
  unsigned long long     capId = 0;
  HIP_CHECK(hipStreamGetCaptureInfo(s, &st2, &capId));
  REQUIRE(st2 == hipStreamCaptureStatusActive);

  HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));
  hipGraph_t g = nullptr;
  HIP_CHECK(hipStreamEndCapture(s, &g));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec, s));
  HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d), SZ, s));
  HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(g));
  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

/**
 * End doxygen group HRRTest.
 * @}
 */
