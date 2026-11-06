/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Per-Thread-Default-Stream Workloads
 * @{
 * @ingroup HRRTest
 * `_spt` entry points are reached by compiling with -fgpu-default-stream=per-thread, so these workloads call the ordinary names.
 * The option is set per source in CMakeLists.txt, which is why this file is its own translation unit.
 *
 * Oracle policy: a replay-validated buffer must be written and read by APIs that snapshot a blob at capture and reproduce their effect at replay.
 * Several `_spt` handlers are no-ops, so those workloads use a scratch buffer and write the validated one separately.
 *
 * Macro ordering: the redirects are undone in stages, each followed by a preprocessor check, so reordering the file breaks the build.
 */

#include "hrr_test_common.hh"

// The compile option is what makes this file meaningful; without it every call
// below resolves to the plain entry point already covered elsewhere.
#if !defined(HIP_API_PER_THREAD_DEFAULT_STREAM) && \
    !defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#error "hrr_spt_workload_test.cc requires the per-thread default stream compile option"
#endif

// Driver-style entry points are never redirected, so they stay usable as an oracle: hipMemsetD32*, hipMemcpyDtoDAsync, hipMemcpyHtoD, hipMemcpyDtoH*.
// Their capture shims or replay handlers are faithful.

__device__ int g_hrr_spt_symbol[256];

#if !defined(hipMemcpy) || !defined(hipMemcpyAsync) || !defined(hipMemcpy2D) || \
    !defined(hipMemcpy2DAsync) || !defined(hipMemcpyToSymbol) ||               \
    !defined(hipMemcpyToSymbolAsync)
#error "Unit_HRR_MemcpySpt_Direct must run before the memcpy redirects are undone"
#endif

// Exercises the six memcpy _spt names. All have no-op playback handlers, so they run on a scratch buffer freed before the validated one is written.
// The oracle is hipMemcpyHtoD / hipMemcpyDtoH, because hipMemcpy must stay redirected here. Final blob: 0x41414141.
TEST_CASE("Unit_HRR_MemcpySpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x41414141;

  constexpr size_t PITCH = 128;
  constexpr size_t ROWS  = SZ / PITCH;
  static_assert(PITCH * ROWS == SZ, "2-D copy must cover the whole buffer");

  int* h_src = new int[N];
  for (int i = 0; i < N; ++i) h_src[i] = VAL;
  int* h_chk = new int[N]();

  int* d_spt = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d_spt, SZ));

  HRR_HIP_CHECK(hipMemcpy(d_spt, h_src, SZ, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpy(h_chk, d_spt, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == VAL);

  // The async variants need a null stream or they are indistinguishable from
  // the plain ones: PER_THREAD_DEFAULT_STREAM substitutes only a null or legacy
  // stream and passes an explicitly created one straight through.
  HRR_HIP_CHECK(hipMemcpyAsync(d_spt, h_src, SZ, hipMemcpyHostToDevice, nullptr));
  HRR_HIP_CHECK(hipMemcpyAsync(h_chk, d_spt, SZ, hipMemcpyDeviceToHost, nullptr));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == VAL);

  HRR_HIP_CHECK(hipMemcpy2D(d_spt, PITCH, h_src, PITCH, PITCH, ROWS,
                        hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpy2D(h_chk, PITCH, d_spt, PITCH, PITCH, ROWS,
                        hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == VAL);

  HRR_HIP_CHECK(hipMemcpy2DAsync(d_spt, PITCH, h_src, PITCH, PITCH, ROWS,
                             hipMemcpyHostToDevice, nullptr));
  HRR_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hrr_spt_symbol), h_src, SZ, 0,
                              hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_hrr_spt_symbol), h_src, SZ, 0,
                                   hipMemcpyHostToDevice, nullptr));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipFree(d_spt));

  int* d_ok = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d_ok, SZ));
  HRR_HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(d_ok), h_src, SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h_out = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoH(h_out, reinterpret_cast<hipDeviceptr_t>(d_ok), SZ));
  for (int i = 0; i < N; ++i) REQUIRE(h_out[i] == VAL);

  HRR_HIP_CHECK(hipFree(d_ok));
  delete[] h_src;
  delete[] h_chk;
  delete[] h_out;
}

// hipMemcpy must resolve to the plain entry point: only its capture shim writes the blob a D2H read is validated against.
// The same holds for every other redirected memcpy name.
#undef hipMemcpy

#if defined(hipMemcpy)
#error "the workloads below need the plain hipMemcpy as their replay oracle"
#endif

// Each API fills its own buffer with its own byte pattern, read back by its own D2H, so a NOOP handler fails its own check.
// Patterns must decode above 1.001e-3 as f16, f32, bf16 and f64, or a zero-initialised replay buffer passes under D2H tolerance.
// Final blobs: 0x41414141 / 0x42424242 / 0x44444444 / 0x4C4C4C4C.
TEST_CASE("Unit_HRR_MemsetSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
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
  HRR_HIP_CHECK(hipMalloc(&d_set, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_set_async, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_set2d, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_set2d_async, SZ));

  HRR_HIP_CHECK(hipMemset(d_set, 0x41, SZ));
  // Omitting the stream instead, and so taking the _spt prototype's
  // hipStreamPerThread default, reaches the same stream but is deliberately not
  // exercised here: it records the sentinel handle, and replay only resolves
  // that because translate_stream returns nullptr for handles it cannot map.
  // A stricter translate_stream would turn that fallback into a failure and
  // break this test as though the memset handlers had regressed.
  HRR_HIP_CHECK(hipMemsetAsync(d_set_async, 0x42, SZ, nullptr));
  HRR_HIP_CHECK(hipMemset2D(d_set2d, PITCH, 0x44, PITCH, ROWS));
  HRR_HIP_CHECK(hipMemset2DAsync(d_set2d_async, PITCH, 0x4C, PITCH, ROWS, nullptr));
  // The per-thread default stream does not implicitly synchronise with the
  // legacy stream the readbacks below run on.
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpy(h, d_set, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x41414141);
  HRR_HIP_CHECK(hipMemcpy(h, d_set_async, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x42424242);
  HRR_HIP_CHECK(hipMemcpy(h, d_set2d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x44444444);
  HRR_HIP_CHECK(hipMemcpy(h, d_set2d_async, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0x4C4C4C4C);

  HRR_HIP_CHECK(hipFree(d_set));
  HRR_HIP_CHECK(hipFree(d_set_async));
  HRR_HIP_CHECK(hipFree(d_set2d));
  HRR_HIP_CHECK(hipFree(d_set2d_async));
  delete[] h;
}

// The 3-D memsets are no-ops on replay, plain or per-thread, so this covers capture only.
// Final blob: 0x45454545.
TEST_CASE("Unit_HRR_Memset3DSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x45454545;

  {
    hipPitchedPtr pp3{};
    hipExtent     ext3 = make_hipExtent(16 * sizeof(int), 4, 2);
    HRR_HIP_CHECK(hipMalloc3D(&pp3, ext3));
    HRR_HIP_CHECK(hipMemset3D(pp3, 0x45, ext3));
    HRR_HIP_CHECK(hipMemset3DAsync(pp3, 0x00, ext3, nullptr));
    HRR_HIP_CHECK(hipDeviceSynchronize());
    HRR_HIP_CHECK(hipFree(pp3.ptr));
  }

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), VAL, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpy(h, d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipFree(d));
  delete[] h;
}

// These two have faithful replay handlers, so each copies a filled source over a differently pre-filled destination with its own D2H.
// Final blobs: 0x46464646 in both destinations, over a 0x4A4A4A4A pre-fill.
TEST_CASE("Unit_HRR_Memcpy3DSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N       = 256;
  constexpr size_t SZ      = N * sizeof(int);
  constexpr int    VAL     = 0x46464646;
  constexpr int    PREFILL = 0x4A4A4A4A;

  constexpr size_t PITCH = 128;
  constexpr size_t ROWS  = SZ / PITCH;
  static_assert(PITCH * ROWS == SZ, "3-D copy must cover the whole buffer");

  int* d_src       = nullptr;
  int* d_dst       = nullptr;
  int* d_dst_async = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d_src, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_dst, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_dst_async, SZ));

  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_src), VAL, N));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_dst), PREFILL, N));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_dst_async), PREFILL, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  {
    hipMemcpy3DParms p{};
    p.srcPtr = make_hipPitchedPtr(d_src, PITCH, PITCH, ROWS);
    p.dstPtr = make_hipPitchedPtr(d_dst, PITCH, PITCH, ROWS);
    p.extent = make_hipExtent(PITCH, ROWS, 1);
    p.kind   = hipMemcpyDeviceToDevice;
    HRR_HIP_CHECK(hipMemcpy3D(&p));

    p.dstPtr = make_hipPitchedPtr(d_dst_async, PITCH, PITCH, ROWS);
    HRR_HIP_CHECK(hipMemcpy3DAsync(&p, nullptr));
    HRR_HIP_CHECK(hipDeviceSynchronize());
  }

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpy(h, d_dst, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);
  HRR_HIP_CHECK(hipMemcpy(h, d_dst_async, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipFree(d_src));
  HRR_HIP_CHECK(hipFree(d_dst));
  HRR_HIP_CHECK(hipFree(d_dst_async));
  delete[] h;
}

// None of these queries writes device memory, so only the recorded ids and a replay that completes are checked.
// Final blob: 0x4C4C4C4C.
TEST_CASE("Unit_HRR_StreamQuerySpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x4C4C4C4C;

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
    HRR_HIP_CHECK(hipEventCreate(&ev));
    hipError_t e = hipEventRecord(ev, nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    HRR_HIP_CHECK(hipEventSynchronize(ev));
    HRR_HIP_CHECK(hipEventDestroy(ev));
  }

  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), VAL, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpy(h, d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipFree(d));
  delete[] h;
}

// Stream s1 waiting on an event recorded on s0 after a memset must be reproduced for the copy chain to carry the pattern.
// Both streams are explicit because the point is ordering. Final blob: 0x42424242.
TEST_CASE("Unit_HRR_StreamWaitEventSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x42424242;

  hipStream_t s0, s1;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  hipEvent_t ev;
  HRR_HIP_CHECK(hipEventCreate(&ev));

  int *d0, *d1;
  HRR_HIP_CHECK(hipMalloc(&d0, SZ));
  HRR_HIP_CHECK(hipMalloc(&d1, SZ));

  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d0), VAL, N, s0));
  HRR_HIP_CHECK(hipEventRecord(ev, s0));

  HRR_HIP_CHECK(hipStreamWaitEvent(s1, ev, 0));

  HRR_HIP_CHECK(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(d1),
                               reinterpret_cast<hipDeviceptr_t>(d0), SZ, s1));
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d1), SZ, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipFree(d0));
  HRR_HIP_CHECK(hipFree(d1));
  HRR_HIP_CHECK(hipEventDestroy(ev));
  HRR_HIP_CHECK(hipStreamDestroy(s0));
  HRR_HIP_CHECK(hipStreamDestroy(s1));
  delete[] h;
}

// The graph frame keeps the plain entry points: replay finds graphs only through graph_map, which the hand-written hipStreamEndCapture handler populates.
// The generated hipStreamEndCapture_spt handler discards its graph. Order is load-bearing:
//   1. undo hipStreamEndCapture first, so hipStreamBeginCapture stays redirected for the workload that tests it;
//   2. then undo hipStreamBeginCapture, for the workloads whose APIs sit inside the frame.
// hipStreamBeginCapture_spt does not set ctx.in_graph_capture; that is harmless for a memset-only frame.
#undef hipStreamEndCapture

#if !defined(hipStreamBeginCapture)
#error "Unit_HRR_StreamCaptureBeginSpt_Direct needs hipStreamBeginCapture still redirected"
#endif
#if defined(hipStreamEndCapture)
#error "the graph capture frame needs the plain hipStreamEndCapture"
#endif

// The API under test opens the frame and the plain end capture closes it, so a regressed handler shows as a graph that never replays.
// Final blob: 0x4A4A4A4A.
TEST_CASE("Unit_HRR_StreamCaptureBeginSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x4A4A4A4A;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));
  hipGraph_t g = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(s, &g));

  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
  HRR_HIP_CHECK(hipGraphLaunch(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipGraphExecDestroy(exec));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
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

// The graph is built by stream capture and launched through the redirected name, whose handler translates the recorded graph-exec handle.
// Final blob: 0x44444444.
TEST_CASE("Unit_HRR_GraphLaunchSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x44444444;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));
  hipGraph_t g = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(s, &g));

  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));

  HRR_HIP_CHECK(hipGraphLaunch(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipGraphExecDestroy(exec));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// The queries run inside a plain begin/end-capture frame so the memset replays as a graph; the REQUIREs on the reported status prove they saw the frame.
// Final blob: 0x5A5A5A5A.
TEST_CASE("Unit_HRR_StreamCaptureQuerySpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x5A5A5A5A;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));

  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  HRR_HIP_CHECK(hipStreamIsCapturing(s, &st));
  REQUIRE(st == hipStreamCaptureStatusActive);

  hipStreamCaptureStatus st2   = hipStreamCaptureStatusNone;
  unsigned long long     capId = 0;
  HRR_HIP_CHECK(hipStreamGetCaptureInfo(s, &st2, &capId));
  REQUIRE(st2 == hipStreamCaptureStatusActive);

  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));
  hipGraph_t g = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(s, &g));

  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
  HRR_HIP_CHECK(hipGraphLaunch(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoHAsync(h, reinterpret_cast<hipDeviceptr_t>(d), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipGraphExecDestroy(exec));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

/**
 * End doxygen group HRRTest.
 * @}
 */
