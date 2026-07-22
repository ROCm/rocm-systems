/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Repro Workload (regression micro-kernels)
 * @{
 * @ingroup HRRTest
 * Direct GPU micro-kernel workloads that reproduce the specific behaviours and
 * regressions exercised by the playback fixes in this area:
 *
 *   - dropped-first-hipMalloc capture (the first HIP call is a hipMalloc),
 *   - embedded device pointers inside a by-value struct kernel argument,
 *   - an uncaptured host write that deterministically diverges replay,
 *   - the "null optional output pointer + 0x20000" GPU-fault class, and
 *   - replay zero-init of an otherwise-uninitialised device allocation.
 *
 * Like hrr_workload.cc, every TEST_CASE here is hidden with the Catch2 [.] tag
 * so it is NOT auto-discovered by CTest.  Each is driven from hrr_roundtrip.cc:
 * a parent test sets HIP_HRR_CAPTURE_OUTPUT, spawns the _Direct case as a
 * subprocess to capture, then runs hrr-playback to validate the replay.
 *
 * All divergence is injected deterministically (via an uncaptured host write),
 * never via genuine GPU nondeterminism, so the replay outcomes are reproducible.
 */

#include <hip_test_common.hh>

#include <cstdint>

// ---------------------------------------------------------------------------
// Workload parameters (kept local so this file is self-contained)
// ---------------------------------------------------------------------------

namespace {
constexpr int    kN          = 1 << 10;          // 1K floats
constexpr size_t kSZ         = kN * sizeof(float);
constexpr int    kBlock      = 256;
constexpr int    kGrid       = (kN + kBlock - 1) / kBlock;

// Divergence loop length.  Must comfortably exceed the divergence-abort
// min-samples used by the roundtrip driver so the guard trips well before the
// loop ends (and, for the null-optional case, before the faulting kernel runs).
constexpr int    kDivergeIters = 64;

// Sentinel stored into mapped host memory by the CPU (NOT a HIP call, so the
// capture layer never records it).  Chosen non-zero so a fresh (zero) replay
// allocation differs from it and reliably diverges.
constexpr int      kFlagSentinel = 7;
constexpr unsigned kSlotSentinel = 0xCAFEBABEu;
// 0x8000 floats * 4 bytes = 0x20000 — reproduces the production fault address.
constexpr int      kNullWriteIndex = 0x8000;
}  // namespace

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void hrr_repro_scale(const float* in, float* out, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * s;
}

__global__ void hrr_repro_copy(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

// By-value struct argument carrying embedded device pointers.  The capture
// layer must detect a, b, c as device pointers inside the struct payload and
// translate them at replay.
struct HrrAddArgs {
  const float* a;
  const float* b;
  float*       c;
  int          n;
};

__global__ void hrr_repro_addStruct(HrrAddArgs args) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < args.n) args.c[i] = args.a[i] + args.b[i];
}

// Writes the (host-mapped) flag value into every output element.  The flag is
// set by an uncaptured CPU store, so replay sees a different value -> the D2H
// readback diverges every iteration.
__global__ void hrr_repro_writeFlag(int* out, const int* flag, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = flag[0];
}

// slotmap-style kernel: required_out is always written (deterministic), while
// optional_out is written ONLY when the uncaptured flag differs from the
// sentinel.  optional_out is genuinely NULL at capture, so the recorded arg is
// 0x0; on replay the branch is taken and the null write faults at 0x20000.
__global__ void hrr_repro_slotmap(float* required_out, const unsigned* flag,
                                  float* optional_out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) required_out[i] = static_cast<float>(i);
  if (i == 0 && flag[0] != kSlotSentinel)
    optional_out[kNullWriteIndex] = 1.0f;
}

// ===========================================================================
// A1. First-hipMalloc capture regression
//
// Makes the VERY FIRST HIP call in the process a hipMalloc (no hipSetDevice,
// no warm-up allocation).  hip_capture_init() is deferred into hip::init(),
// which the HIP_INIT macro runs *before* the body of this first hipMalloc — so
// the allocation must be captured.  If the first-call capture ever regresses,
// `da` will not be translated at replay and the D2H validation fails.
//
// H2D(da=1) -> scale(out = da*2) -> D2H(out)  => out[i] == 2.0f
// ===========================================================================
TEST_CASE("Unit_HRR_FirstMalloc_Direct", "[.][hrr-direct]") {
  float* da = nullptr;
  HIP_CHECK(hipMalloc(&da, kSZ));  // intentionally the first HIP call
  float* dout = nullptr;
  HIP_CHECK(hipMalloc(&dout, kSZ));

  float* ha = new float[kN];
  float* hout = new float[kN];
  for (int i = 0; i < kN; ++i) ha[i] = 1.0f;

  HIP_CHECK(hipMemcpy(da, ha, kSZ, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(hrr_repro_scale, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     da, dout, 2.0f, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hout, dout, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(hout[i] == 2.0f);

  HIP_CHECK(hipFree(da));
  HIP_CHECK(hipFree(dout));
  delete[] ha;
  delete[] hout;
}

// ===========================================================================
// A2. Embedded device pointers in a by-value struct argument
//
// The kernel takes a single HrrAddArgs by value; a, b, c are device pointers
// embedded inside the struct.  Exercises embedded-pointer capture + replay
// translation (kind-3 args).
//
// a=1, b=1 -> c = a + b = 2  => c[i] == 2.0f
// ===========================================================================
TEST_CASE("Unit_HRR_EmbeddedPtrStruct_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  HIP_CHECK(hipMalloc(&da, kSZ));
  HIP_CHECK(hipMalloc(&db, kSZ));
  HIP_CHECK(hipMalloc(&dc, kSZ));

  float* ha = new float[kN];
  float* hc = new float[kN];
  for (int i = 0; i < kN; ++i) ha[i] = 1.0f;

  HIP_CHECK(hipMemcpy(da, ha, kSZ, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, ha, kSZ, hipMemcpyHostToDevice));

  HrrAddArgs args{da, db, dc, kN};
  hipLaunchKernelGGL(hrr_repro_addStruct, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     args);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hc, dc, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(hc[i] == 2.0f);

  HIP_CHECK(hipFree(da));
  HIP_CHECK(hipFree(db));
  HIP_CHECK(hipFree(dc));
  delete[] ha;
  delete[] hc;
}

// ===========================================================================
// A3. Uncaptured host write -> deterministic replay divergence
//
// A mapped host flag is written by the CPU (kFlagSentinel) — NOT a HIP call, so
// it is never recorded.  A loop of kDivergeIters kernels each write the flag
// value into an output buffer and read it back (D2H).  At capture every D2H
// blob holds kFlagSentinel; at replay the flag buffer is fresh (zero), so every
// D2H validation fails.  This is the deterministic lever for the divergence
// guard test — no genuine GPU nondeterminism is involved.
// ===========================================================================
TEST_CASE("Unit_HRR_UncapturedHostWrite_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* flag = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(int),
                          hipHostMallocMapped));
  int* flag_dev = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&flag_dev), flag, 0));
  flag[0] = kFlagSentinel;  // uncaptured CPU store

  int* dout = nullptr;
  HIP_CHECK(hipMalloc(&dout, kN * sizeof(int)));
  int* hout = new int[kN];

  for (int it = 0; it < kDivergeIters; ++it) {
    hipLaunchKernelGGL(hrr_repro_writeFlag, dim3(kGrid), dim3(kBlock), 0, nullptr,
                       dout, flag_dev, kN);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(hout, dout, kN * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < kN; ++i) REQUIRE(hout[i] == kFlagSentinel);
  }

  HIP_CHECK(hipFree(dout));
  HIP_CHECK(hipHostFree(flag));
  delete[] hout;
}

// ===========================================================================
// A4. Null optional output pointer (the "null + 0x20000" fault class)
//
// Mirrors ~/hrr-testing/slotmap-repro/slotmap_repro.cpp, but front-loads the
// uncaptured-host-write divergence loop so the divergence guard trips BEFORE
// the faulting slotmap kernel is dispatched.  At replay the guard sets the
// fatal flag during the readback loop and the replay stops cleanly (exit 2),
// so the recorded-null optional_out write at 0x20000 is never reached.
//
// At capture flag == sentinel, so the optional branch is skipped and the run is
// clean; the optional_out argument is genuinely NULL (recorded as 0x0).
// ===========================================================================
TEST_CASE("Unit_HRR_NullOptionalPtr_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  unsigned* flag = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(unsigned),
                          hipHostMallocMapped));
  unsigned* flag_dev = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&flag_dev), flag, 0));
  flag[0] = kSlotSentinel;  // uncaptured CPU store

  // Divergence source: many D2H readbacks whose value depends on the flag.
  int* dcount = nullptr;
  HIP_CHECK(hipMalloc(&dcount, kN * sizeof(int)));
  int* hcount = new int[kN];
  int* iflag_dev = reinterpret_cast<int*>(flag_dev);
  for (int it = 0; it < kDivergeIters; ++it) {
    hipLaunchKernelGGL(hrr_repro_writeFlag, dim3(kGrid), dim3(kBlock), 0, nullptr,
                       dcount, iflag_dev, kN);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(hcount, dcount, kN * sizeof(int), hipMemcpyDeviceToHost));
  }

  float* required_out = nullptr;
  HIP_CHECK(hipMalloc(&required_out, kSZ));
  float* optional_out = nullptr;  // genuinely NULL at capture (recorded as 0x0)

  hipLaunchKernelGGL(hrr_repro_slotmap, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     required_out, flag_dev, optional_out, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // D2H readback of the always-valid output so capture records a blob.
  float* hreq = new float[kN];
  HIP_CHECK(hipMemcpy(hreq, required_out, kSZ, hipMemcpyDeviceToHost));
  REQUIRE(hreq[1] == 1.0f);

  HIP_CHECK(hipFree(dcount));
  HIP_CHECK(hipFree(required_out));
  HIP_CHECK(hipHostFree(flag));
  delete[] hcount;
  delete[] hreq;
}

// ===========================================================================
// A5. Zero-init read of an uninitialised device allocation
//
// hipMalloc a buffer and do NOT initialise it; a kernel copies it to `out`,
// then D2H.  Fresh ROCm allocations are zeroed, so at capture out == 0 and the
// recorded blob is all-zero.  Replay with HIP_HRR_REPLAY_ZERO_INIT=1 reproduces
// the zeroed source deterministically, so the replayed out matches; with the
// knob off, replay may reuse stale bytes and diverge.
// ===========================================================================
TEST_CASE("Unit_HRR_ZeroInitRead_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  float* dsrc = nullptr;  // intentionally never written
  HIP_CHECK(hipMalloc(&dsrc, kSZ));
  float* dout = nullptr;
  HIP_CHECK(hipMalloc(&dout, kSZ));

  hipLaunchKernelGGL(hrr_repro_copy, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     dsrc, dout, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  float* hout = new float[kN];
  HIP_CHECK(hipMemcpy(hout, dout, kSZ, hipMemcpyDeviceToHost));
  // Precondition: fresh device allocations are zeroed by the driver.  This is
  // what replay zero-init reproduces; assert it so a non-zeroing platform fails
  // here (clearly) rather than flaking the roundtrip.
  for (int i = 0; i < kN; ++i) REQUIRE(hout[i] == 0.0f);

  HIP_CHECK(hipFree(dsrc));
  HIP_CHECK(hipFree(dout));
  delete[] hout;
}

/**
 * @}
 */
