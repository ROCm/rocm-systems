/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Repro Workload (regression micro-kernels)
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"

#ifndef _WIN32
#include <hsa/hsa_ext_amd.h>
#endif

#include <cstdint>


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


__global__ void hrr_repro_scale(const float* in, float* out, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * s;
}

__global__ void hrr_repro_copy(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

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

// The first HIP call is a hipMalloc. hip_capture_init() is deferred into hip::init(), which runs before the body.
// The allocation must still be captured, or `da` is not translated at replay and D2H validation fails.
TEST_CASE("Unit_HRR_FirstMalloc_Direct", "[.][hrr-direct]") {
  float* da = nullptr;
  HRR_HIP_CHECK(hipMalloc(&da, kSZ));  // intentionally the first HIP call
  float* dout = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dout, kSZ));

  float* ha = new float[kN];
  float* hout = new float[kN];
  for (int i = 0; i < kN; ++i) ha[i] = 1.0f;

  HRR_HIP_CHECK(hipMemcpy(da, ha, kSZ, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(hrr_repro_scale, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     da, dout, 2.0f, kN);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipMemcpy(hout, dout, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(hout[i] == 2.0f);

  HRR_HIP_CHECK(hipFree(da));
  HRR_HIP_CHECK(hipFree(dout));
  delete[] ha;
  delete[] hout;
}

TEST_CASE("Unit_HRR_EmbeddedPtrStruct_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  HRR_HIP_CHECK(hipMalloc(&da, kSZ));
  HRR_HIP_CHECK(hipMalloc(&db, kSZ));
  HRR_HIP_CHECK(hipMalloc(&dc, kSZ));

  float* ha = new float[kN];
  float* hc = new float[kN];
  for (int i = 0; i < kN; ++i) ha[i] = 1.0f;

  HRR_HIP_CHECK(hipMemcpy(da, ha, kSZ, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpy(db, ha, kSZ, hipMemcpyHostToDevice));

  HrrAddArgs args{da, db, dc, kN};
  hipLaunchKernelGGL(hrr_repro_addStruct, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     args);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipMemcpy(hc, dc, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(hc[i] == 2.0f);

  HRR_HIP_CHECK(hipFree(da));
  HRR_HIP_CHECK(hipFree(db));
  HRR_HIP_CHECK(hipFree(dc));
  delete[] ha;
  delete[] hc;
}

// A mapped host flag is written by the CPU, which is not a HIP call and so is never recorded.
// At replay the flag buffer is fresh (zero), so every D2H validation fails: the lever for the divergence guard test.
TEST_CASE("Unit_HRR_UncapturedHostWrite_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* flag = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(int),
                          hipHostMallocMapped));
  int* flag_dev = nullptr;
  HRR_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&flag_dev), flag, 0));
  flag[0] = kFlagSentinel;  // uncaptured CPU store

  int* dout = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dout, kN * sizeof(int)));
  int* hout = new int[kN];

  for (int it = 0; it < kDivergeIters; ++it) {
    hipLaunchKernelGGL(hrr_repro_writeFlag, dim3(kGrid), dim3(kBlock), 0, nullptr,
                       dout, flag_dev, kN);
    HRR_HIP_CHECK(hipGetLastError());
    HRR_HIP_CHECK(hipDeviceSynchronize());
    HRR_HIP_CHECK(hipMemcpy(hout, dout, kN * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < kN; ++i) REQUIRE(hout[i] == kFlagSentinel);
  }

  HRR_HIP_CHECK(hipFree(dout));
  HRR_HIP_CHECK(hipHostFree(flag));
  delete[] hout;
}

// Mirrors slotmap_repro.cpp but runs the divergence loop first, so the guard stops replay (exit 2) before the faulting kernel is dispatched.
// At capture the optional branch is skipped; optional_out is genuinely NULL (recorded as 0x0).
TEST_CASE("Unit_HRR_NullOptionalPtr_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  unsigned* flag = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(unsigned),
                          hipHostMallocMapped));
  unsigned* flag_dev = nullptr;
  HRR_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&flag_dev), flag, 0));
  flag[0] = kSlotSentinel;  // uncaptured CPU store

  int* dcount = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dcount, kN * sizeof(int)));
  int* hcount = new int[kN];
  int* iflag_dev = reinterpret_cast<int*>(flag_dev);
  for (int it = 0; it < kDivergeIters; ++it) {
    hipLaunchKernelGGL(hrr_repro_writeFlag, dim3(kGrid), dim3(kBlock), 0, nullptr,
                       dcount, iflag_dev, kN);
    HRR_HIP_CHECK(hipGetLastError());
    HRR_HIP_CHECK(hipDeviceSynchronize());
    HRR_HIP_CHECK(hipMemcpy(hcount, dcount, kN * sizeof(int), hipMemcpyDeviceToHost));
  }

  float* required_out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&required_out, kSZ));
  float* optional_out = nullptr;  // genuinely NULL at capture (recorded as 0x0)

  hipLaunchKernelGGL(hrr_repro_slotmap, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     required_out, flag_dev, optional_out, kN);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  float* hreq = new float[kN];
  HRR_HIP_CHECK(hipMemcpy(hreq, required_out, kSZ, hipMemcpyDeviceToHost));
  REQUIRE(hreq[1] == 1.0f);

  HRR_HIP_CHECK(hipFree(dcount));
  HRR_HIP_CHECK(hipFree(required_out));
  HRR_HIP_CHECK(hipHostFree(flag));
  delete[] hcount;
  delete[] hreq;
}

// HRR records HIP calls only, so replay sees the allocation but not the HSA zero fill.
// It must reproduce the zeros through HIP_HRR_REPLAY_ZERO_INIT, without recording a hipMemset that playback would run independently.
TEST_CASE("Unit_HRR_ZeroInitRead_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  float* dsrc = nullptr;
#ifndef _WIN32
  HRR_HIP_CHECK(hipMalloc(&dsrc, kSZ));
  // hsa_amd_memory_fill() counts in 32-bit words, not elements.  kN happens to
  // be the same number here only because kSZ is float[kN]; spell out the unit
  // so a change of element type cannot silently under- or over-fill.
  REQUIRE(hsa_amd_memory_fill(dsrc, 0, kSZ / sizeof(uint32_t)) == HSA_STATUS_SUCCESS);
#else
  // Native Windows has no ROCr/HSA runtime. Keep its deterministic oracle via
  // host stores; Linux retains coverage of the ordinary hipMalloc replay path.
  HRR_HIP_CHECK(hipMallocManaged(&dsrc, kSZ));
  for (int i = 0; i < kN; ++i) dsrc[i] = 0.0f;
#endif
  float* dout = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dout, kSZ));

  hipLaunchKernelGGL(hrr_repro_copy, dim3(kGrid), dim3(kBlock), 0, nullptr,
                     dsrc, dout, kN);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  float* hout = new float[kN];
  HRR_HIP_CHECK(hipMemcpy(hout, dout, kSZ, hipMemcpyDeviceToHost));
  // Precondition: fresh device allocations are zeroed by the driver. This is
  // what replay zero-init reproduces; assert it so a non-zeroing platform fails
  // here (clearly) rather than flaking the roundtrip.
  //
  // But with ASAN enabled this won't be true, because doing so would
  // add significant performance overhead and hide bugs related to uninitialized data
#if !defined(ENABLE_ADDRESS_SANITIZER)
  for (int i = 0; i < kN; ++i) REQUIRE(hout[i] == 0.0f);
#endif

  HRR_HIP_CHECK(hipFree(dsrc));
  HRR_HIP_CHECK(hipFree(dout));
  delete[] hout;
}

/**
 * @}
 */
