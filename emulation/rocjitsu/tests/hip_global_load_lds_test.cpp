// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_global_load_lds_test.cpp
/// @brief GLOBAL_LOAD_LDS_DWORDX4 (direct global->LDS load) on a simulated CDNA4 GPU.
///
/// CDNA4 ISA 10.3 defines the destination as
///
///     LDS_ADDR = LDSbase(hw alloc) + LDSoffset(M0[17:2] * 4)
///                + INST.OFFSET + ThreadID * stride
///
/// Two cases are needed to cover that expression: M0 = 0 exercises the base
/// and the per-lane stride, and M0 = 64 exercises the offset term, which is
/// where this differs from the MUBUF buffer-load-to-LDS form (ISA 9.1.9 takes
/// a raw byte offset from M0[17:0] rather than masking and dword-aligning).
/// A wrong mask or stride still produces a running kernel, just one that
/// silently shuffles data between lanes, so the checks below report the
/// offending lane rather than only a pass/fail.
///
/// Compiled with amdclang++ -x hip for gfx950 and run through the CLI
/// launcher.

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

// ROCR keeps process-lifetime runtime state allocated after hipDeviceReset().
// Ignore only allocations whose stack includes the external HSA runtime while
// retaining LeakSanitizer coverage for rocjitsu and this test executable.
extern "C" const char *__lsan_default_suppressions() { return "leak:libhsa-runtime64.so\n"; }

// ROCR serializes its async-event pool with rocr::HybridMutex, an atomic CAS
// spinlock. The external HSA runtime is not instrumented, so ThreadSanitizer
// cannot observe that happens-before edge and reports the pool's heap reuse as
// a race. Ignore only interceptors called from the HSA runtime while retaining
// ThreadSanitizer coverage for rocjitsu and this test executable.
extern "C" const char *__tsan_default_suppressions() {
  return "called_from_lib:libhsa-runtime64.so\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

constexpr int kLanes = 64;
constexpr int kFloats = kLanes * 4; ///< One float4 (one DWORDX4) per lane.
constexpr int kM0Bytes = 64;        ///< Must be a multiple of 4: M0[1:0] is dropped.
constexpr int kM0Floats = kM0Bytes / 4;
constexpr float kCanary = -7.0f;

/// The LDS array must be dynamic. With a static __shared__ array the compiler
/// sees no write to it -- an asm "memory" clobber is not enough -- emits
/// group_segment_fixed_size 0 and deletes the array along with every store
/// that depends on it, leaving a kernel of just the asm block and s_endpgm.
/// The test then passes while checking nothing.
extern "C" __global__ void lds_dma_x4(const float *__restrict__ src, float *__restrict__ dst) {
  extern __shared__ float tile[];
  const int tid = static_cast<int>(threadIdx.x);
  const float *p = src + tid * 4;

  asm volatile("s_mov_b32 m0, 0\n\t"
               "global_load_lds_dwordx4 %0, off\n\t"
               "s_waitcnt vmcnt(0)"
               :
               : "v"(p)
               : "memory");
  __syncthreads();

  for (int i = 0; i < 4; ++i)
    dst[tid * 4 + i] = tile[tid * 4 + i];
}

/// Same, with M0 = 64, so the destination shifts by M0[17:2]*4 = 64 bytes.
/// The region below that offset is pre-filled and re-checked: it must survive,
/// which catches an M0 term that is dropped, scaled wrongly, or applied to the
/// memory address instead of the LDS address.
extern "C" __global__ void lds_dma_x4_m0(const float *__restrict__ src, float *__restrict__ dst,
                                         int *__restrict__ canary_ok) {
  extern __shared__ float tile[];
  const int tid = static_cast<int>(threadIdx.x);
  const float *p = src + tid * 4;

  for (int i = tid; i < kM0Floats; i += kLanes)
    tile[i] = kCanary;
  __syncthreads();

  asm volatile("s_mov_b32 m0, 64\n\t"
               "global_load_lds_dwordx4 %0, off\n\t"
               "s_waitcnt vmcnt(0)"
               :
               : "v"(p)
               : "memory");
  __syncthreads();

  for (int i = 0; i < 4; ++i)
    dst[tid * 4 + i] = tile[kM0Floats + tid * 4 + i];

  if (tid == 0) {
    int ok = 1;
    for (int i = 0; i < kM0Floats; ++i)
      if (tile[i] != kCanary)
        ok = 0;
    *canary_ok = ok;
  }
}

std::vector<float> make_source() {
  std::vector<float> src(kFloats);
  for (int i = 0; i < kFloats; ++i)
    src[i] = static_cast<float>(i) * 1.5f + 1.0f;
  return src;
}

void expect_lane_wise_equal(const std::vector<float> &got, const std::vector<float> &want) {
  int bad = 0;
  for (int i = 0; i < kFloats; ++i) {
    if (got[i] == want[i])
      continue;
    ++bad;
    // Only the first few are worth printing; a stride error breaks every lane.
    if (bad <= 4)
      EXPECT_EQ(got[i], want[i])
          << "lane " << (i / 4) << " element " << (i % 4) << " (flat index " << i << ")";
  }
  EXPECT_EQ(bad, 0) << bad << "/" << kFloats << " elements differ";
}

} // namespace

TEST(HipGlobalLoadLdsTest, Dwordx4WritesLdsAtThreadStride) {
  const std::vector<float> h_src = make_source();
  std::vector<float> h_dst(kFloats, -1.0f);

  float *d_src = nullptr, *d_dst = nullptr;
  HIP_ASSERT(hipMalloc(&d_src, kFloats * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_dst, kFloats * sizeof(float)));
  HIP_ASSERT(hipMemcpy(d_src, h_src.data(), kFloats * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_dst, h_dst.data(), kFloats * sizeof(float), hipMemcpyHostToDevice));

  lds_dma_x4<<<1, kLanes, kFloats * sizeof(float)>>>(d_src, d_dst);
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(h_dst.data(), d_dst, kFloats * sizeof(float), hipMemcpyDeviceToHost));

  expect_lane_wise_equal(h_dst, h_src);

  (void)hipFree(d_src);
  (void)hipFree(d_dst);
}

TEST(HipGlobalLoadLdsTest, Dwordx4AppliesM0LdsOffset) {
  const std::vector<float> h_src = make_source();
  std::vector<float> h_dst(kFloats, -1.0f);
  int h_canary_ok = 0;

  float *d_src = nullptr, *d_dst = nullptr;
  int *d_canary_ok = nullptr;
  HIP_ASSERT(hipMalloc(&d_src, kFloats * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_dst, kFloats * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_canary_ok, sizeof(int)));
  HIP_ASSERT(hipMemcpy(d_src, h_src.data(), kFloats * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_dst, h_dst.data(), kFloats * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_canary_ok, &h_canary_ok, sizeof(int), hipMemcpyHostToDevice));

  lds_dma_x4_m0<<<1, kLanes, (kM0Floats + kFloats) * sizeof(float)>>>(d_src, d_dst, d_canary_ok);
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(h_dst.data(), d_dst, kFloats * sizeof(float), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(&h_canary_ok, d_canary_ok, sizeof(int), hipMemcpyDeviceToHost));

  EXPECT_EQ(h_canary_ok, 1) << "LDS below the M0 offset was overwritten, so the "
                               "M0[17:2]*4 term was not applied as the ISA specifies";
  expect_lane_wise_equal(h_dst, h_src);

  (void)hipFree(d_src);
  (void)hipFree(d_dst);
  (void)hipFree(d_canary_ok);
}
