/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Guards the gfx1250 LL/LL128 comm-FIFO system-scope load+store pairing
// (RCCL_LL_FIFO_SYS_SCOPE). A cacheable FIFO between sibling DPX partitions
// hangs at slot reuse (NCCL_STEPS=8, first hang on the 9th op) unless both the
// flag poll and the FIFO store use system-scope b128.
//
//   Gfx1250EnablesSysScope     — device-side guard check (1 GPU); verifies the
//                                macro is 1 on gfx1250 (default build), 0 elsewhere
//   FifoLineSysScopeRoundtrip  — b128 store+load intrinsic sanity check (1 GPU);
//                                mirrors prims_ll storeLL/loadLLLineB128 shape
//   Gfx1250SelectsCooperative  — device-side guard check (1 GPU); verifies the
//                                LL128 FIFO uses the cooperative 8x16B builtins
//                                on gfx1250 (default build), and not elsewhere
//   CooperativeLineRoundtrip   — cooperative 8x16B store+load sanity check
//                                (1 GPU); mirrors load128Fifo/store128Fifo
//   SiblingBroadcastSlotReuse  — 64 Ring broadcasts (LL and LL128) on a sibling
//                                pair; the only behavioral test of the fix,
//                                requires gfx1250 DPX siblings (same PCI bus)

#include "DeviceTestBase.hpp"

#include "../common/ProcessIsolatedTestRunner.hpp"
#include "rccl_ptr.h"

#include <rccl/rccl.h>

#include <chrono>
#include <string>
#include <vector>

namespace RcclUnitTesting
{

namespace
{

union alignas(16) TestLLLine {
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
};

// Writes RCCL_LL_FIFO_SYS_SCOPE to out[0] and the expected value to out[1].
// Both are evaluated in the device pass, avoiding the host/device macro mismatch.
__global__ void kernelSysScopeEnabled(int* out, bool isGfx1250Device) {
  out[0] = RCCL_LL_FIFO_SYS_SCOPE;
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  out[1] = isGfx1250Device ? 1 : 0;
#else
  out[1] = 0;  // builtins disabled at build time
#endif
}

// Sanity check for the b128 intrinsics used by prims_ll storeLL/loadLLLineB128.
// This is NOT a behavioral test of the production code (the device helpers are
// template class members and cannot be called directly from a test kernel).
// SiblingBroadcastSlotReuse provides the behavioral coverage via ncclBroadcast.
__global__ void kernelFifoLineRoundtrip(TestLLLine* line, uint32_t data, uint32_t flag, int* ok) {
#if RCCL_LL_FIFO_SYS_SCOPE
  union {
    v4u v;
    uint32_t w[4];
  } st, ld;
  st.w[0] = data;
  st.w[1] = flag;
  st.w[2] = data;
  st.w[3] = flag;
  __builtin_amdgcn_global_store_b128((v4u_gptr)line, st.v, RCCL_SYSTEM_SYNCSCOPE);
  ld.v = __builtin_amdgcn_global_load_b128((v4u_gptr)line, RCCL_SYSTEM_SYNCSCOPE);
  *ok = (ld.w[0] == data && ld.w[1] == flag && ld.w[2] == data && ld.w[3] == flag) ? 1 : 0;
#else
  line->v[0] = (uint64_t)data | ((uint64_t)flag << 32);
  line->v[1] = (uint64_t)data | ((uint64_t)flag << 32);
  *ok = (line->data1 == data && line->flag1 == flag && line->data2 == data && line->flag2 == flag)
          ? 1
          : 0;
#endif
}

// Writes RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS to out[0] and the expected value to out[1].
__global__ void kernelCooperativeEnabled(int* out, bool isGfx1250Device) {
  out[0] = RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS;
#if defined(COOPERATIVE_ATOMIC_INTRINSICS_FORCE_OFF)
  out[1] = 0;  // builtins disabled at build time
#else
  out[1] = isGfx1250Device ? 1 : 0;
#endif
}

constexpr int kCoopLanes = 64;  // whole waves: every lane of each 8-lane group participates

// Sanity check for the cooperative 8x16B builtins used by load128Fifo/store128Fifo
// (same reason as kernelFifoLineRoundtrip for not calling them directly). Lane i
// owns 16 bytes at lines[2*i], so each 8-lane group covers one 128-byte line.
__global__ void kernelCooperativeLineRoundtrip(uint64_t* lines, int* ok) {
  const uint32_t i = threadIdx.x;
  const uint64_t lo = 0xA5A5A5A500000000ull | i;
  const uint64_t hi = ((uint64_t)(i + 9) << 32) | 0x5A5A5A5Au;
  uint64_t* line = lines + 2 * i;
  rcclB128 u;
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
  u.u64[0] = lo;
  u.u64[1] = hi;
  __builtin_amdgcn_cooperative_atomic_store_8x16B(
    (v4i_gptr)line, u.vi, __ATOMIC_RELAXED, RCCL_SYSTEM_SYNCSCOPE);
  u.vi = rcclIdentity(__builtin_amdgcn_cooperative_atomic_load_8x16B(
    (v4i_gptr)line, __ATOMIC_RELAXED, RCCL_SYSTEM_SYNCSCOPE));
#else
  line[0] = lo;
  line[1] = hi;
  u.u64[0] = line[0];
  u.u64[1] = line[1];
#endif
  ok[i] = (u.u64[0] == lo && u.u64[1] == hi) ? 1 : 0;
}

// Reloads the lines written by kernelCooperativeLineRoundtrip and masks them.
// Guards rcclIdentity: before ROCm 10.2, this expression on the raw builtin
// result crashes instruction selection. It only does so when the mask is the
// load's sole use, hence a separate kernel.
constexpr uint64_t kCoopMasked = 0xA5 + 0x5A5A;
__global__ void kernelCooperativeLineMask(const uint64_t* lines, uint64_t* masked) {
  const uint32_t i = threadIdx.x;
  const uint64_t* line = lines + 2 * i;
  rcclB128 u;
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
  u.vi = rcclIdentity(__builtin_amdgcn_cooperative_atomic_load_8x16B(
    (v4i_gptr)line, __ATOMIC_RELAXED, RCCL_SYSTEM_SYNCSCOPE));
#else
  u.u64[0] = line[0];
  u.u64[1] = line[1];
#endif
  masked[i] = ((u.u64[0] >> 40) & 0xff) + ((u.u64[1] & 0xffff0000ull) >> 16);
}

bool isGfx1250(int device)
{
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
  return std::string(prop.gcnArchName).rfind("gfx1250", 0) == 0;
}

// Sibling DPX partitions share PCI domain:bus (writeup / hang matrix).
// Returns logical device indices within the current HIP visibility.
bool findSiblingPair(int* devA, int* devB)
{
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return false;
  if (n < 2) return false;
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t pi{};
    if (hipGetDeviceProperties(&pi, i) != hipSuccess) continue;
    for (int j = i + 1; j < n; ++j) {
      hipDeviceProp_t pj{};
      if (hipGetDeviceProperties(&pj, j) != hipSuccess) continue;
      if (pi.pciDomainID == pj.pciDomainID && pi.pciBusID == pj.pciBusID &&
          std::string(pi.gcnArchName).rfind("gfx1250", 0) == 0 &&
          std::string(pj.gcnArchName).rfind("gfx1250", 0) == 0) {
        *devA = i;
        *devB = j;
        return true;
      }
    }
  }
  return false;
}

void runSiblingBroadcastSlotReuse(int devA, int devB);

// Factory for sibling broadcast test configs. The parent TEST body gates on
// findSiblingPair so the skip is visible in the report; the child just runs.
ProcessIsolatedTestRunner::TestConfig makeSiblingBroadcastConfig(const char* name,
                                                                  const char* proto) {
  return ProcessIsolatedTestRunner::TestConfig(name, []() {
           int a = 0, b = 1;
           ASSERT_TRUE(findSiblingPair(&a, &b)) << "parent should have skipped";
           runSiblingBroadcastSlotReuse(a, b);
         })
    .withEnvironment({
      {"NCCL_PROTO", proto},
      {"NCCL_ALGO", "Ring"},
      {"RCCL_DDA_ENABLE", "0"},
      {"NCCL_IB_DISABLE", "1"},
      {"NCCL_SOCKET_IFNAME", "lo"},
    })
    .withTimeout(std::chrono::seconds(180))
    .withNumGpus(2);
}

void runSiblingBroadcastSlotReuse(int devA, int devB) {
  const int devs[2] = {devA, devB};
  ncclComm_t comms[2] = {};
  ASSERT_EQ(ncclCommInitAll(comms, 2, devs), ncclSuccess);

  constexpr int kCount = 256; // 1 KiB float broadcast; one LL step per coll
  constexpr int kIters = 64;  // comfortably past FIFO slot reuse (NCCL_STEPS == 8)
  float* send[2] = {};
  float* recv[2] = {};
  hipStream_t streams[2] = {};

  for (int r = 0; r < 2; ++r) {
    ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
    ASSERT_EQ(hipMalloc(&send[r], kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&recv[r], kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&streams[r]), hipSuccess);
  }

  for (int iter = 0; iter < kIters; ++iter) {
    const float expected = 42.0f + iter;
    for (int r = 0; r < 2; ++r) {
      ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
      std::vector<float> h(kCount, r == 0 ? expected : -1.0f);
      ASSERT_EQ(hipMemcpy(send[r], h.data(), kCount * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess);
      ASSERT_EQ(hipMemset(recv[r], 0, kCount * sizeof(float)), hipSuccess);
    }

    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int r = 0; r < 2; ++r) {
      ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
      ASSERT_EQ(ncclBroadcast(send[r], recv[r], kCount, ncclFloat, 0, comms[r], streams[r]),
                ncclSuccess);
    }
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);

    for (int r = 0; r < 2; ++r) {
      ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
      ASSERT_EQ(hipStreamSynchronize(streams[r]), hipSuccess);
      std::vector<float> h(kCount, 0);
      ASSERT_EQ(hipMemcpy(h.data(), recv[r], kCount * sizeof(float), hipMemcpyDeviceToHost),
                hipSuccess);
      for (int i = 0; i < kCount; ++i) {
        ASSERT_EQ(h[i], expected) << "iter " << iter << " rank " << r << " elem " << i;
      }
    }
  }

  for (int r = 0; r < 2; ++r) {
    ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(streams[r]), hipSuccess);
    ASSERT_EQ(hipFree(send[r]), hipSuccess);
    ASSERT_EQ(hipFree(recv[r]), hipSuccess);
    ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
  }
}

} // namespace

class LlFifoSysScopeDeviceTest : public DeviceTestBase {};

TEST_F(LlFifoSysScopeDeviceTest, Gfx1250EnablesSysScope)
{
  DeviceBuffer<int> d_out(2);
  const bool gfx1250 = isGfx1250(0);
  kernelSysScopeEnabled<<<1, 1>>>(d_out.ptr, gfx1250);
  syncAndCheck();
  std::vector<int> h(2);
  ASSERT_EQ(hipMemcpy(h.data(), d_out.ptr, 2 * sizeof(int), hipMemcpyDeviceToHost), hipSuccess);
  const int actual = h[0];
  const int expected = h[1];
  EXPECT_EQ(actual, expected)
    << "RCCL_LL_FIFO_SYS_SCOPE mismatch: got " << actual << ", expected " << expected
    << " (gfx1250=" << gfx1250 << ")";
}

TEST_F(LlFifoSysScopeDeviceTest, FifoLineSysScopeRoundtrip)
{
  DeviceBuffer<TestLLLine> d_line(1);
  DeviceBuffer<int> d_ok(1);
  d_line.zero();
  constexpr uint32_t kData = 0xA5A5A5A5u;
  constexpr uint32_t kFlag = 9;  // first reused-slot flag (step 8 -> flag 9)
  kernelFifoLineRoundtrip<<<1, 1>>>(d_line.ptr, kData, kFlag, d_ok.ptr);
  syncAndCheck();
  EXPECT_EQ(d_ok.download(), 1);
}

TEST_F(LlFifoSysScopeDeviceTest, Gfx1250SelectsCooperative)
{
  DeviceBuffer<int> d_out(2);
  const bool gfx1250 = isGfx1250(0);
  kernelCooperativeEnabled<<<1, 1>>>(d_out.ptr, gfx1250);
  syncAndCheck();
  const std::vector<int> h = d_out.copyTo();
  EXPECT_EQ(h[0], h[1]) << "RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS mismatch: got " << h[0]
                        << ", expected " << h[1] << " (gfx1250=" << gfx1250 << ")";
}

TEST_F(LlFifoSysScopeDeviceTest, CooperativeLineRoundtrip)
{
  DeviceBuffer<uint64_t> d_lines(2 * kCoopLanes);
  DeviceBuffer<int> d_ok(kCoopLanes);
  DeviceBuffer<uint64_t> d_masked(kCoopLanes);
  d_lines.zero();
  d_ok.zero();
  d_masked.zero();
  kernelCooperativeLineRoundtrip<<<1, kCoopLanes>>>(d_lines.ptr, d_ok.ptr);
  kernelCooperativeLineMask<<<1, kCoopLanes>>>(d_lines.ptr, d_masked.ptr);
  syncAndCheck();
  const std::vector<int> ok = d_ok.copyTo();
  const std::vector<uint64_t> masked = d_masked.copyTo();
  for (int i = 0; i < kCoopLanes; ++i) {
    EXPECT_EQ(ok[i], 1) << "lane " << i;
    EXPECT_EQ(masked[i], kCoopMasked) << "lane " << i;
  }
}

TEST(LlFifoSysScope, SiblingBroadcastSlotReuse_LL)
{
  int a, b;
  if (!findSiblingPair(&a, &b)) {
    GTEST_SKIP() << "needs visible gfx1250 DPX sibling devices";
  }
  RUN_ISOLATED_TESTS(makeSiblingBroadcastConfig("LlFifoSysScope.SiblingBroadcastSlotReuse_LL", "LL"));
}

TEST(LlFifoSysScope, SiblingBroadcastSlotReuse_LL128)
{
  int a, b;
  if (!findSiblingPair(&a, &b)) {
    GTEST_SKIP() << "needs visible gfx1250 DPX sibling devices";
  }
  RUN_ISOLATED_TESTS(
    makeSiblingBroadcastConfig("LlFifoSysScope.SiblingBroadcastSlotReuse_LL128", "LL128"));
}

} // namespace RcclUnitTesting
