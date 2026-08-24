/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for comm-FIFO and registered-user-buffer 128-bit helpers in op128.h.

#include "DeviceTestBase.hpp"

#include <string>

#include "op128.h"

namespace RcclUnitTesting
{

namespace
{

static bool deviceArchIsGfx1250() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return false;
  return std::string(prop.gcnArchName).rfind("gfx1250", 0) == 0;
}

} // namespace

class LL128VectorTest : public DeviceTestBase
{
};

__global__ void kernelLL128Load128NTStore128Plain(const uint64_t* src, uint64_t* dst) {
  uint64_t v0, v1;
  load128NT(src, v0, v1);
  store128Plain(dst, v0, v1);
}

TEST_F(LL128VectorTest, Load128NTStore128PlainRoundtrip) {
  const uint64_t h_src[2] = {0xAAAABBBBCCCCDDDDULL, 0x1111222233334444ULL};
  DeviceBuffer<uint64_t> d_src(2), d_dst(2);
  d_src.copyFrom(h_src, 2);

  kernelLL128Load128NTStore128Plain<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  auto h_dst = d_dst.copyTo();
  EXPECT_EQ(h_dst[0], h_src[0]);
  EXPECT_EQ(h_dst[1], h_src[1]);
}

__global__ void kernelLL128Load128Store128(const uint64_t* src, uint64_t* dst) {
  uint64_t v0, v1;
  load128(src, v0, v1);
  store128(dst, v0, v1);
}

TEST_F(LL128VectorTest, Load128Store128Gfx1250CooperativePath) {
#if !__has_builtin(__builtin_amdgcn_cooperative_atomic_load_8x16B)
  GTEST_SKIP() << "Cooperative atomic 128-bit builtins unavailable in this toolchain";
#endif
  if (!deviceArchIsGfx1250()) {
    GTEST_SKIP() << "Cooperative atomic load128/store128 path is gfx1250-specific";
  }

  const uint64_t h_src[2] = {0x0F0E0D0C0B0A0908ULL, 0x0706050403020100ULL};
  DeviceBuffer<uint64_t> d_src(2), d_dst(2);
  d_src.copyFrom(h_src, 2);

  kernelLL128Load128Store128<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  auto h_dst = d_dst.copyTo();
  EXPECT_EQ(h_dst[0], h_src[0]);
  EXPECT_EQ(h_dst[1], h_src[1]);
}

__global__ void kernelLL128Store16Load16Global(const uint64_t* in, uint64_t* out) {
  store16global(cvta_to_global(out),
                load16global(cvta_to_global(in)));
}

TEST_F(LL128VectorTest, Load16Store16GlobalRoundtrip) {
  uint64_t h_in[2] = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
  DeviceBuffer<uint64_t> d_in(2), d_out(2);
  d_in.copyFrom(h_in, 2);

  kernelLL128Store16Load16Global<<<1, 1>>>(d_in.ptr, d_out.ptr);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  EXPECT_EQ(h_out[0], h_in[0]);
  EXPECT_EQ(h_out[1], h_in[1]);
}

} // namespace RcclUnitTesting
