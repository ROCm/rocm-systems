/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for op128.h: BytePack, toPack/fromPack, load/store primitives.
//
// BytePack<N> is the fundamental data abstraction in RCCL device code.
// It overlays different-width integer views (u8, u16, u32, u64) on the
// same memory, enabling the template machinery in reduce_kernel.h to
// recurse over halves for element-wise reduction.

#include "DeviceTestBase.hpp"
#include <limits>

#include "op128.h"

namespace RcclUnitTesting
{

// ---------------------------------------------------------------------------
// toPack / fromPack roundtrip
// ---------------------------------------------------------------------------

template<typename T>
__global__ void kernelPackRoundtrip(const T* __restrict__ in, T* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  auto pack = toPack(in[i]);
  out[i] = fromPack<T>(pack);
}

class PackRoundtripTest : public DeviceTestBase {
protected:
  template<typename T>
  void TestRoundtrip(const std::vector<T>& h_in) {
    const int N = static_cast<int>(h_in.size());
    DeviceBuffer<T> d_in(N), d_out(N);
    d_in.copyFrom(h_in);

    kernelPackRoundtrip<<<gridFor(N), kDefaultBlockSize>>>(d_in.ptr, d_out.ptr, N);
    syncAndCheck();

    auto h_out = d_out.copyTo();
    for (int i = 0; i < N; i++)
      EXPECT_EQ(h_in[i], h_out[i]) << "at index " << i;
  }
};

TEST_F(PackRoundtripTest, Float) {
  const int N = 1024;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1.0f / (i + 1);
  TestRoundtrip(h_in);
}

TEST_F(PackRoundtripTest, Double) {
  const int N = 512;
  std::vector<double> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<double>(i) * 3.14159;
  TestRoundtrip(h_in);
}

TEST_F(PackRoundtripTest, Uint32) {
  const int N = 1024;
  std::vector<uint32_t> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 0xDEAD0000u + i;
  TestRoundtrip(h_in);
}

TEST_F(PackRoundtripTest, Uint8) {
  const int N = 512;
  std::vector<uint8_t> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<uint8_t>(i & 0xFF);
  TestRoundtrip(h_in);
}

TEST_F(PackRoundtripTest, Int16) {
  const int N = 512;
  std::vector<int16_t> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<int16_t>(i - 256);
  TestRoundtrip(h_in);
}

TEST_F(PackRoundtripTest, SingleElement) {
  std::vector<float> h_in = {42.0f};
  TestRoundtrip(h_in);
}

TEST_F(PackRoundtripTest, FloatExtremeValues) {
  std::vector<float> h_in = {
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::min(),          // smallest positive normal
    std::numeric_limits<float>::denorm_min(),   // smallest positive subnormal
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::epsilon(),
    0.0f,
    1.0f,
    -1.0f,
  };
  TestRoundtrip(h_in);
}

TEST_F(PackRoundtripTest, DoubleExtremeValues) {
  std::vector<double> h_in = {
    std::numeric_limits<double>::max(),
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::denorm_min(),
    std::numeric_limits<double>::lowest(),
    0.0,
    -0.0,
  };
  TestRoundtrip(h_in);
}

// ---------------------------------------------------------------------------
// BytePack half splitting: verify that BytePack<N>.half[0/1] correctly
// partitions data into two halves.
// ---------------------------------------------------------------------------

__global__ void kernelBytePackHalfSplit(const uint64_t* __restrict__ in, uint32_t* __restrict__ out) {
  BytePack<8> pack;
  pack.native = in[0];
  out[0] = pack.half[0].native;
  out[1] = pack.half[1].native;
}

TEST_F(PackRoundtripTest, BytePackHalfSplit) {
  uint64_t h_in = 0x0102030405060708ULL;

  DeviceBuffer<uint64_t> d_in(1);
  DeviceBuffer<uint32_t> d_out(2);
  d_in.upload(h_in);

  kernelBytePackHalfSplit<<<1, 1>>>(d_in.ptr, d_out.ptr);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  EXPECT_EQ(h_out[0], static_cast<uint32_t>(h_in & 0xFFFFFFFFu));
  EXPECT_EQ(h_out[1], static_cast<uint32_t>((h_in >> 32) & 0xFFFFFFFFu));
}

// ---------------------------------------------------------------------------
// BytePack<16> half splitting into two 64-bit halves
// ---------------------------------------------------------------------------

__global__ void kernelBytePackHalfSplit16(const uint64_t* __restrict__ in, uint64_t* __restrict__ out) {
  BytePack<16> pack;
  pack.u64[0] = in[0];
  pack.u64[1] = in[1];
  out[0] = pack.half[0].native;
  out[1] = pack.half[1].native;
}

TEST_F(PackRoundtripTest, BytePack16HalfSplit) {
  uint64_t h_in[2] = {0xAAAABBBBCCCCDDDDULL, 0x1111222233334444ULL};

  DeviceBuffer<uint64_t> d_in(2), d_out(2);
  d_in.copyFrom(h_in, 2);

  kernelBytePackHalfSplit16<<<1, 1>>>(d_in.ptr, d_out.ptr);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  EXPECT_EQ(h_out[0], h_in[0]);
  EXPECT_EQ(h_out[1], h_in[1]);
}

// ---------------------------------------------------------------------------
// BytePack u8 array access: verify per-byte access works correctly
// ---------------------------------------------------------------------------

__global__ void kernelBytePackU8Access(const uint32_t* __restrict__ in, uint8_t* __restrict__ out) {
  BytePack<4> pack;
  pack.native = in[0];
  out[0] = pack.u8[0];
  out[1] = pack.u8[1];
  out[2] = pack.u8[2];
  out[3] = pack.u8[3];
}

TEST_F(PackRoundtripTest, BytePackU8Access) {
  uint32_t h_in = 0x04030201u;  // bytes: 01 02 03 04 (little-endian)

  DeviceBuffer<uint32_t> d_in(1);
  DeviceBuffer<uint8_t> d_out(4);
  d_in.upload(h_in);

  kernelBytePackU8Access<<<1, 1>>>(d_in.ptr, d_out.ptr);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  EXPECT_EQ(h_out[0], 0x01);
  EXPECT_EQ(h_out[1], 0x02);
  EXPECT_EQ(h_out[2], 0x03);
  EXPECT_EQ(h_out[3], 0x04);
}

// ---------------------------------------------------------------------------
// cvta_to_global: verify the address-space cast is a transparent passthrough
// ---------------------------------------------------------------------------

__global__ void kernelCvtaToGlobal(float* ptr, uintptr_t* outAddr) {
  outAddr[0] = cvta_to_global(ptr);
}

TEST_F(PackRoundtripTest, CvtaToGlobal) {
  DeviceBuffer<float> d_ptr(1);
  DeviceBuffer<uintptr_t> d_addr(1);

  kernelCvtaToGlobal<<<1, 1>>>(d_ptr.ptr, d_addr.ptr);
  syncAndCheck();

  uintptr_t h_addr = d_addr.download();
  EXPECT_EQ(h_addr, reinterpret_cast<uintptr_t>(d_ptr.ptr));
}

// ---------------------------------------------------------------------------
// ld_volatile_global / st_global roundtrip through BytePack<4>
// ---------------------------------------------------------------------------

__global__ void kernelLdStGlobal4(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst) {
  uintptr_t srcAddr = cvta_to_global(src);
  uintptr_t dstAddr = cvta_to_global(dst);
  BytePack<4> val = ld_volatile_global<4>(srcAddr);
  st_global<4>(dstAddr, val);
}

TEST_F(PackRoundtripTest, LdStGlobal4) {
  uint32_t h_val = 0xCAFEBABEu;
  DeviceBuffer<uint32_t> d_src(1), d_dst(1);
  d_src.upload(h_val);

  kernelLdStGlobal4<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  EXPECT_EQ(d_dst.download(), h_val);
}

// ---------------------------------------------------------------------------
// ld_volatile_global / st_global roundtrip through BytePack<8>
// ---------------------------------------------------------------------------

__global__ void kernelLdStGlobal8(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst) {
  uintptr_t srcAddr = cvta_to_global(src);
  uintptr_t dstAddr = cvta_to_global(dst);
  BytePack<8> val = ld_volatile_global<8>(srcAddr);
  st_global<8>(dstAddr, val);
}

TEST_F(PackRoundtripTest, LdStGlobal8) {
  uint64_t h_val = 0xDEADBEEFCAFEBABEULL;
  DeviceBuffer<uint64_t> d_src(1), d_dst(1);
  d_src.upload(h_val);

  kernelLdStGlobal8<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  EXPECT_EQ(d_dst.download(), h_val);
}

// ---------------------------------------------------------------------------
// ld_volatile_global / st_global roundtrip through BytePack<16> (128-bit)
// ---------------------------------------------------------------------------

__global__ void kernelLdStGlobal16(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst) {
  uintptr_t srcAddr = cvta_to_global(src);
  uintptr_t dstAddr = cvta_to_global(dst);
  BytePack<16> val = ld_volatile_global<16>(srcAddr);
  st_global<16>(dstAddr, val);
}

__global__ void kernelLdStGlobal16_N(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  const uint64_t* s = src + tid * 2;
  uint64_t* d = dst + tid * 2;
  BytePack<16> val = ld_volatile_global<16>(cvta_to_global(s));
  st_global<16>(cvta_to_global(d), val);
}

TEST_F(PackRoundtripTest, LdStGlobal16) {
  uint64_t h_val[2] = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
  DeviceBuffer<uint64_t> d_src(2), d_dst(2);
  d_src.copyFrom(h_val, 2);

  kernelLdStGlobal16<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  auto h_out = d_dst.copyTo();
  EXPECT_EQ(h_out[0], h_val[0]);
  EXPECT_EQ(h_out[1], h_val[1]);
}

// ---------------------------------------------------------------------------
// load128 / store128 (legacy 128-bit helpers)
// ---------------------------------------------------------------------------

__global__ void kernelLoad128Store128(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst) {
  uint64_t v0, v1;
  load128(src, v0, v1);
  store128(dst, v0, v1);
}

TEST_F(PackRoundtripTest, Load128Store128) {
  uint64_t h_src[2] = {0xAAAABBBBCCCCDDDDULL, 0x1111222233334444ULL};
  DeviceBuffer<uint64_t> d_src(2), d_dst(2);
  d_src.copyFrom(h_src, 2);

  kernelLoad128Store128<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  auto h_dst = d_dst.copyTo();
  EXPECT_EQ(h_dst[0], h_src[0]);
  EXPECT_EQ(h_dst[1], h_src[1]);
}

// ---------------------------------------------------------------------------
// ld_volatile_global / st_global roundtrip through BytePack<1> and <2>
// ---------------------------------------------------------------------------

__global__ void kernelLdStGlobal1(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst) {
  uintptr_t srcAddr = cvta_to_global(src);
  uintptr_t dstAddr = cvta_to_global(dst);
  BytePack<1> val = ld_volatile_global<1>(srcAddr);
  st_global<1>(dstAddr, val);
}

TEST_F(PackRoundtripTest, LdStGlobal1) {
  uint8_t h_val = 0xAB;
  DeviceBuffer<uint8_t> d_src(1), d_dst(1);
  d_src.upload(h_val);

  kernelLdStGlobal1<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  EXPECT_EQ(d_dst.download(), h_val);
}

__global__ void kernelLdStGlobal2(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst) {
  uintptr_t srcAddr = cvta_to_global(src);
  uintptr_t dstAddr = cvta_to_global(dst);
  BytePack<2> val = ld_volatile_global<2>(srcAddr);
  st_global<2>(dstAddr, val);
}

TEST_F(PackRoundtripTest, LdStGlobal2) {
  uint16_t h_val = 0xBEEF;
  DeviceBuffer<uint16_t> d_src(1), d_dst(1);
  d_src.upload(h_val);

  kernelLdStGlobal2<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  EXPECT_EQ(d_dst.download(), h_val);
}

// NaN bit-pattern preservation: NaN != NaN under float ==, so we must
// compare the underlying bits to verify the roundtrip is lossless.
__global__ void kernelPackRoundtripBits(const uint32_t* __restrict__ in,
                                        uint32_t* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float val;
  std::memcpy(&val, &in[i], sizeof(float));
  auto pack = toPack(val);
  float result = fromPack<float>(pack);
  std::memcpy(&out[i], &result, sizeof(float));
}

TEST_F(PackRoundtripTest, NaN_BitPatternPreserved) {
  uint32_t nan_bits[] = {
    0x7FC00000u,   // quiet NaN
    0x7F800001u,   // signaling NaN
    0xFFC00000u,   // negative quiet NaN
    0x7FFFFFFFu,   // NaN with all mantissa bits set
  };
  const int N = sizeof(nan_bits) / sizeof(nan_bits[0]);

  DeviceBuffer<uint32_t> d_in(N), d_out(N);
  d_in.copyFrom(nan_bits, N);

  kernelPackRoundtripBits<<<1, N>>>(d_in.ptr, d_out.ptr, N);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  for (int i = 0; i < N; i++)
    EXPECT_EQ(h_out[i], nan_bits[i]) << "NaN bit pattern 0x" << std::hex << nan_bits[i];
}

// Negative zero: -0.0 == +0.0 under float ==, but their bit patterns differ.
// Verify the roundtrip preserves the sign bit.
TEST_F(PackRoundtripTest, NegativeZero_BitPatternPreserved) {
  float neg_zero = -0.0f;
  float pos_zero = +0.0f;
  uint32_t neg_bits, pos_bits;
  std::memcpy(&neg_bits, &neg_zero, sizeof(float));
  std::memcpy(&pos_bits, &pos_zero, sizeof(float));

  DeviceBuffer<uint32_t> d_in(2), d_out(2);
  uint32_t input[2] = {neg_bits, pos_bits};
  d_in.copyFrom(input, 2);

  kernelPackRoundtripBits<<<1, 2>>>(d_in.ptr, d_out.ptr, 2);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  EXPECT_EQ(h_out[0], neg_bits) << "-0.0 sign bit lost";
  EXPECT_EQ(h_out[1], pos_bits) << "+0.0 corrupted";
  EXPECT_NE(h_out[0], h_out[1]) << "-0.0 and +0.0 should have different bits";
}

// Infinity roundtrip: +Inf and -Inf must survive toPack/fromPack.
TEST_F(PackRoundtripTest, Infinity_Preserved) {
  std::vector<float> h_in = {
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
  };
  TestRoundtrip(h_in);
}

// BytePack all-zeros: every view (u8, u16, u32) must be zero.
__global__ void kernelBytePackAllZero(uint32_t* __restrict__ out) {
  BytePack<4> pack;
  pack.native = 0;
  out[0] = pack.native;
  out[1] = pack.u8[0] | pack.u8[1] | pack.u8[2] | pack.u8[3];
  out[2] = pack.u16[0] | pack.u16[1];
}

TEST_F(PackRoundtripTest, BytePackAllZero) {
  DeviceBuffer<uint32_t> d_out(3);

  kernelBytePackAllZero<<<1, 1>>>(d_out.ptr);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  EXPECT_EQ(h_out[0], 0u) << "native should be 0";
  EXPECT_EQ(h_out[1], 0u) << "OR of all u8 bytes should be 0";
  EXPECT_EQ(h_out[2], 0u) << "OR of all u16 halves should be 0";
}

// BytePack all-ones: every byte must be 0xFF.
__global__ void kernelBytePackAllOnes(uint8_t* __restrict__ out) {
  BytePack<4> pack;
  pack.native = 0xFFFFFFFFu;
  out[0] = pack.u8[0];
  out[1] = pack.u8[1];
  out[2] = pack.u8[2];
  out[3] = pack.u8[3];
}

TEST_F(PackRoundtripTest, BytePackAllOnes) {
  DeviceBuffer<uint8_t> d_out(4);

  kernelBytePackAllOnes<<<1, 1>>>(d_out.ptr);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(h_out[i], 0xFF) << "byte " << i << " should be 0xFF";
}

// ld_volatile_global / st_global of zero-initialized memory returns zeros.
TEST_F(PackRoundtripTest, LdStGlobal4_ZeroMemory) {
  DeviceBuffer<uint32_t> d_src(1), d_dst(1);
  d_src.zero();
  d_dst.zero();

  kernelLdStGlobal4<<<1, 1>>>(d_src.ptr, d_dst.ptr);
  syncAndCheck();

  EXPECT_EQ(d_dst.download(), 0u);
}

// Excess threads beyond data count must not corrupt output.
// Launch 1024 threads for only 4 elements — extra threads should exit early
// via the bounds check, leaving unrelated memory untouched.
TEST_F(PackRoundtripTest, ExcessThreads_NoCorruption) {
  const int N = 4;
  std::vector<float> h_in = {1.0f, 2.0f, 3.0f, 4.0f};

  DeviceBuffer<float> d_in(N), d_out(N);
  d_in.copyFrom(h_in);
  d_out.zero();

  kernelPackRoundtrip<<<4, kDefaultBlockSize>>>(d_in.ptr, d_out.ptr, N);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  for (int i = 0; i < N; i++)
    EXPECT_EQ(h_out[i], h_in[i]) << "at index " << i;
}

// cvta_to_global with nullptr equivalent: verify the cast of a known-zero
// address produces zero (not a crash — the kernel never dereferences it).
__global__ void kernelCvtaNullptr(uintptr_t* outAddr) {
  outAddr[0] = cvta_to_global((float*)nullptr);
}

TEST_F(PackRoundtripTest, CvtaToGlobal_Nullptr) {
  DeviceBuffer<uintptr_t> d_addr(1);

  kernelCvtaNullptr<<<1, 1>>>(d_addr.ptr);
  syncAndCheck();

  EXPECT_EQ(d_addr.download(), uintptr_t(0));
}

// Integration tests that combine multiple op128 components or exercise
// non-trivial internal paths.

// Verify that writing via one BytePack view and reading via another produces
// consistent results (union aliasing correctness on GPU).
__global__ void kernelCrossViewWrite(uint32_t* __restrict__ out) {
  BytePack<4> pack;
  pack.u16[0] = 0xBEEF;
  pack.u16[1] = 0xCAFE;
  out[0] = pack.native;
  out[1] = pack.u8[0];
  out[2] = pack.u8[1];
  out[3] = pack.u8[2];
  out[4] = pack.u8[3];
}

TEST_F(PackRoundtripTest, L2_CrossViewAliasing) {
  DeviceBuffer<uint32_t> d_out(5);

  kernelCrossViewWrite<<<1, 1>>>(d_out.ptr);
  syncAndCheck();

  auto h = d_out.copyTo();
  EXPECT_EQ(h[0], 0xCAFEBEEFu);
  EXPECT_EQ(h[1], 0xEF);
  EXPECT_EQ(h[2], 0xBE);
  EXPECT_EQ(h[3], 0xFE);
  EXPECT_EQ(h[4], 0xCA);
}

// Verify BytePack<8> cross-view: write u32 halves, read u64 and u8 views.
__global__ void kernelCrossView8(uint64_t* __restrict__ out64, uint8_t* __restrict__ out8) {
  BytePack<8> pack;
  pack.u32[0] = 0x04030201u;
  pack.u32[1] = 0x08070605u;
  out64[0] = pack.native;
  for (int i = 0; i < 8; i++) out8[i] = pack.u8[i];
}

TEST_F(PackRoundtripTest, L2_CrossView8) {
  DeviceBuffer<uint64_t> d_out64(1);
  DeviceBuffer<uint8_t> d_out8(8);

  kernelCrossView8<<<1, 1>>>(d_out64.ptr, d_out8.ptr);
  syncAndCheck();

  EXPECT_EQ(d_out64.download(), 0x0807060504030201ULL);
  auto h8 = d_out8.copyTo();
  for (int i = 0; i < 8; i++)
    EXPECT_EQ(h8[i], i + 1) << "byte " << i;
}

// Verify BytePack<4> custom copy constructor preserves bit patterns including
// special values (alternating bits, sign bit patterns).
__global__ void kernelBytePack4Copy(const uint32_t* __restrict__ in,
                                    uint32_t* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  BytePack<4> src;
  src.native = in[i];
  BytePack<4> dst(src);  // copy ctor
  BytePack<4> dst2;
  dst2 = dst;            // assignment
  out[i] = dst2.native;
}

TEST_F(PackRoundtripTest, L2_BytePack4CopySemantics) {
  std::vector<uint32_t> h_in = {
    0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u,
    0x80000000u, 0x80000001u, 0x7FFFFFFFu, 0xDEADBEEFu,
  };
  const int N = static_cast<int>(h_in.size());

  DeviceBuffer<uint32_t> d_in(N), d_out(N);
  d_in.copyFrom(h_in);

  kernelBytePack4Copy<<<1, N>>>(d_in.ptr, d_out.ptr, N);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  for (int i = 0; i < N; i++)
    EXPECT_EQ(h_out[i], h_in[i]) << "pattern 0x" << std::hex << h_in[i];
}

// Chain: store as BytePack<4>, load back as BytePack<4>, then store as
// BytePack<8> from two halves. Tests interplay of different-width operations.
__global__ void kernelChainedSizes(const uint32_t* __restrict__ in,
                                   uint64_t* __restrict__ out) {
  uintptr_t inAddr = cvta_to_global(in);
  BytePack<4> lo = ld_volatile_global<4>(inAddr);
  BytePack<4> hi = ld_volatile_global<4>(inAddr + 4);
  BytePack<8> combined;
  combined.half[0] = lo;
  combined.half[1] = hi;
  uintptr_t outAddr = cvta_to_global(out);
  st_global<8>(outAddr, combined);
}

TEST_F(PackRoundtripTest, L2_ChainedLdStDifferentSizes) {
  uint32_t h_in[2] = {0xAABBCCDDu, 0x11223344u};
  DeviceBuffer<uint32_t> d_in(2);
  DeviceBuffer<uint64_t> d_out(1);
  d_in.copyFrom(h_in, 2);

  kernelChainedSizes<<<1, 1>>>(d_in.ptr, d_out.ptr);
  syncAndCheck();

  EXPECT_EQ(d_out.download(), 0x11223344AABBCCDDuLL);
}

// Large-scale roundtrip mixing denormals, NaN, Inf, and normal values.
// Verifies no corruption across 64K elements spanning multiple blocks.
TEST_F(PackRoundtripTest, L2_LargeScaleMixedSpecialValues) {
  const int N = 65536;
  std::vector<uint32_t> h_in(N);
  for (int i = 0; i < N; i++) {
    switch (i % 5) {
      case 0: h_in[i] = 0x7FC00000u; break;  // quiet NaN
      case 1: h_in[i] = 0x7F800000u; break;  // +Inf
      case 2: h_in[i] = 0x00000001u; break;  // smallest denormal
      case 3: h_in[i] = 0x80000000u; break;  // -0.0
      case 4: h_in[i] = static_cast<uint32_t>(i); break;
    }
  }

  DeviceBuffer<uint32_t> d_in(N), d_out(N);
  d_in.copyFrom(h_in);

  kernelPackRoundtripBits<<<gridFor(N), kDefaultBlockSize>>>(d_in.ptr, d_out.ptr, N);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  for (int i = 0; i < N; i++)
    EXPECT_EQ(h_out[i], h_in[i]) << "at i=" << i;
}

// Stress and extreme-boundary tests.

// 128-bit ld/st with many concurrent threads — verifies no cross-thread
// corruption when each thread operates on a separate 16-byte slot.
TEST_F(PackRoundtripTest, L4_LdStGlobal16_ManyThreads) {
  const int N = 4096;
  std::vector<uint64_t> h_in(N * 2);
  for (int i = 0; i < N; i++) {
    h_in[2*i]   = 0xA0B0C0D0E0F00000ULL + i;
    h_in[2*i+1] = 0x1020304050600000ULL + i;
  }

  DeviceBuffer<uint64_t> d_src(N * 2), d_dst(N * 2);
  d_src.copyFrom(h_in);
  d_dst.zero();

  kernelLdStGlobal16_N<<<gridFor(N), kDefaultBlockSize>>>(d_src.ptr, d_dst.ptr, N);
  syncAndCheck();

  auto h_out = d_dst.copyTo();
  for (int i = 0; i < N * 2; i++)
    EXPECT_EQ(h_out[i], h_in[i]) << "at index " << i;
}

// All 256 possible byte values in a large roundtrip — exercises every bit
// pattern through the toPack/fromPack path for uint8_t.
TEST_F(PackRoundtripTest, L4_AllByteValues_Uint8) {
  const int N = 256 * 64;
  std::vector<uint8_t> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<uint8_t>(i & 0xFF);

  DeviceBuffer<uint8_t> d_in(N), d_out(N);
  d_in.copyFrom(h_in);

  kernelPackRoundtrip<<<gridFor(N), kDefaultBlockSize>>>(d_in.ptr, d_out.ptr, N);
  syncAndCheck();

  auto h_out = d_out.copyTo();
  for (int i = 0; i < N; i++)
    EXPECT_EQ(h_out[i], h_in[i]) << "at index " << i;
}

// Alternating 0xAA / 0x55 stress at 32-bit level — a classic bit-pattern
// that catches byte-swap or bit-shift bugs.
TEST_F(PackRoundtripTest, L4_AlternatingBitPatterns_Uint32) {
  const int N = 8192;
  std::vector<uint32_t> h_in(N);
  for (int i = 0; i < N; i++)
    h_in[i] = (i & 1) ? 0xAAAAAAAAu : 0x55555555u;

  TestRoundtrip(h_in);
}

// Verify ld_volatile_global/st_global with every possible single-bit-set
// 32-bit value (power-of-2 walk). Catches bit-lane corruption.
TEST_F(PackRoundtripTest, L4_SingleBitWalk_LdSt4) {
  const int N = 32;
  std::vector<uint32_t> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1u << i;

  DeviceBuffer<uint32_t> d_src(N), d_dst(N);
  d_src.copyFrom(h_in);

  for (int i = 0; i < N; i++) {
    kernelLdStGlobal4<<<1, 1>>>(d_src.ptr + i, d_dst.ptr + i);
  }
  syncAndCheck();

  auto h_out = d_dst.copyTo();
  for (int i = 0; i < N; i++)
    EXPECT_EQ(h_out[i], 1u << i) << "bit " << i;
}

} // namespace RcclUnitTesting
