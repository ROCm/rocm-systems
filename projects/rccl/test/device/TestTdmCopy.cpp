/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for tdm/tdmCopy.h -- the HBM->LDS->HBM Tensor Data Mover copy library.
//
// tdmCopy.h exposes four device entry points plus a wait and a host/device
// capability query. This file exercises every one of them and the internal
// [head][aligned 256B rows][tail] partitioning of detail::issue():
//
//   * IsTdmCopySupported()  - host and device capability query
//   * tdmCopy()             - blocking, block-collective (all warps)
//   * tdmCopyAsync()+tdmWait- non-blocking, block-collective
//   * tdmCopyByTeam()       - blocking, warp-specialized (a contiguous team)
//   * tdmCopyAsyncByTeam()  - non-blocking, warp-specialized
//
// The copy is byte-oriented (void* / size_t). Every test rounds a source buffer
// through the copy into a distinct destination buffer and compares byte-for-byte,
// and also checks sentinel padding around the destination window so an out-of-
// bounds write is caught.
//
// Source and destination offsets are independent: detail::issue() aligns each
// chunk to the destination and absorbs the difference delta = (src - dst) % 256
// in LDS, shifting the chunk in place when delta % 4 != 0. The destination sits
// past a kGuard-byte pad, so matched offsets give delta 128.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "tdm/tdmCopy.h"

namespace RcclUnitTesting
{

// Bytes of sentinel padding placed on both sides of the destination window so a
// stray write just outside [0, sizeBytes) is detected.
constexpr int kGuard = 128;
constexpr uint8_t kSentinel = 0xAB;

// "to the end" marker for a team's stopWarpId (mirrors the ~0u default).
constexpr uint32_t kTeamToEnd = ~0u;

// ---------------------------------------------------------------------------
//  Device kernels: thin wrappers so a whole block (or a team) drives one copy.
// ---------------------------------------------------------------------------

// The tdm:: entry points are declared `= delete` unless TDM_SUPPORTED (which
// tdmCopy.h leaves defined after the include). In HIP the HOST compilation pass
// never defines the device arch macro, so TDM_SUPPORTED is 0 there and any call
// would be a "use of deleted function" error. Guarding the call sites with
// `#if TDM_SUPPORTED` keeps the kernels body-less in the host pass (and on any
// non-TDM device pass) while emitting the real copy for the gfx1250 device pass.

// Whole-block copy of [0, n): dst <- src. Every thread calls with identical args.
// CP is the compile-time cache policy threaded into the tensor load/store.
template<bool ASYNC, CachePolicy CP = DEFAULT_CACHE_POLICY>
__global__ void kTdmBlockCopy([[maybe_unused]] uint8_t* dst, [[maybe_unused]] const uint8_t* src,
                              [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytes) {
#if TDM_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  if constexpr (ASYNC) {
    tdm::tdmCopyAsync<CP>(dst, src, n, lds, ldsBytes);
    tdm::tdmWait();
  } else {
    tdm::tdmCopy<CP>(dst, src, n, lds, ldsBytes);
  }
  __syncthreads();
#endif
}

// Warp-specialized copy: only warps in [start, stop) participate.
template<bool ASYNC>
__global__ void kTdmTeamCopy([[maybe_unused]] uint8_t* dst, [[maybe_unused]] const uint8_t* src,
                             [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytes,
                             [[maybe_unused]] uint32_t start, [[maybe_unused]] uint32_t stop) {
#if TDM_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  if constexpr (ASYNC) {
    tdm::tdmCopyAsyncByTeam(dst, src, n, lds, ldsBytes, start, stop);
    tdm::tdmWait();   // no-op on warps that issued nothing
  } else {
    tdm::tdmCopyByTeam(dst, src, n, lds, ldsBytes, start, stop);
  }
  __syncthreads();
#endif
}

// Two disjoint teams copy two independent buffers concurrently, each with its
// own LDS region. Every warp calls both entry points; the off-team calls return
// immediately, so this exercises two live teams in one block.
template<bool ASYNC>
__global__ void kTdmTwoTeamCopy([[maybe_unused]] uint8_t* dst0, [[maybe_unused]] const uint8_t* src0,
                                [[maybe_unused]] uint8_t* dst1, [[maybe_unused]] const uint8_t* src1,
                                [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytesPerTeam,
                                [[maybe_unused]] uint32_t split) {
#if TDM_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  uint8_t* lds1 = lds + ldsBytesPerTeam;
  if constexpr (ASYNC) {
    tdm::tdmCopyAsyncByTeam(dst0, src0, n, lds,  ldsBytesPerTeam, 0u,    split);
    tdm::tdmCopyAsyncByTeam(dst1, src1, n, lds1, ldsBytesPerTeam, split, kTeamToEnd);
    tdm::tdmWait();
  } else {
    tdm::tdmCopyByTeam(dst0, src0, n, lds,  ldsBytesPerTeam, 0u,    split);
    tdm::tdmCopyByTeam(dst1, src1, n, lds1, ldsBytesPerTeam, split, kTeamToEnd);
  }
  __syncthreads();
#endif
}

// Writes the device-side compile-time capability into result[0].
__global__ void kTdmSupportProbe(int* result) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    result[0] = tdm::IsTdmCopySupported() ? 1 : 0;
}

// ---------------------------------------------------------------------------
//  Host fixture + helpers
// ---------------------------------------------------------------------------

class TdmCopyTest : public DeviceTestBase {
protected:
  int  warpSize_  = 32;
  bool supported_ = false;

  void SetUp() override {
    DeviceTestBase::SetUp();
    hipDeviceProp_t p{};
    ASSERT_EQ(hipGetDeviceProperties(&p, 0), hipSuccess);
    warpSize_  = p.warpSize > 0 ? p.warpSize : 32;
    supported_ = tdm::IsTdmCopySupported(0);
  }

  // Deterministic, non-trivial byte pattern so every position has a distinct,
  // reproducible value (a simple LCG).
  static std::vector<uint8_t> makePattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      v[i] = static_cast<uint8_t>(s >> 24);
    }
    return v;
  }

  // Allocate a destination buffer of [guard][off][n][guard] laid out so that
  // element (off) is the start of the copy window, and fill it with sentinels.
  // Returns the device buffer plus the host-side initial contents.
  struct DstBuf {
    DeviceBuffer<uint8_t> dev;
    std::vector<uint8_t>  initHost;
    size_t                copyStart;   // index in the buffer where the copy lands
  };

  // Verify a byte range matches the expected source, and that all padding
  // outside [copyStart, copyStart+n) is still the sentinel.
  void checkCopy(const std::vector<uint8_t>& out, const std::vector<uint8_t>& src,
                 size_t srcStart, size_t copyStart, size_t n, const std::string& tag) {
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(static_cast<int>(out[copyStart + i]), static_cast<int>(src[srcStart + i]))
          << tag << ": data mismatch at byte " << i << " of " << n;
    }
    for (size_t i = 0; i < out.size(); ++i) {
      if (i >= copyStart && i < copyStart + n) continue;
      ASSERT_EQ(static_cast<int>(out[i]), static_cast<int>(kSentinel))
          << tag << ": sentinel clobbered at buffer index " << i;
    }
  }
};

// ===========================================================================
//  Capability query (IsTdmCopySupported)
// ===========================================================================

TEST_F(TdmCopyTest, HostCapabilityMatchesArch) {
  hipDeviceProp_t p{};
  ASSERT_EQ(hipGetDeviceProperties(&p, 0), hipSuccess);
  const std::string arch(p.gcnArchName);
  const bool archIsTdm = arch.rfind("gfx1250", 0) == 0;
  // The host query must agree with the arch for TDM-capable devices. On a
  // non-TDM device it must report false.
  if (archIsTdm) {
    EXPECT_TRUE(supported_) << "gfx1250 device should report TDM support, arch=" << arch;
  } else {
    EXPECT_FALSE(supported_) << "non-gfx1250 device must report no TDM support, arch=" << arch;
  }
}

TEST_F(TdmCopyTest, DeviceCapabilityMatchesHost) {
  DeviceBuffer<int> d_res(1);
  int init = -1;
  d_res.copyFrom(&init, 1);
  kTdmSupportProbe<<<1, 32>>>(d_res.ptr);
  syncAndCheck();
  const int deviceSupported = d_res.download();
  EXPECT_EQ(deviceSupported, supported_ ? 1 : 0)
      << "device-side IsTdmCopySupported() disagrees with the host query";
}

// ===========================================================================
//  Block-collective copies (tdmCopy / tdmCopyAsync)
// ===========================================================================

struct BlockCase {
  size_t      n;          // bytes to copy
  size_t      lds;        // LDS staging bytes handed to the copy
  int         soff;       // src byte offset from an allocation boundary
  int         doff;       // dst byte offset past the kGuard pad
  int         block;      // threads per block
  std::string name;
};

class TdmBlockCopyTest : public TdmCopyTest,
                         public ::testing::WithParamInterface<BlockCase> {
protected:
  template<bool ASYNC>
  void run(const BlockCase& c) {
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const size_t srcTotal = static_cast<size_t>(c.soff) + c.n;
    const size_t dstTotal = static_cast<size_t>(kGuard) + c.doff + c.n + kGuard;
    const size_t copyStart = static_cast<size_t>(kGuard) + c.doff;

    auto h_src = makePattern(srcTotal, static_cast<uint32_t>(c.n ^ (c.soff * 131) ^ (c.doff * 17) ^ c.block));
    DeviceBuffer<uint8_t> d_src(srcTotal ? srcTotal : 1);
    if (srcTotal) d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    uint8_t* dstPtr = d_dst.ptr + copyStart;
    const uint8_t* srcPtr = d_src.ptr + c.soff;
    kTdmBlockCopy<ASYNC><<<1, c.block, c.lds>>>(dstPtr, srcPtr, c.n, c.lds);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(c.soff), copyStart, c.n, c.name);
  }
};

TEST_P(TdmBlockCopyTest, Blocking)  { run<false>(GetParam()); }
TEST_P(TdmBlockCopyTest, Async)     { run<true>(GetParam());  }

// A broad gamut hitting: pure aligned bulk; head+bulk+tail; tail-only (n<256B);
// head-only (head clamped by n); tiny LDS (< 256B, vector fallback); LDS that
// only feeds a subset of warps; single-warp blocks; and awkward remainders.
INSTANTIATE_TEST_SUITE_P(
  Gamut, TdmBlockCopyTest,
  ::testing::Values(
    BlockCase{4096,        4096, 0,   0,   64,  "aligned_bulk_2warp"},
    BlockCase{8192,        8192, 0,   0,   256, "aligned_bulk_8warp"},
    BlockCase{256,         1024, 0,   0,   32,  "exactly_one_row"},
    BlockCase{255,         1024, 0,   0,   32,  "tail_only_255"},
    BlockCase{257,         1024, 0,   0,   32,  "one_row_plus_tail"},
    BlockCase{100,         1024, 0,   0,   32,  "tail_only_100"},
    BlockCase{4096,        4096, 5,   5,   128, "head_bulk_tail_off5"},
    BlockCase{6000,        2048, 64,  64,  256, "head_bulk_tail_off64"},
    BlockCase{50,          1024, 100, 100, 32,  "head_only_clamped"},
    BlockCase{4096,        128,  0,   0,   64,  "tiny_lds_vector_fallback"},
    BlockCase{16384,       2176, 0,   0,   256, "lds_limits_issuers"},
    BlockCase{4096,        1024, 0,   0,   32,  "single_warp_block"},
    BlockCase{65536 + 37,  8192, 0,   0,   256, "large_odd_remainder"},
    BlockCase{1,           1024, 3,   3,   32,  "one_byte_head"},
    // Two slots: shifting (delta 129) on a 4KB window, and not shifting on a window wide enough for two anyway.
    BlockCase{65536 + 37,  4096,  5, 4, 32,  "two_slot_shift"},
    BlockCase{(1 << 18) + 5, 32768, 0, 0, 32, "two_slot_wide"},
    // Every delta % 4 across a whole block.
    BlockCase{(1 << 20) + 13, 65536, 0, 0, 256, "team_delta128"},
    BlockCase{(1 << 20) + 13, 65536, 1, 0, 256, "team_delta129"},
    BlockCase{(1 << 20) + 13, 65536, 2, 0, 256, "team_delta130"},
    BlockCase{(1 << 20) + 13, 65536, 3, 0, 256, "team_delta131"}
  ),
  [](const ::testing::TestParamInfo<BlockCase>& i){ return i.param.name; });

// Independent offsets across the slot layouts, on one warp so that the window is the whole budget: one slot (just
// under and at MIN_SLOT), two slots when shifting (at 2 * MIN_SLOT and above), and two slots regardless (16KB+).
// With doff 0 the destination sits kGuard bytes past an allocation boundary, so delta = (soff - kGuard) % 256.
std::vector<BlockCase> offsetSlotCases() {
  const int soffs[] = {0, 1, 2, 3, 4, 5, 7, 128, 131, 255};
  const size_t ldsSizes[] = {544, 1087, 1088, 4096, 16384, 65536};
  std::vector<BlockCase> v;
  for (size_t lds : ldsSizes) {
    for (int soff : soffs) {
      const std::string tag = "lds" + std::to_string(lds) + "_soff" + std::to_string(soff);
      v.push_back({300, lds, soff, 0, 32, tag + "_short"});
      v.push_back({3 * lds + 37, lds, soff, 0, 32, tag + "_long"});
    }
  }
  return v;
}

INSTANTIATE_TEST_SUITE_P(
  OffsetSlots, TdmBlockCopyTest, ::testing::ValuesIn(offsetSlotCases()),
  [](const ::testing::TestParamInfo<BlockCase>& i){ return i.param.name; });

// ===========================================================================
//  Warp-specialized copies (tdmCopyByTeam / tdmCopyAsyncByTeam)
// ===========================================================================

struct TeamCase {
  size_t      n;
  size_t      lds;        // LDS for the participating team
  int         off;
  int         block;
  uint32_t    start;      // first warp of the team
  uint32_t    stop;       // one past the last (kTeamToEnd = to the end)
  std::string name;
};

class TdmTeamCopyTest : public TdmCopyTest,
                        public ::testing::WithParamInterface<TeamCase> {
protected:
  template<bool ASYNC>
  void run(const TeamCase& c) {
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const size_t srcTotal  = static_cast<size_t>(c.off) + c.n;
    const size_t dstTotal  = static_cast<size_t>(kGuard) + c.off + c.n + kGuard;
    const size_t copyStart = static_cast<size_t>(kGuard) + c.off;

    auto h_src = makePattern(srcTotal, static_cast<uint32_t>(c.n ^ (c.start * 7) ^ c.block));
    DeviceBuffer<uint8_t> d_src(srcTotal ? srcTotal : 1);
    if (srcTotal) d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    kTdmTeamCopy<ASYNC><<<1, c.block, c.lds>>>(
        d_dst.ptr + copyStart, d_src.ptr + c.off, c.n, c.lds, c.start, c.stop);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(c.off), copyStart, c.n, c.name);
  }
};

TEST_P(TdmTeamCopyTest, Blocking)  { run<false>(GetParam()); }
TEST_P(TdmTeamCopyTest, Async)     { run<true>(GetParam());  }

// A single sub-team must still copy the ENTIRE buffer, regardless of where the
// team starts, because work is partitioned by rank within the team.
INSTANTIATE_TEST_SUITE_P(
  Teams, TdmTeamCopyTest,
  ::testing::Values(
    TeamCase{8192, 8192, 0,  256, 0, kTeamToEnd, "full_block_team"},
    TeamCase{4096, 1024, 0,  256, 0, 1,          "single_warp_team"},
    TeamCase{8192, 2048, 0,  256, 2, 5,          "interior_team_2_5"},
    TeamCase{6000, 2048, 64, 128, 1, kTeamToEnd, "tail_team_head_bulk_tail"},
    TeamCase{257,  1024, 0,  256, 3, 4,          "single_warp_team_small"}
  ),
  [](const ::testing::TestParamInfo<TeamCase>& i){ return i.param.name; });

// ===========================================================================
//  Two concurrent teams, each copying its own buffer with its own LDS window.
// ===========================================================================

class TdmTwoTeamTest : public TdmCopyTest {
protected:
  template<bool ASYNC>
  void run(size_t n, size_t ldsPerTeam, int block) {
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const int nWarps = block / warpSize_;
    ASSERT_GE(nWarps, 2) << "two-team test needs at least two warps";
    const uint32_t split = static_cast<uint32_t>(nWarps / 2);

    const size_t total     = static_cast<size_t>(kGuard) + n + kGuard;
    const size_t copyStart = kGuard;

    auto h_src0 = makePattern(n, 0xC0FFEE01u);
    auto h_src1 = makePattern(n, 0xBADF00D2u);
    DeviceBuffer<uint8_t> d_src0(n), d_src1(n);
    d_src0.copyFrom(h_src0);
    d_src1.copyFrom(h_src1);

    std::vector<uint8_t> init(total, kSentinel);
    DeviceBuffer<uint8_t> d_dst0(total), d_dst1(total);
    d_dst0.copyFrom(init);
    d_dst1.copyFrom(init);

    kTdmTwoTeamCopy<ASYNC><<<1, block, 2 * ldsPerTeam>>>(
        d_dst0.ptr + copyStart, d_src0.ptr,
        d_dst1.ptr + copyStart, d_src1.ptr,
        n, ldsPerTeam, split);
    syncAndCheck();

    auto out0 = d_dst0.copyTo();
    auto out1 = d_dst1.copyTo();
    checkCopy(out0, h_src0, 0, copyStart, n, "two_team_buf0");
    checkCopy(out1, h_src1, 0, copyStart, n, "two_team_buf1");
  }
};

TEST_F(TdmTwoTeamTest, Blocking) { run<false>(8192, 2048, 256); }
TEST_F(TdmTwoTeamTest, Async)    { run<true>(8192, 2048, 256);  }

// ===========================================================================
//  Non-default cache policy still copies correctly.
// ===========================================================================

// A non-default cache policy proves the cp template argument threads all the way
// through the tensor load/store as their immediate cpol operand.
constexpr CachePolicy kTdmAltCachePolicy = createCachePolicy(TemporalHint::NT, MemScope::DEV);

class TdmCachePolicyTest : public TdmCopyTest {
protected:
  void run(size_t n, size_t lds, int off, int block) {
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const size_t srcTotal  = static_cast<size_t>(off) + n;
    const size_t dstTotal  = static_cast<size_t>(kGuard) + off + n + kGuard;
    const size_t copyStart = static_cast<size_t>(kGuard) + off;

    auto h_src = makePattern(srcTotal, 0x5EEDu);
    DeviceBuffer<uint8_t> d_src(srcTotal);
    d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    kTdmBlockCopy<false, kTdmAltCachePolicy><<<1, block, lds>>>(
        d_dst.ptr + copyStart, d_src.ptr + off, n, lds);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(off), copyStart, n, "alt_cache_policy");
  }
};

TEST_F(TdmCachePolicyTest, NonDefaultPolicy) { run(6000, 2048, 64, 256); }

}  // namespace RcclUnitTesting
