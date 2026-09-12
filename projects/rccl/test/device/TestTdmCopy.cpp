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
// NOTE ON ALIGNMENT: detail::issue() peels its "head" against the SOURCE pointer
// only (that is the direct-copy requirement of load_to_lds). The bulk TDM store
// then targets dst+head, so the destination is only guaranteed correct when it
// shares the source's sub-128B alignment. Tests therefore use matched src/dst
// byte offsets, which is the supported contract, while still driving the head /
// bulk / tail split across a wide range of sizes and offsets.

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
// LOCAL_CP is the cache policy for the read leg (src), REMOTE_CP for the write leg
// (dst); each becomes the cpol immediate of its own tensor instruction.
template<bool ASYNC, CachePolicy LOCAL_CP = DEFAULT_CACHE_POLICY, CachePolicy REMOTE_CP = DEFAULT_CACHE_POLICY>
__global__ void kTdmBlockCopy([[maybe_unused]] uint8_t* dst, [[maybe_unused]] const uint8_t* src,
                              [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytes) {
#if TDM_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  if constexpr (ASYNC) {
    tdm::tdmCopyAsync<LOCAL_CP, REMOTE_CP>(dst, src, n, lds, ldsBytes);
    tdm::tdmWait();
  } else {
    tdm::tdmCopy<LOCAL_CP, REMOTE_CP>(dst, src, n, lds, ldsBytes);
  }
  __syncthreads();
#endif
}

// Warp-specialized copy: only warps in [start, stop) participate.
template<bool ASYNC, CachePolicy LOCAL_CP = DEFAULT_CACHE_POLICY, CachePolicy REMOTE_CP = DEFAULT_CACHE_POLICY>
__global__ void kTdmTeamCopy([[maybe_unused]] uint8_t* dst, [[maybe_unused]] const uint8_t* src,
                             [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytes,
                             [[maybe_unused]] uint32_t start, [[maybe_unused]] uint32_t stop) {
#if TDM_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  if constexpr (ASYNC) {
    tdm::tdmCopyAsyncByTeam<LOCAL_CP, REMOTE_CP>(dst, src, n, lds, ldsBytes, start, stop);
    tdm::tdmWait();   // no-op on warps that issued nothing
  } else {
    tdm::tdmCopyByTeam<LOCAL_CP, REMOTE_CP>(dst, src, n, lds, ldsBytes, start, stop);
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
  int         off;        // matched src/dst byte offset (drives the head peel)
  int         block;      // threads per block
  std::string name;
};

class TdmBlockCopyTest : public TdmCopyTest,
                         public ::testing::WithParamInterface<BlockCase> {
protected:
  template<bool ASYNC>
  void run(const BlockCase& c) {
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const size_t srcTotal = static_cast<size_t>(c.off) + c.n;
    const size_t dstTotal = static_cast<size_t>(kGuard) + c.off + c.n + kGuard;
    const size_t copyStart = static_cast<size_t>(kGuard) + c.off;

    auto h_src = makePattern(srcTotal, static_cast<uint32_t>(c.n ^ (c.off * 131) ^ c.block));
    DeviceBuffer<uint8_t> d_src(srcTotal ? srcTotal : 1);
    if (srcTotal) d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    uint8_t* dstPtr = d_dst.ptr + copyStart;
    const uint8_t* srcPtr = d_src.ptr + c.off;
    kTdmBlockCopy<ASYNC><<<1, c.block, c.lds>>>(dstPtr, srcPtr, c.n, c.lds);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(c.off), copyStart, c.n, c.name);
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
    BlockCase{4096,        4096, 0,   64,  "aligned_bulk_2warp"},
    BlockCase{8192,        8192, 0,   256, "aligned_bulk_8warp"},
    BlockCase{256,         1024, 0,   32,  "exactly_one_row"},
    BlockCase{255,         1024, 0,   32,  "tail_only_255"},
    BlockCase{257,         1024, 0,   32,  "one_row_plus_tail"},
    BlockCase{100,         1024, 0,   32,  "tail_only_100"},
    BlockCase{4096,        4096, 5,   128, "head_bulk_tail_off5"},
    BlockCase{6000,        2048, 64,  256, "head_bulk_tail_off64"},
    BlockCase{50,          1024, 100, 32,  "head_only_clamped"},
    BlockCase{4096,        128,  0,   64,  "tiny_lds_vector_fallback"},
    BlockCase{16384,       512,  0,   256, "lds_limits_issuers"},
    BlockCase{4096,        1024, 0,   32,  "single_warp_block"},
    BlockCase{65536 + 37,  8192, 0,   256, "large_odd_remainder"},
    BlockCase{1,           1024, 3,   32,  "one_byte_head"}
  ),
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
//  Non-default cache policies still copy correctly.
// ===========================================================================

// A non-default cache policy proves a policy template argument threads all the way
// through to a tensor instruction's immediate cpol operand.
constexpr CachePolicy kTdmAltCachePolicy = createCachePolicy(TemporalHint::NT, MemScope::DEV);

// The pair the two-policy API exists for: the read leg stays at device scope (the
// source is local HBM) while the write leg goes out to system scope (the
// destination is a peer's memory). Distinct values, so a swapped or shared policy
// shows up as a different cpol immediate in the generated ISA.
constexpr CachePolicy kTdmLocalCachePolicy  = createCachePolicy(TemporalHint::NT, MemScope::DEV);
constexpr CachePolicy kTdmRemoteCachePolicy = createCachePolicy(TemporalHint::HT, MemScope::SYS);

class TdmCachePolicyTest : public TdmCopyTest {
protected:
  template<CachePolicy LOCAL_CP, CachePolicy REMOTE_CP>
  void run(size_t n, size_t lds, int off, int block, const std::string& tag) {
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

    kTdmBlockCopy<false, LOCAL_CP, REMOTE_CP><<<1, block, lds>>>(
        d_dst.ptr + copyStart, d_src.ptr + off, n, lds);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(off), copyStart, n, tag);
  }

  // Same, driven through the warp-specialized entry points.
  template<CachePolicy LOCAL_CP, CachePolicy REMOTE_CP>
  void runByTeam(size_t n, size_t lds, int off, int block, uint32_t start, uint32_t stop,
                 const std::string& tag) {
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const size_t srcTotal  = static_cast<size_t>(off) + n;
    const size_t dstTotal  = static_cast<size_t>(kGuard) + off + n + kGuard;
    const size_t copyStart = static_cast<size_t>(kGuard) + off;

    auto h_src = makePattern(srcTotal, 0xB0A7u);
    DeviceBuffer<uint8_t> d_src(srcTotal);
    d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    kTdmTeamCopy<false, LOCAL_CP, REMOTE_CP><<<1, block, lds>>>(
        d_dst.ptr + copyStart, d_src.ptr + off, n, lds, start, stop);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(off), copyStart, n, tag);
  }
};

// Each policy is varied on its own, so a mix-up that dropped one of the two (or
// applied the other's value to both legs) still has to copy correctly here.
TEST_F(TdmCachePolicyTest, NonDefaultRemotePolicy) {
  run<DEFAULT_CACHE_POLICY, kTdmAltCachePolicy>(6000, 2048, 64, 256, "alt_remote_policy");
}

TEST_F(TdmCachePolicyTest, NonDefaultLocalPolicy) {
  run<kTdmAltCachePolicy, DEFAULT_CACHE_POLICY>(6000, 2048, 64, 256, "alt_local_policy");
}

TEST_F(TdmCachePolicyTest, DistinctLocalAndRemotePolicies) {
  run<kTdmLocalCachePolicy, kTdmRemoteCachePolicy>(6000, 2048, 64, 256, "distinct_policies");
}

// The head/tail edges and the 1-D tail tile are issued by the team's first warp, so
// a small odd size on a sub-team covers those paths with distinct policies too.
TEST_F(TdmCachePolicyTest, DistinctPoliciesByTeam) {
  runByTeam<kTdmLocalCachePolicy, kTdmRemoteCachePolicy>(6000, 2048, 64, 128, 1, kTeamToEnd,
                                                        "distinct_policies_team");
}

}  // namespace RcclUnitTesting
