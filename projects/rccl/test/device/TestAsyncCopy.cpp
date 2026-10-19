/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for tdm/asyncCopy.h -- the async-to/from-LDS HBM->LDS->HBM copy library.
//
// asyncCopy.h exposes the EXACT same public surface as tdm/tdmCopy.h, only in the
// `async` namespace and implemented on top of the global<->LDS async builtins
// instead of the tensor-data-mover instructions. This file mirrors TestTdmCopy.cpp
// so the two implementations are exercised through an identical gamut:
//
//   * async::IsTdmCopySupported()  - host and device capability query
//   * async::tdmCopy()             - blocking, block-collective (all warps)
//   * async::tdmCopyAsync()+Wait   - non-blocking, block-collective
//   * async::tdmCopyByTeam()       - blocking, warp-specialized (a contiguous team)
//   * async::tdmCopyAsyncByTeam()  - non-blocking, warp-specialized
//
// It additionally checks that a non-default compile-time cache policy still
// produces a correct copy (the policy is baked into the async instructions as
// their immediate cpol operand).

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "tdm/asyncCopy.h"

namespace RcclUnitTesting
{

// Bytes of sentinel padding placed on both sides of the destination window so a
// stray write just outside [0, sizeBytes) is detected.
constexpr int kAsyncGuard = 128;
constexpr uint8_t kAsyncSentinel = 0xAB;

// "to the end" marker for a team's stopWarpId (mirrors the ~0u default).
constexpr uint32_t kAsyncTeamToEnd = ~0u;

// A non-default cache policy used to prove the cp template argument threads all
// the way through the async instructions without breaking correctness.
constexpr CachePolicy kAltCachePolicy = createCachePolicy(TemporalHint::NT, MemScope::DEV);

// ---------------------------------------------------------------------------
//  Device kernels: thin wrappers so a whole block (or a team) drives one copy.
// ---------------------------------------------------------------------------

// Whole-block copy of [0, n): dst <- src. Every thread calls with identical args. The copy's LDS region starts
// ldsOff bytes into the dynamic allocation, which must hold ldsOff + ldsBytes.
template<bool ASYNC, CachePolicy CP = DEFAULT_CACHE_POLICY>
__global__ void kAsyncBlockCopy([[maybe_unused]] uint8_t* dst, [[maybe_unused]] const uint8_t* src,
                                [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytes,
                                [[maybe_unused]] size_t ldsOff) {
#if ASYNC_COPY_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  if constexpr (ASYNC) {
    async::tdmCopyAsync<CP>(dst, src, n, lds + ldsOff, ldsBytes);
    async::tdmWait();
  } else {
    async::tdmCopy<CP>(dst, src, n, lds + ldsOff, ldsBytes);
  }
  __syncthreads();
#endif
}

// Warp-specialized copy: only warps in [start, stop) participate.
template<bool ASYNC, CachePolicy CP = DEFAULT_CACHE_POLICY>
__global__ void kAsyncTeamCopy([[maybe_unused]] uint8_t* dst, [[maybe_unused]] const uint8_t* src,
                               [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytes,
                               [[maybe_unused]] uint32_t start, [[maybe_unused]] uint32_t stop) {
#if ASYNC_COPY_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  if constexpr (ASYNC) {
    async::tdmCopyAsyncByTeam<CP>(dst, src, n, lds, ldsBytes, start, stop);
    async::tdmWait();   // no-op on warps that issued nothing
  } else {
    async::tdmCopyByTeam<CP>(dst, src, n, lds, ldsBytes, start, stop);
  }
  __syncthreads();
#endif
}

// Two disjoint teams copy two independent buffers concurrently, each with its
// own LDS region.
template<bool ASYNC>
__global__ void kAsyncTwoTeamCopy([[maybe_unused]] uint8_t* dst0, [[maybe_unused]] const uint8_t* src0,
                                  [[maybe_unused]] uint8_t* dst1, [[maybe_unused]] const uint8_t* src1,
                                  [[maybe_unused]] size_t n, [[maybe_unused]] size_t ldsBytesPerTeam,
                                  [[maybe_unused]] uint32_t split) {
#if ASYNC_COPY_SUPPORTED
  extern __shared__ __align__(128) uint8_t lds[];
  uint8_t* lds1 = lds + ldsBytesPerTeam;
  if constexpr (ASYNC) {
    async::tdmCopyAsyncByTeam(dst0, src0, n, lds,  ldsBytesPerTeam, 0u,    split);
    async::tdmCopyAsyncByTeam(dst1, src1, n, lds1, ldsBytesPerTeam, split, kAsyncTeamToEnd);
    async::tdmWait();
  } else {
    async::tdmCopyByTeam(dst0, src0, n, lds,  ldsBytesPerTeam, 0u,    split);
    async::tdmCopyByTeam(dst1, src1, n, lds1, ldsBytesPerTeam, split, kAsyncTeamToEnd);
  }
  __syncthreads();
#endif
}

// Writes the device-side compile-time capability into result[0].
__global__ void kAsyncSupportProbe(int* result) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    result[0] = async::IsTdmCopySupported() ? 1 : 0;
}

// ---------------------------------------------------------------------------
//  Host fixture + helpers
// ---------------------------------------------------------------------------

class AsyncCopyTest : public DeviceTestBase {
protected:
  int  warpSize_  = 32;
  bool supported_ = false;

  void SetUp() override {
    DeviceTestBase::SetUp();
    hipDeviceProp_t p{};
    ASSERT_EQ(hipGetDeviceProperties(&p, 0), hipSuccess);
    warpSize_  = p.warpSize > 0 ? p.warpSize : 32;
    supported_ = async::IsTdmCopySupported(0);
  }

  static std::vector<uint8_t> makePattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      v[i] = static_cast<uint8_t>(s >> 24);
    }
    return v;
  }

  void checkCopy(const std::vector<uint8_t>& out, const std::vector<uint8_t>& src,
                 size_t srcStart, size_t copyStart, size_t n, const std::string& tag) {
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(static_cast<int>(out[copyStart + i]), static_cast<int>(src[srcStart + i]))
          << tag << ": data mismatch at byte " << i << " of " << n;
    }
    for (size_t i = 0; i < out.size(); ++i) {
      if (i >= copyStart && i < copyStart + n) continue;
      ASSERT_EQ(static_cast<int>(out[i]), static_cast<int>(kAsyncSentinel))
          << tag << ": sentinel clobbered at buffer index " << i;
    }
  }
};

// ===========================================================================
//  Capability query (IsTdmCopySupported)
// ===========================================================================

TEST_F(AsyncCopyTest, HostCapabilityMatchesArch) {
  hipDeviceProp_t p{};
  ASSERT_EQ(hipGetDeviceProperties(&p, 0), hipSuccess);
  const std::string arch(p.gcnArchName);
  const bool archIsAsync = arch.rfind("gfx1250", 0) == 0;
  if (archIsAsync) {
    EXPECT_TRUE(supported_) << "gfx1250 device should report async copy support, arch=" << arch;
  } else {
    EXPECT_FALSE(supported_) << "non-gfx1250 device must report no async copy support, arch=" << arch;
  }
}

TEST_F(AsyncCopyTest, DeviceCapabilityMatchesHost) {
  DeviceBuffer<int> d_res(1);
  int init = -1;
  d_res.copyFrom(&init, 1);
  kAsyncSupportProbe<<<1, 32>>>(d_res.ptr);
  syncAndCheck();
  const int deviceSupported = d_res.download();
  EXPECT_EQ(deviceSupported, supported_ ? 1 : 0)
      << "device-side async::IsTdmCopySupported() disagrees with the host query";
}

// ===========================================================================
//  Block-collective copies (async::tdmCopy / async::tdmCopyAsync)
// ===========================================================================

struct AsyncBlockCase {
  size_t      n;          // bytes to copy
  size_t      lds;        // LDS staging bytes handed to the copy
  int         soff;       // src byte offset from a 512B boundary
  int         doff;       // dst byte offset past the 128B guard
  int         block;      // threads per block
  std::string name;
  size_t      ldsOff = 0; // offset of the copy's LDS region within the allocation
};

class AsyncBlockCopyTest : public AsyncCopyTest,
                           public ::testing::WithParamInterface<AsyncBlockCase> {
protected:
  template<bool ASYNC>
  void run(const AsyncBlockCase& c) {
    if (!supported_) GTEST_SKIP() << "async copy not supported on this device";

    const size_t srcTotal = static_cast<size_t>(c.soff) + c.n;
    const size_t dstTotal = static_cast<size_t>(kAsyncGuard) + c.doff + c.n + kAsyncGuard;
    const size_t copyStart = static_cast<size_t>(kAsyncGuard) + c.doff;

    auto h_src = makePattern(srcTotal, static_cast<uint32_t>(c.n ^ (c.soff * 131) ^ (c.doff * 17) ^ c.block));
    DeviceBuffer<uint8_t> d_src(srcTotal ? srcTotal : 1);
    if (srcTotal) d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kAsyncSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    uint8_t* dstPtr = d_dst.ptr + copyStart;
    const uint8_t* srcPtr = d_src.ptr + c.soff;
    const size_t ldsAlloc = c.lds + c.ldsOff;
    ASSERT_EQ(hipFuncSetAttribute(reinterpret_cast<const void*>(kAsyncBlockCopy<ASYNC>),
                                  hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(ldsAlloc)),
              hipSuccess);
    kAsyncBlockCopy<ASYNC><<<1, c.block, ldsAlloc>>>(dstPtr, srcPtr, c.n, c.lds, c.ldsOff);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(c.soff), copyStart, c.n, c.name);
  }
};

TEST_P(AsyncBlockCopyTest, Blocking)  { run<false>(GetParam()); }
TEST_P(AsyncBlockCopyTest, Async)     { run<true>(GetParam());  }

// A broad gamut hitting: pure aligned bulk; unaligned head+bulk+tail; small
// copies (n<one window); tiny LDS (< one window, direct fallback); LDS that only
// feeds a subset of warps; single-warp blocks; and awkward remainders.
INSTANTIATE_TEST_SUITE_P(
  Gamut, AsyncBlockCopyTest,
  ::testing::Values(
    AsyncBlockCase{4096,        4096, 0,   0,   64,  "aligned_bulk_2warp"},
    AsyncBlockCase{8192,        8192, 0,   0,   256, "aligned_bulk_8warp"},
    AsyncBlockCase{256,         1024, 0,   0,   32,  "exactly_one_row"},
    AsyncBlockCase{255,         1024, 0,   0,   32,  "tail_only_255"},
    AsyncBlockCase{257,         1024, 0,   0,   32,  "one_row_plus_tail"},
    AsyncBlockCase{100,         1024, 0,   0,   32,  "tail_only_100"},
    AsyncBlockCase{4096,        4096, 5,   5,   128, "head_bulk_tail_off5"},
    AsyncBlockCase{6000,        2048, 64,  64,  256, "head_bulk_tail_off64"},
    AsyncBlockCase{50,          1024, 100, 100, 32,  "head_only_clamped"},
    AsyncBlockCase{4096,        64,   0,   0,   64,  "tiny_lds_direct_fallback"},
    AsyncBlockCase{16384,       512,  0,   0,   256, "lds_limits_issuers"},
    AsyncBlockCase{4096,        1024, 0,   0,   32,  "single_warp_block"},
    AsyncBlockCase{65536 + 37,  8192, 0,   0,   256, "large_odd_remainder"},
    AsyncBlockCase{1,           1024, 3,   3,   32,  "one_byte_head"},
    AsyncBlockCase{4096,        4096, 8,   8,   64,  "head_width8"},
    // Load frames under the issuer split: a budget that narrows the team for the ring frames, and one that doesn't.
    AsyncBlockCase{(1 << 20) + 13, 65536, 128, 0, 256, "team_slot"},
    AsyncBlockCase{(1 << 20) + 13, 65536, 0,   0, 256, "team_ring"},
    AsyncBlockCase{(1 << 20) + 13, 65536, 4,   0, 256, "team_ring4"},
    AsyncBlockCase{(1 << 20) + 13, 65536, 5,   0, 256, "team_shift"},
    AsyncBlockCase{(1 << 20) + 13, 32768, 0,   0, 256, "team_ring_narrowed"},
    AsyncBlockCase{(1 << 20) + 13, 32768, 5,   0, 256, "team_shift_32k"},
    // An LDS region that is not 16B aligned.
    AsyncBlockCase{(1 << 18) + 7, 16384, 5,   0, 32,  "lds_off4_shift", 4},
    AsyncBlockCase{(1 << 18) + 7, 16384, 4,   0, 32,  "lds_off5_ring4", 5},
    AsyncBlockCase{(1 << 18) + 7, 16384, 128, 0, 64,  "lds_off12_slot", 12}
  ),
  [](const ::testing::TestParamInfo<AsyncBlockCase>& i){ return i.param.name; });

// Every load frame of the pipeline at every window tier, on one warp so that the window is the whole budget. With
// doff 0 the dst sits 128 bytes past a 512B boundary, so after the head the source is t0 = (soff - 128) % 512
// bytes past one: soff 128 gives the Slot frame, 0 Ring, 4 Ring4 and 5 Shift. Each window is run with a copy that
// fits in it, one whose bulk ends exactly on a slot boundary, and one with an odd tail.
std::vector<AsyncBlockCase> frameTierCases() {
  struct Frame {
    const char* name;
    int soff;
  };
  const Frame frames[] = {{"slot", 128}, {"ring", 0}, {"ring4", 4}, {"shift", 5}};
  const size_t windows[] = {512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536};
  std::vector<AsyncBlockCase> v;
  for (const Frame& f : frames) {
    for (size_t w : windows) {
      const std::string tag = std::string(f.name) + "_w" + std::to_string(w);
      v.push_back({w / 2 + 3, w, f.soff, 0, 32, tag + "_whole"});
      v.push_back({(512 - kAsyncGuard) + 3 * w, w, f.soff, 0, 32, tag + "_slot_end"}); // head + 3 windows
      v.push_back({3 * w + 37, w, f.soff, 0, 32, tag + "_odd"});
    }
  }
  return v;
}

INSTANTIATE_TEST_SUITE_P(
  FrameTiers, AsyncBlockCopyTest, ::testing::ValuesIn(frameTierCases()),
  [](const ::testing::TestParamInfo<AsyncBlockCase>& i){ return i.param.name; });

// ===========================================================================
//  Warp-specialized copies (async::tdmCopyByTeam / async::tdmCopyAsyncByTeam)
// ===========================================================================

struct AsyncTeamCase {
  size_t      n;
  size_t      lds;
  int         off;
  int         block;
  uint32_t    start;
  uint32_t    stop;
  std::string name;
};

class AsyncTeamCopyTest : public AsyncCopyTest,
                          public ::testing::WithParamInterface<AsyncTeamCase> {
protected:
  template<bool ASYNC>
  void run(const AsyncTeamCase& c) {
    if (!supported_) GTEST_SKIP() << "async copy not supported on this device";

    const size_t srcTotal  = static_cast<size_t>(c.off) + c.n;
    const size_t dstTotal  = static_cast<size_t>(kAsyncGuard) + c.off + c.n + kAsyncGuard;
    const size_t copyStart = static_cast<size_t>(kAsyncGuard) + c.off;

    auto h_src = makePattern(srcTotal, static_cast<uint32_t>(c.n ^ (c.start * 7) ^ c.block));
    DeviceBuffer<uint8_t> d_src(srcTotal ? srcTotal : 1);
    if (srcTotal) d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kAsyncSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    kAsyncTeamCopy<ASYNC><<<1, c.block, c.lds>>>(
        d_dst.ptr + copyStart, d_src.ptr + c.off, c.n, c.lds, c.start, c.stop);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(c.off), copyStart, c.n, c.name);
  }
};

TEST_P(AsyncTeamCopyTest, Blocking)  { run<false>(GetParam()); }
TEST_P(AsyncTeamCopyTest, Async)     { run<true>(GetParam());  }

// A single sub-team must still copy the ENTIRE buffer, regardless of where the
// team starts, because work is partitioned by rank within the team.
INSTANTIATE_TEST_SUITE_P(
  Teams, AsyncTeamCopyTest,
  ::testing::Values(
    AsyncTeamCase{8192, 8192, 0,  256, 0, kAsyncTeamToEnd, "full_block_team"},
    AsyncTeamCase{4096, 1024, 0,  256, 0, 1,               "single_warp_team"},
    AsyncTeamCase{8192, 2048, 0,  256, 2, 5,               "interior_team_2_5"},
    AsyncTeamCase{6000, 2048, 64, 128, 1, kAsyncTeamToEnd, "tail_team_head_bulk_tail"},
    AsyncTeamCase{257,  1024, 0,  256, 3, 4,               "single_warp_team_small"}
  ),
  [](const ::testing::TestParamInfo<AsyncTeamCase>& i){ return i.param.name; });

// ===========================================================================
//  Two concurrent teams, each copying its own buffer with its own LDS window.
// ===========================================================================

class AsyncTwoTeamTest : public AsyncCopyTest {
protected:
  template<bool ASYNC>
  void run(size_t n, size_t ldsPerTeam, int block) {
    if (!supported_) GTEST_SKIP() << "async copy not supported on this device";

    const int nWarps = block / warpSize_;
    ASSERT_GE(nWarps, 2) << "two-team test needs at least two warps";
    const uint32_t split = static_cast<uint32_t>(nWarps / 2);

    const size_t total     = static_cast<size_t>(kAsyncGuard) + n + kAsyncGuard;
    const size_t copyStart = kAsyncGuard;

    auto h_src0 = makePattern(n, 0xC0FFEE01u);
    auto h_src1 = makePattern(n, 0xBADF00D2u);
    DeviceBuffer<uint8_t> d_src0(n), d_src1(n);
    d_src0.copyFrom(h_src0);
    d_src1.copyFrom(h_src1);

    std::vector<uint8_t> init(total, kAsyncSentinel);
    DeviceBuffer<uint8_t> d_dst0(total), d_dst1(total);
    d_dst0.copyFrom(init);
    d_dst1.copyFrom(init);

    kAsyncTwoTeamCopy<ASYNC><<<1, block, 2 * ldsPerTeam>>>(
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

TEST_F(AsyncTwoTeamTest, Blocking) { run<false>(8192, 2048, 256); }
TEST_F(AsyncTwoTeamTest, Async)    { run<true>(8192, 2048, 256);  }

// ===========================================================================
//  Non-default cache policy still copies correctly.
// ===========================================================================

class AsyncCachePolicyTest : public AsyncCopyTest {
protected:
  void run(size_t n, size_t lds, int off, int block) {
    if (!supported_) GTEST_SKIP() << "async copy not supported on this device";

    const size_t srcTotal  = static_cast<size_t>(off) + n;
    const size_t dstTotal  = static_cast<size_t>(kAsyncGuard) + off + n + kAsyncGuard;
    const size_t copyStart = static_cast<size_t>(kAsyncGuard) + off;

    auto h_src = makePattern(srcTotal, 0x5EEDu);
    DeviceBuffer<uint8_t> d_src(srcTotal);
    d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kAsyncSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    kAsyncBlockCopy<false, kAltCachePolicy><<<1, block, lds>>>(
        d_dst.ptr + copyStart, d_src.ptr + off, n, lds, 0);
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(off), copyStart, n, "alt_cache_policy");
  }
};

TEST_F(AsyncCachePolicyTest, NonDefaultPolicy) { run(6000, 2048, 64, 256); }

}  // namespace RcclUnitTesting