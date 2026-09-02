/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for the one-way warp-level global<->LDS transfer primitives:
//
//   * asyncLoadToLDS() / asyncStoreFromLDS()      (tdm/asyncCopy.h, global scope)
//       implemented with the global_load_async_to_lds / global_store_async_from_lds
//       builtins.
//   * tdm::asyncLoadToLDS() / tdm::asyncStoreFromLDS()   (tdm/tdmCopy.h)
//       implemented with the tensor data mover's tensor_load_to_lds /
//       tensor_store_from_lds instructions.
//
// The two are meant to be drop-in replacements for one another, so this file
// defines ONE gamut and ONE staging kernel and runs both implementations through
// them via a dispatch shim (TdmLdsOps / AsyncLdsOps). Anything that passes for
// one implementation and fails for the other is a parity bug.
//
// Each warp stages its own slice of the buffer global -> LDS -> global through
// its own LDS window and the result is compared byte-for-byte against the source,
// with sentinel padding around the destination window catching stray writes.
//
// NOTE ON ALIGNMENT: both implementations peel their head against the GLOBAL
// pointer and apply the same offset to the LDS side, so the contract is that the
// two pointers share a sub-128B offset. The kernel therefore skews each warp's
// LDS window by the same byte offset the test applies to src/dst.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "tdm/asyncCopy.h"
#include "tdm/tdmCopy.h"

namespace RcclUnitTesting
{

// Bytes of sentinel padding on both sides of the destination window.
constexpr int     kLdsGuard    = 128;
constexpr uint8_t kLdsSentinel = 0xAB;

// A non-default cache policy, to prove the cp template argument threads through
// to both implementations' memory instructions without breaking correctness.
constexpr CachePolicy kLdsAltCachePolicy = createCachePolicy(TemporalHint::NT, MemScope::DEV);

// ---------------------------------------------------------------------------
//  Dispatch shims: one per implementation of the primitive.
// ---------------------------------------------------------------------------
// Each entry point is `= delete`d (TDM) or built on arch-specific builtins
// (async) unless the device pass targets a capable arch, so the device bodies
// are guarded and compile away to nothing elsewhere -- including in the HOST
// pass, which never defines the arch macro. The capability query is always
// callable and is what the fixture skips on.

struct TdmLdsOps {
  static constexpr const char* kName = "TDM";
  static bool hostSupported() { return tdm::IsTdmCopySupported(0); }

  template<SyncPolicy SP, CachePolicy CP, bool ALIGNED>
  __device__ static void load([[maybe_unused]] const uint8_t* globalSrc,
                              [[maybe_unused]] uint8_t* ldsDst,
                              [[maybe_unused]] size_t n) {
#if TDM_SUPPORTED
    tdm::asyncLoadToLDS<SP, CP, ALIGNED>(globalSrc, ldsDst, n);
#endif
  }

  template<SyncPolicy SP, CachePolicy CP, bool ALIGNED>
  __device__ static void store([[maybe_unused]] const uint8_t* ldsSrc,
                               [[maybe_unused]] uint8_t* globalDst,
                               [[maybe_unused]] size_t n) {
#if TDM_SUPPORTED
    tdm::asyncStoreFromLDS<SP, CP, ALIGNED>(ldsSrc, globalDst, n);
#endif
  }

  __device__ static void wait() {
#if TDM_SUPPORTED
    tdm::tdmWait();
#endif
  }
};

struct AsyncLdsOps {
  static constexpr const char* kName = "async-to-LDS";
  static bool hostSupported() { return async::IsTdmCopySupported(0); }

  template<SyncPolicy SP, CachePolicy CP, bool ALIGNED>
  __device__ static void load([[maybe_unused]] const uint8_t* globalSrc,
                              [[maybe_unused]] uint8_t* ldsDst,
                              [[maybe_unused]] size_t n) {
#if ASYNC_COPY_SUPPORTED
    ::asyncLoadToLDS<SP, CP, ALIGNED>(globalSrc, ldsDst, n);
#endif
  }

  template<SyncPolicy SP, CachePolicy CP, bool ALIGNED>
  __device__ static void store([[maybe_unused]] const uint8_t* ldsSrc,
                               [[maybe_unused]] uint8_t* globalDst,
                               [[maybe_unused]] size_t n) {
#if ASYNC_COPY_SUPPORTED
    ::asyncStoreFromLDS<SP, CP, ALIGNED>(ldsSrc, globalDst, n);
#endif
  }

  __device__ static void wait() {
#if ASYNC_COPY_SUPPORTED
    asyncWait<0>();
#endif
  }
};

// ---------------------------------------------------------------------------
//  Staging kernel: warp w round-trips [w*bytesPerWarp, ...) through its own LDS
//  window. Warps past the end of the buffer transfer zero bytes, which exercises
//  the empty-transfer guard rather than branching around the call.
//
//  The two legs are dispatched independently so a TDM load can be paired with an
//  async store and vice versa; a mixed round trip only reproduces the source if
//  both implementations lay the staged bytes out in LDS the same way.
// ---------------------------------------------------------------------------
template<typename LoadOps, typename StoreOps, SyncPolicy SP, CachePolicy CP, bool ALIGNED>
__global__ void kLdsStage(uint8_t* dst, const uint8_t* src, size_t bytesPerWarp,
                          size_t totalBytes, uint32_t window, uint32_t ldsSkew) {
  extern __shared__ __align__(128) uint8_t lds[];

  const uint32_t warpId = threadIdx.x / warpSize;
  const size_t   start  = static_cast<size_t>(warpId) * bytesPerWarp;
  const size_t   begin  = start < totalBytes ? start : totalBytes;
  const size_t   avail  = totalBytes - begin;
  const size_t   myBytes = avail < bytesPerWarp ? avail : bytesPerWarp;

  uint8_t* myLds = lds + static_cast<size_t>(warpId) * window + ldsSkew;

  LoadOps::template load<SP, CP, ALIGNED>(src + begin, myLds, myBytes);
  // Under SyncPolicy::Sync each call drains itself; otherwise the RAW hazard on
  // the LDS window (and the final drain before the kernel exits) is ours to cover.
  if constexpr (SP == SyncPolicy::Async) LoadOps::wait();
  StoreOps::template store<SP, CP, ALIGNED>(myLds, dst + begin, myBytes);
  if constexpr (SP == SyncPolicy::Async) StoreOps::wait();

  __syncthreads();
}

// Writes the device-side compile-time capability of each implementation.
__global__ void kLdsSupportProbe(int* result) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    result[0] = tdm::IsTdmCopySupported()   ? 1 : 0;
    result[1] = async::IsTdmCopySupported() ? 1 : 0;
  }
}

// ---------------------------------------------------------------------------
//  Test cases
// ---------------------------------------------------------------------------

struct LdsCase {
  size_t      bytesPerWarp;   // slice each warp round-trips through LDS
  int         off;            // matched src/dst/LDS byte skew (drives the head peel)
  int         block;          // threads per block
  size_t      shortBy;        // bytes trimmed off the end, to ragged the last warps
  std::string name;
};

// Sizes stay small so the whole gamut is practical under the ffmlite emulator;
// the shapes, not the volume, are what distinguishes the code paths.
inline const std::vector<LdsCase>& ldsGamut() {
  static const std::vector<LdsCase> cases = {
    LdsCase{1,    0,   32,  0,    "one_byte"},
    LdsCase{4,    0,   32,  0,    "four_bytes"},
    LdsCase{127,  0,   32,  0,    "sub_line_127"},
    LdsCase{128,  0,   32,  0,    "exactly_one_line"},
    LdsCase{255,  0,   32,  0,    "tail_only_255"},
    LdsCase{256,  0,   32,  0,    "exactly_one_row"},
    LdsCase{257,  0,   32,  0,    "one_row_plus_tail"},
    LdsCase{1024, 0,   32,  0,    "aligned_bulk"},
    LdsCase{1024, 5,   32,  0,    "head_bulk_tail_off5"},
    LdsCase{1024, 64,  32,  0,    "head_bulk_tail_off64"},
    LdsCase{300,  1,   32,  0,    "head_bulk_tail_off1"},
    LdsCase{100,  100, 32,  0,    "head_only_clamped"},
    LdsCase{2048, 0,   64,  0,    "two_warps"},
    LdsCase{512,  5,   128, 0,    "four_warps_off5"},
    LdsCase{1024, 0,   256, 0,    "eight_warps"},
    LdsCase{512,  0,   128, 612,  "ragged_and_idle_warps"},
    LdsCase{4096, 0,   64,  0,    "larger_bulk"},
  };
  return cases;
}

inline std::string ldsCaseName(const ::testing::TestParamInfo<LdsCase>& info) {
  return info.param.name;
}

// ---------------------------------------------------------------------------
//  Fixture, templated on the implementation under test
// ---------------------------------------------------------------------------

template<typename Ops>
class LdsTransferBase : public DeviceTestBase {
protected:
  int warpSize_ = 32;

  void SetUp() override {
    DeviceTestBase::SetUp();
    hipDeviceProp_t p{};
    ASSERT_EQ(hipGetDeviceProperties(&p, 0), hipSuccess);
    warpSize_ = p.warpSize > 0 ? p.warpSize : 32;
  }

  // Deterministic, non-trivial byte pattern (a simple LCG) so every position has
  // a distinct, reproducible value.
  static std::vector<uint8_t> makePattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      v[i] = static_cast<uint8_t>(s >> 24);
    }
    return v;
  }

  static size_t roundUpLine(size_t n) { return (n + 127) & ~static_cast<size_t>(127); }

  void checkCopy(const std::vector<uint8_t>& out, const std::vector<uint8_t>& src,
                 size_t srcStart, size_t copyStart, size_t n, const std::string& tag) {
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(static_cast<int>(out[copyStart + i]), static_cast<int>(src[srcStart + i]))
          << tag << ": data mismatch at byte " << i << " of " << n;
    }
    for (size_t i = 0; i < out.size(); ++i) {
      if (i >= copyStart && i < copyStart + n) continue;
      ASSERT_EQ(static_cast<int>(out[i]), static_cast<int>(kLdsSentinel))
          << tag << ": sentinel clobbered at buffer index " << i;
    }
  }

  template<SyncPolicy SP, CachePolicy CP = DEFAULT_CACHE_POLICY, bool ALIGNED = false>
  void run(const LdsCase& c) {
    runWith<Ops, Ops, SP, CP, ALIGNED>(c);
  }

  template<typename LoadOps, typename StoreOps, SyncPolicy SP,
           CachePolicy CP = DEFAULT_CACHE_POLICY, bool ALIGNED = false>
  void runWith(const LdsCase& c) {
    if (!LoadOps::hostSupported() || !StoreOps::hostSupported())
      GTEST_SKIP() << LoadOps::kName << "/" << StoreOps::kName
                   << " global<->LDS transfers not supported on this device";

    const int nWarps = c.block / warpSize_;
    ASSERT_GE(nWarps, 1) << "block must hold at least one warp";

    const size_t span  = c.bytesPerWarp * static_cast<size_t>(nWarps);
    const size_t total = c.shortBy < span ? span - c.shortBy : 0;
    ASSERT_GT(total, 0u) << "case trims away the whole buffer";

    const size_t srcTotal  = static_cast<size_t>(c.off) + total;
    const size_t dstTotal  = static_cast<size_t>(kLdsGuard) + c.off + total + kLdsGuard;
    const size_t copyStart = static_cast<size_t>(kLdsGuard) + c.off;

    auto h_src = makePattern(srcTotal, static_cast<uint32_t>(total ^ (c.off * 131) ^ c.block));
    DeviceBuffer<uint8_t> d_src(srcTotal);
    d_src.copyFrom(h_src);

    std::vector<uint8_t> h_dstInit(dstTotal, kLdsSentinel);
    DeviceBuffer<uint8_t> d_dst(dstTotal);
    d_dst.copyFrom(h_dstInit);

    // Each warp gets a 128B-multiple window; its slice sits `off` bytes into that
    // window so the LDS and global pointers share a sub-128B offset.
    const size_t window = roundUpLine(static_cast<size_t>(c.off) + c.bytesPerWarp);
    const size_t shared = window * static_cast<size_t>(nWarps);

    kLdsStage<LoadOps, StoreOps, SP, CP, ALIGNED><<<1, c.block, shared>>>(
        d_dst.ptr + copyStart, d_src.ptr + c.off, c.bytesPerWarp, total,
        static_cast<uint32_t>(window), static_cast<uint32_t>(c.off));
    syncAndCheck();

    auto h_out = d_dst.copyTo();
    checkCopy(h_out, h_src, static_cast<size_t>(c.off), copyStart, total, c.name);
  }
};

template<typename Ops>
class LdsGamutBase : public LdsTransferBase<Ops>,
                     public ::testing::WithParamInterface<LdsCase> {};

class TdmLdsGamutTest   : public LdsGamutBase<TdmLdsOps>   {};
class AsyncLdsGamutTest : public LdsGamutBase<AsyncLdsOps> {};

class TdmLdsFeatureTest   : public LdsTransferBase<TdmLdsOps>   {};
class AsyncLdsFeatureTest : public LdsTransferBase<AsyncLdsOps> {};

// The same suites are declared for both implementations; only the fixture (and
// therefore the dispatch shim) differs.
#define LDS_TRANSFER_GAMUT_SUITE(FIXTURE)                                        \
  TEST_P(FIXTURE, Sync)  { run<SyncPolicy::Sync>(GetParam());  }                 \
  TEST_P(FIXTURE, Async) { run<SyncPolicy::Async>(GetParam()); }                 \
  INSTANTIATE_TEST_SUITE_P(Gamut, FIXTURE, ::testing::ValuesIn(ldsGamut()), ldsCaseName)

#define LDS_TRANSFER_FEATURE_SUITE(FIXTURE)                                                   \
  /* Aligned=true compiles out the head peel; only legal when both sides start on a line. */  \
  TEST_F(FIXTURE, AlignedFastPath) {                                                          \
    run<SyncPolicy::Sync, DEFAULT_CACHE_POLICY, true>(                                        \
        LdsCase{1024, 0, 64, 0, "aligned_fast_path"});                                        \
  }                                                                                           \
  TEST_F(FIXTURE, AlignedFastPathRagged) {                                                    \
    run<SyncPolicy::Async, DEFAULT_CACHE_POLICY, true>(                                       \
        LdsCase{384, 0, 128, 500, "aligned_fast_path_ragged"});                               \
  }                                                                                           \
  TEST_F(FIXTURE, NonDefaultCachePolicy) {                                                    \
    run<SyncPolicy::Sync, kLdsAltCachePolicy>(LdsCase{1024, 64, 32, 0, "alt_cache_policy"});  \
  }

LDS_TRANSFER_GAMUT_SUITE(TdmLdsGamutTest);
LDS_TRANSFER_GAMUT_SUITE(AsyncLdsGamutTest);

LDS_TRANSFER_FEATURE_SUITE(TdmLdsFeatureTest)
LDS_TRANSFER_FEATURE_SUITE(AsyncLdsFeatureTest)

// ===========================================================================
//  Cross-implementation round trips.
// ===========================================================================
// Staging one leg with each implementation only reproduces the source if the two
// agree byte-for-byte on how a transfer is laid out in LDS. That is the one thing
// a same-implementation round trip cannot catch: the TDM bulk stages through a
// 2-D tile of 256B rows, and a load that padded or reordered those rows in LDS
// would still store back correctly through the matching TDM descriptor.

class MixedLdsTransferTest : public LdsTransferBase<TdmLdsOps> {
protected:
  template<typename LoadOps, typename StoreOps>
  void runGamut() {
    for (const LdsCase& c : ldsGamut()) {
      SCOPED_TRACE(c.name);
      runWith<LoadOps, StoreOps, SyncPolicy::Sync>(c);
      if (HasFatalFailure() || IsSkipped()) return;
    }
  }
};

TEST_F(MixedLdsTransferTest, TdmLoadAsyncStore) { runGamut<TdmLdsOps, AsyncLdsOps>(); }
TEST_F(MixedLdsTransferTest, AsyncLoadTdmStore) { runGamut<AsyncLdsOps, TdmLdsOps>(); }

// ===========================================================================
//  Capability queries agree between host and device for both implementations.
// ===========================================================================

class LdsSupportTest : public DeviceTestBase {};

TEST_F(LdsSupportTest, DeviceCapabilityMatchesHost) {
  DeviceBuffer<int> d_res(2);
  const int init[2] = {-1, -1};
  d_res.copyFrom(init, 2);
  kLdsSupportProbe<<<1, 32>>>(d_res.ptr);
  syncAndCheck();

  const auto res = d_res.copyTo();
  EXPECT_EQ(res[0], tdm::IsTdmCopySupported(0) ? 1 : 0)
      << "device-side tdm::IsTdmCopySupported() disagrees with the host query";
  EXPECT_EQ(res[1], async::IsTdmCopySupported(0) ? 1 : 0)
      << "device-side async::IsTdmCopySupported() disagrees with the host query";
}

}  // namespace RcclUnitTesting