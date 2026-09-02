/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for the rccl_ep wave primitives and symmetric-window layout.
//
// These cover the parts of the module that need neither peer memory nor a
// communicator, so they run on a single GPU: the wave64 primitives every kernel
// is built on, the two window helpers, and the layout arithmetic that decides
// where each region lands. Dispatch and combine need more than one rank and
// symmetric memory, so they are out of scope for this single-GPU binary.

#include "DeviceTestBase.hpp"
#include <hip/hip_bf16.h>
#include <numeric>
#include <vector>

#include "device/ep_common.h"
#include "device/hip_prims.h"
#include "include/ep_layout.h"

namespace RcclUnitTesting
{

// The module is wave64 throughout. A literal 32 in the index arithmetic fails
// silently rather than at compile time, so pin the geometry here.
static_assert(rccl_ep::kWarpSize == 64, "rccl_ep assumes wave64");
static_assert(sizeof(rccl_ep::wave_mask_t) == 8, "wave masks must be 64-bit");
static_assert(rccl_ep::kFullWaveMask == ~static_cast<rccl_ep::wave_mask_t>(0), "mask must be all ones");

// ---------------------------------------------------------------------------
// Wave geometry
// ---------------------------------------------------------------------------

__global__ void kernelWaveGeometry(int* __restrict__ lane, int* __restrict__ warp, int* __restrict__ nwarps) {
  const int t = static_cast<int>(threadIdx.x);
  lane[t]   = rccl_ep::get_lane_idx();
  warp[t]   = rccl_ep::get_warp_idx();
  nwarps[t] = rccl_ep::get_num_warps();
}

class RcclEpWaveTest : public DeviceTestBase {
protected:
  static constexpr int kThreads = 256;  // four wave64s
  static constexpr int kWaves   = kThreads / rccl_ep::kWarpSize;

  // rccl_ep is wave64-only, and the default target list includes wave32 parts.
  void SetUp() override {
    DeviceTestBase::SetUp();
    hipDeviceProp_t p{};
    ASSERT_EQ(hipGetDeviceProperties(&p, 0), hipSuccess);
    if (p.warpSize != rccl_ep::kWarpSize) {
      GTEST_SKIP() << "rccl_ep requires wave64; this device is wave" << p.warpSize;
    }
  }

  // Each wave gets a different pattern; every primitive here is wave-local, so
  // a leak across a wave boundary shows up as a wrong total.
  static std::vector<int> MakeInput() {
    std::vector<int> v(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      v[t] = (t % rccl_ep::kWarpSize) + 1 + (t / rccl_ep::kWarpSize);
    }
    return v;
  }
};

TEST_F(RcclEpWaveTest, GeometryMatchesWave64) {
  DeviceBuffer<int> d_lane(kThreads), d_warp(kThreads), d_nwarps(kThreads);

  kernelWaveGeometry<<<1, kThreads>>>(d_lane.ptr, d_warp.ptr, d_nwarps.ptr);
  syncAndCheck();

  const std::vector<int> h_lane   = d_lane.copyTo();
  const std::vector<int> h_warp   = d_warp.copyTo();
  const std::vector<int> h_nwarps = d_nwarps.copyTo();

  for (int t = 0; t < kThreads; ++t) {
    EXPECT_EQ(h_lane[t], t % rccl_ep::kWarpSize) << "thread " << t;
    EXPECT_EQ(h_warp[t], t / rccl_ep::kWarpSize) << "thread " << t;
    EXPECT_EQ(h_nwarps[t], kWaves) << "thread " << t;
  }
}

// ---------------------------------------------------------------------------
// Wave-wide reduction and scans
// ---------------------------------------------------------------------------

__global__ void kernelReduceAdd(const int* __restrict__ in, int* __restrict__ out) {
  const int t = static_cast<int>(threadIdx.x);
  out[t] = rccl_ep::reduce_add(in[t]);
}

__global__ void kernelInclusiveSum(const int* __restrict__ in, int* __restrict__ out) {
  const int t = static_cast<int>(threadIdx.x);
  out[t] = rccl_ep::warp_inclusive_sum(in[t]);
}

__global__ void kernelExclusiveSum(const int* __restrict__ in, int* __restrict__ out) {
  const int t = static_cast<int>(threadIdx.x);
  out[t] = rccl_ep::warp_exclusive_sum(in[t]);
}

TEST_F(RcclEpWaveTest, ReduceAddIsWaveLocal) {
  const std::vector<int> h_in = MakeInput();

  DeviceBuffer<int> d_in(kThreads), d_out(kThreads);
  d_in.copyFrom(h_in);
  kernelReduceAdd<<<1, kThreads>>>(d_in.ptr, d_out.ptr);
  syncAndCheck();

  const std::vector<int> h_out = d_out.copyTo();
  for (int w = 0; w < kWaves; ++w) {
    const int base     = w * rccl_ep::kWarpSize;
    const int expected = std::accumulate(h_in.begin() + base, h_in.begin() + base + rccl_ep::kWarpSize, 0);
    // reduce_add broadcasts: every lane in the wave holds the wave total.
    for (int l = 0; l < rccl_ep::kWarpSize; ++l) {
      EXPECT_EQ(h_out[base + l], expected) << "wave " << w << " lane " << l;
    }
  }
}

TEST_F(RcclEpWaveTest, InclusiveAndExclusiveSums) {
  const std::vector<int> h_in = MakeInput();

  DeviceBuffer<int> d_in(kThreads), d_inc(kThreads), d_exc(kThreads);
  d_in.copyFrom(h_in);
  kernelInclusiveSum<<<1, kThreads>>>(d_in.ptr, d_inc.ptr);
  syncAndCheck();
  kernelExclusiveSum<<<1, kThreads>>>(d_in.ptr, d_exc.ptr);
  syncAndCheck();

  const std::vector<int> h_inc = d_inc.copyTo();
  const std::vector<int> h_exc = d_exc.copyTo();

  for (int w = 0; w < kWaves; ++w) {
    const int base = w * rccl_ep::kWarpSize;
    int running    = 0;
    for (int l = 0; l < rccl_ep::kWarpSize; ++l) {
      const int before = running;
      running += h_in[base + l];
      EXPECT_EQ(h_inc[base + l], running) << "wave " << w << " lane " << l;
      EXPECT_EQ(h_exc[base + l], before) << "wave " << w << " lane " << l;
    }
  }
}

// ---------------------------------------------------------------------------
// Window helpers
// ---------------------------------------------------------------------------

__global__ void kernelLastLocalSlot(const int32_t* __restrict__ topk, int rows, int numTopk,
                                    int* __restrict__ out) {
  for (int r = 0; r < rows; ++r) out[r] = rccl_ep::last_local_slot(topk, r, numTopk);
}

TEST_F(RcclEpWaveTest, LastLocalSlotFindsHighestOwnedPosition) {
  constexpr int kTopk = 4;
  // -1 means "not owned by this rank". The helper returns the largest owned
  // position, which is where the origin rank -- the rank a token was routed
  // from -- folds this rank's partial sum.
  const std::vector<int32_t> h_topk = {
      3, -1, -1, -1,   // only position 0
      -1, -1, -1, 7,   // only position 3
      1, -1, 5, -1,    // highest owned is 2
      -1, -1, -1, -1,  // none owned
      0, 1, 2, 3,      // all owned
  };
  const std::vector<int> expected = {0, 3, 2, -1, 3};
  const int rows                  = static_cast<int>(expected.size());

  DeviceBuffer<int32_t> d_topk(h_topk.size());
  DeviceBuffer<int> d_out(rows);
  d_topk.copyFrom(h_topk);

  kernelLastLocalSlot<<<1, 1>>>(d_topk.ptr, rows, kTopk, d_out.ptr);
  syncAndCheck();

  const std::vector<int> h_out = d_out.copyTo();
  for (int r = 0; r < rows; ++r) EXPECT_EQ(h_out[r], expected[r]) << "row " << r;
}

__global__ void kernelWaveCopy(const rccl_ep::bf16_t* __restrict__ src, rccl_ep::bf16_t* __restrict__ dst,
                               int n) {
  rccl_ep::wave_copy_bf16(dst, src, n);
}

TEST_F(RcclEpWaveTest, WaveCopyMovesEveryElement) {
  // 512 exercises the vectorised dwordx4 path alone; 517 forces the scalar
  // tail, which handles a leftover count and is not an alignment fallback.
  const std::vector<int> sizes = {1, 8, 512, 517};

  for (int n : sizes) {
    SCOPED_TRACE(testing::Message() << "n = " << n);

    std::vector<rccl_ep::bf16_t> h_src(n);
    for (int i = 0; i < n; ++i) {
      h_src[i] = __float2bfloat16(static_cast<float>((i % 251) - 125));
    }

    DeviceBuffer<rccl_ep::bf16_t> d_src(n), d_dst(n);
    d_src.copyFrom(h_src);
    d_dst.zero();

    kernelWaveCopy<<<1, rccl_ep::kWarpSize>>>(d_src.ptr, d_dst.ptr, n);
    syncAndCheck();

    const std::vector<rccl_ep::bf16_t> h_dst = d_dst.copyTo();
    for (int i = 0; i < n; ++i) {
      // Values are small integers, so bf16 holds them exactly.
      EXPECT_EQ(__bfloat162float(h_dst[i]), __bfloat162float(h_src[i])) << "element " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// Window layout arithmetic (host only)
// ---------------------------------------------------------------------------

static rccl_ep::EpConfig makeConfig(int ranks, int tokens, int hidden, int topk) {
  rccl_ep::EpConfig c{};
  c.num_ranks               = ranks;
  c.rank                    = 0;
  c.num_experts             = ranks * 32;
  c.num_topk                = topk;
  c.hidden                  = hidden;
  c.num_max_tokens_per_rank = tokens;
  return c;
}

TEST(RcclEpLayoutTest, RegionsAreAlignedOrderedAndDisjoint) {
  const rccl_ep::EpConfig c        = makeConfig(8, 4096, 7168, 6);
  const size_t elemBytes           = sizeof(uint16_t);
  const rccl_ep::EpWindowLayout l  = rccl_ep::EpWindowLayout::make(c, elemBytes);

  const size_t R     = c.num_ranks;
  const size_t slots = R * c.num_max_tokens_per_rank;

  struct Region {
    const char* name;
    size_t off;
    size_t used;
  };
  const Region regions[] = {
      {"counts",   l.off_counts,   R * sizeof(int32_t)},
      {"flags",    l.off_flags,    R * sizeof(uint32_t)},
      {"x",        l.off_x,        slots * c.hidden * elemBytes},
      {"sf",       l.off_sf,       slots * c.hidden_sf() * sizeof(float)},
      {"topk_idx", l.off_topk_idx, slots * c.num_topk * sizeof(int32_t)},
      {"topk_w",   l.off_topk_w,   slots * c.num_topk * sizeof(float)},
      {"src_idx",  l.off_src_idx,  slots * sizeof(int32_t)},
      {"y",        l.off_y,        slots * c.num_topk * c.hidden * sizeof(uint16_t)},
      {"cw",       l.off_cw,       slots * c.num_topk * sizeof(float)},
      {"arch",     l.off_arch,     R * rccl_ep::kArchLen},
  };
  const int n = static_cast<int>(sizeof(regions) / sizeof(regions[0]));

  for (int i = 0; i < n; ++i) {
    // Every region starts on a 256B boundary so a peer store never straddles.
    EXPECT_EQ(regions[i].off % rccl_ep::kAlign, 0u) << regions[i].name;
    const size_t end  = regions[i].off + regions[i].used;
    const size_t next = (i + 1 < n) ? regions[i + 1].off : l.total_bytes;
    EXPECT_LE(end, next) << regions[i].name << " runs into "
                         << ((i + 1 < n) ? regions[i + 1].name : "the end of the window");
  }
  EXPECT_EQ(l.total_bytes % rccl_ep::kAlign, 0u);
}

TEST(RcclEpLayoutTest, SlotIsRankMajorAndInjective) {
  const rccl_ep::EpConfig c = makeConfig(4, 16, 128, 2);

  // Concatenating per-source regions in rank order is what lets the receiver
  // reconstruct the required ordering with no sort, so slot() has to be
  // rank-major, contiguous per source, and collision-free.
  std::vector<bool> seen(static_cast<size_t>(c.num_ranks) * c.num_max_tokens_per_rank, false);
  for (int s = 0; s < c.num_ranks; ++s) {
    for (int i = 0; i < c.num_max_tokens_per_rank; ++i) {
      const size_t slot = rccl_ep::EpWindowLayout::slot(c, s, i);
      ASSERT_LT(slot, seen.size());
      EXPECT_FALSE(seen[slot]) << "slot " << slot << " reused at source " << s << " token " << i;
      seen[slot] = true;
      if (i > 0) EXPECT_EQ(slot, rccl_ep::EpWindowLayout::slot(c, s, i - 1) + 1);
    }
  }
  for (size_t i = 0; i < seen.size(); ++i) EXPECT_TRUE(seen[i]) << "slot " << i << " unreachable";
}

TEST(RcclEpLayoutTest, GrowsWithRanksTokensHiddenAndTopk) {
  const size_t elemBytes = sizeof(uint16_t);
  const size_t base      = rccl_ep::EpWindowLayout::make(makeConfig(8, 1024, 1024, 4), elemBytes).total_bytes;

  // Payload dominates the window. Assert monotonic growth rather than an exact
  // size, which the per-region alignment perturbs.
  EXPECT_GT(rccl_ep::EpWindowLayout::make(makeConfig(16, 1024, 1024, 4), elemBytes).total_bytes, base);
  EXPECT_GT(rccl_ep::EpWindowLayout::make(makeConfig(8, 2048, 1024, 4), elemBytes).total_bytes, base);
  EXPECT_GT(rccl_ep::EpWindowLayout::make(makeConfig(8, 1024, 2048, 4), elemBytes).total_bytes, base);
  EXPECT_GT(rccl_ep::EpWindowLayout::make(makeConfig(8, 1024, 1024, 8), elemBytes).total_bytes, base);
}

TEST(RcclEpLayoutTest, ExpertAndScaleFactorArithmetic) {
  const rccl_ep::EpConfig c = makeConfig(8, 128, 7168, 6);

  EXPECT_EQ(c.experts_per_rank(), c.num_experts / c.num_ranks);
  EXPECT_EQ(c.expert_begin(0), 0);
  EXPECT_EQ(c.expert_end(c.num_ranks - 1), c.num_experts);
  for (int r = 1; r < c.num_ranks; ++r) EXPECT_EQ(c.expert_begin(r), c.expert_end(r - 1));

  // One scale per 128-element block, rounded up.
  EXPECT_EQ(c.hidden_sf(), (7168 + 127) / 128);
  EXPECT_EQ(makeConfig(8, 128, 1, 6).hidden_sf(), 1);
  EXPECT_EQ(makeConfig(8, 128, 128, 6).hidden_sf(), 1);
  EXPECT_EQ(makeConfig(8, 128, 129, 6).hidden_sf(), 2);
}

} // namespace RcclUnitTesting
