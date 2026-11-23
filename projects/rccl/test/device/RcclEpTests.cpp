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
// is built on, the wave copies, combine's per-token ownership rule and its reduce's
// bit-exactness contract, the dispatch epilogue's and the expanded layout's handling
// of expert ids, and the layout arithmetic that decides where each region lands.
// The transfers of dispatch and combine need more than one rank and symmetric
// memory, so they are out of scope for this single-GPU binary.

#include "DeviceTestBase.hpp"
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "device/combine.h"
#include "device/dispatch.h"
#include "device/ep_common.h"
#include "device/expand.h"
#include "device/hip_prims.h"
#include "include/ep_arch.h"
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
// Combine ownership
// ---------------------------------------------------------------------------

// One token per block, one wave: lane k gets slot k's local expert (-1: another rank's), and
// every lane joins the ballot, as in combine's grouped push.
__global__ void kernelOwnsAnySlot(const int32_t* __restrict__ experts, int numTopk, int* __restrict__ owns) {
  const int t    = static_cast<int>(blockIdx.x);
  const int lane = rccl_ep::get_lane_idx();
  const bool o   = rccl_ep::owns_any_slot(lane < numTopk ? experts[t * numTopk + lane] : -1);
  if (lane == 0) owns[t] = o ? 1 : 0;
}

TEST_F(RcclEpWaveTest, OwnsAnySlotOfTheToken) {
  constexpr int kTopk = 4;
  // A rank owns a token when any slot names one of its experts, wherever that slot is: the
  // origin then reads this rank's row, so the grouped push writes it.
  const std::vector<int32_t> h_experts = {
      3,  -1, -1, -1,  // only slot 0
      -1, -1, -1, 7,   // only slot 3
      -1, -1, -1, -1,  // none owned
      1,  5,  2,  0,   // all owned
      6,  6,  -1, -1,  // an expert id repeated within the token
  };
  const std::vector<int> want = {1, 1, 0, 1, 1};
  const int tokens            = static_cast<int>(want.size());

  DeviceBuffer<int32_t> d_experts(h_experts.size());
  DeviceBuffer<int> d_owns(tokens);
  d_experts.copyFrom(h_experts);
  kernelOwnsAnySlot<<<tokens, rccl_ep::kWarpSize>>>(d_experts.ptr, kTopk, d_owns.ptr);
  syncAndCheck();
  const std::vector<int> owns = d_owns.copyTo();
  for (int t = 0; t < tokens; ++t) EXPECT_EQ(owns[t], want[t]) << "token " << t;
}

// ---------------------------------------------------------------------------
// The combine reduce's bit-exactness contract (reduce_rows)
// ---------------------------------------------------------------------------

// One wave reduces `nrow` rows of `src`, listed in `rows`, into dst, as both combine paths do. The
// token's bias rows start biasOff elements into bias0 and bias1.
template <int kMaxRows>
__global__ void kernelReduceRows(rccl_ep::bf16_t* __restrict__ dst, const rccl_ep::bf16_t* __restrict__ src,
                                 const int32_t* __restrict__ rows, int nrow, const rccl_ep::bf16_t* __restrict__ bias0,
                                 const rccl_ep::bf16_t* __restrict__ bias1, size_t biasOff, int hidden) {
  int32_t r[rccl_ep::kMaxTopk];
  for (int j = 0; j < rccl_ep::kMaxTopk; ++j) r[j] = j < nrow ? rows[j] : 0;
  rccl_ep::reduce_rows<kMaxRows>(dst, src, r, nrow, bias0, bias1, biasOff, hidden, rccl_ep::get_lane_idx());
}

// The same through reduce_rows_ballot: lane l offers row laneRows[l], and the set bits of `mask`
// pick the lanes whose rows are reduced.
__global__ void kernelReduceRowsBallot(rccl_ep::bf16_t* __restrict__ dst, const rccl_ep::bf16_t* __restrict__ src,
                                       const int32_t* __restrict__ laneRows, unsigned long long mask,
                                       const rccl_ep::bf16_t* __restrict__ bias0,
                                       const rccl_ep::bf16_t* __restrict__ bias1, size_t biasOff, int hidden) {
  const int lane = rccl_ep::get_lane_idx();
  rccl_ep::reduce_rows_ballot(dst, src, mask, laneRows[lane], bias0, bias1, biasOff, hidden, lane);
}

// Source rows of very different binades and signs, exact in bf16. Row 3 is the exact negative of
// row 0, and both sit near 2^20, far above the rest, so reordering the adds or rounding more than
// once changes the bf16 result.
static float reduceSrcValue(int r, int h) {
  static const int kExp[16] = {20, 1, -2, 20, 3, 0, -4, 2, -1, 4, -3, 1, -6, 5, 0, -5};
  const int rr = r == 3 ? 0 : r;
  const float m = 1.0f + static_cast<float>((rr * 13 + h * 7) % 128) / 128.0f;
  float v = m * std::ldexp(1.0f, kExp[rr] + (rr == 0 ? h % 2 : 0));
  if (rr != 0 && (r + h) % 3 == 0) v = -v;
  return r == 3 ? -v : v;
}

// The contract, on the host: fp32 from 0, bias0, bias1 (the token's rows, biasOff elements in),
// then the rows in the order listed; one rounding to bf16 at the end.
static float reduceReference(const std::vector<rccl_ep::bf16_t>& src, const int32_t* rows, int nrow,
                             const std::vector<rccl_ep::bf16_t>* bias0, const std::vector<rccl_ep::bf16_t>* bias1,
                             size_t biasOff, int h, int hidden) {
  float acc = 0.f;
  if (bias0) acc += __bfloat162float((*bias0)[biasOff + h]);
  if (bias1) acc += __bfloat162float((*bias1)[biasOff + h]);
  for (int j = 0; j < nrow; ++j) acc += __bfloat162float(src[static_cast<size_t>(rows[j]) * hidden + h]);
  return __bfloat162float(__float2bfloat16(acc));
}

TEST_F(RcclEpWaveTest, ReduceRowsFollowsTheContract) {
  constexpr int kHidden   = 512;  // one 8-element vector per lane
  constexpr int kSrcRows  = 16;
  constexpr int kBiasRows = 3;  // the bias rows of tokens 0..2
  constexpr int kFast     = rccl_ep::kFastRows;
  const int32_t kRows[12] = {5, 0, 9, 3, 12, 1, 7, 14, 2, 11, 4, 15};
  static_assert(kFast <= 12, "the fast-path cases take kFastRows of kRows");

  std::vector<rccl_ep::bf16_t> h_src(static_cast<size_t>(kSrcRows) * kHidden), h_b0(kBiasRows * kHidden),
      h_b1(kBiasRows * kHidden);
  for (int r = 0; r < kSrcRows; ++r) {
    for (int h = 0; h < kHidden; ++h) {
      h_src[static_cast<size_t>(r) * kHidden + h] = __float2bfloat16(reduceSrcValue(r, h));
    }
  }
  // Token b's bias0 row is 2^b times token 0's, and its bias1 row a rotation of token 0's.
  for (int b = 0; b < kBiasRows; ++b) {
    for (int h = 0; h < kHidden; ++h) {
      h_b0[b * kHidden + h] = __float2bfloat16(std::ldexp(1.0f + static_cast<float>(h % 64) / 64.0f, b - 3));
      h_b1[b * kHidden + h] = __float2bfloat16(-std::ldexp(1.0f + static_cast<float>((h + 5 * b) % 32) / 32.0f, -5));
    }
  }

  DeviceBuffer<rccl_ep::bf16_t> d_src(h_src.size()), d_b0(h_b0.size()), d_b1(h_b1.size()), d_out(kHidden);
  DeviceBuffer<int32_t> d_rows(12);
  d_src.copyFrom(h_src);
  d_b0.copyFrom(h_b0);
  d_b1.copyFrom(h_b1);
  d_rows.copyFrom(std::vector<int32_t>(kRows, kRows + 12));

  // d_out against the contract over `rows`, every element.
  auto expectContract = [&](const int32_t* rows, int nrow, bool biases, size_t biasOff) {
    const std::vector<rccl_ep::bf16_t> h_out = d_out.copyTo();
    int mismatches = 0;
    for (int h = 0; h < kHidden; ++h) {
      const float want = reduceReference(h_src, rows, nrow, biases ? &h_b0 : nullptr, biases ? &h_b1 : nullptr,
                                         biasOff, h, kHidden);
      if (!(__bfloat162float(h_out[h]) == want) && mismatches++ < 8) {
        ADD_FAILURE() << "element " << h << ": got " << __bfloat162float(h_out[h]) << ", expected " << want;
      }
    }
    EXPECT_EQ(mismatches, 0);
  };

  struct Case {
    const char* what;
    int maxRows;  // kFastRows: the register fast path; 0: any number of rows
    bool biases;
    int nrow;
    int biasRow;  // the token whose bias rows are added
  };
  const Case cases[] = {
      {"fast path full, biases", kFast, true, kFast, 0},
      {"3 rows", kFast, false, 3, 0},
      {"no rows, biases only", kFast, true, 0, 0},
      {"12 rows, biases", 0, true, 12, 0},
      {"12 rows", 0, false, 12, 0},
      // Token 2's biases, which differ from token 0's at every element (checked below), so a reduce
      // that ignored bias_off fails these.
      {"fast path full, token 2's biases", kFast, true, kFast, 2},
      {"no rows, token 2's biases only", kFast, true, 0, 2},
      {"12 rows, token 2's biases", 0, true, 12, 2},
      {"no rows, token 2's biases only, any rows", 0, true, 0, 2},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.what);
    const rccl_ep::bf16_t* b0 = c.biases ? d_b0.ptr : nullptr;
    const rccl_ep::bf16_t* b1 = c.biases ? d_b1.ptr : nullptr;
    const size_t biasOff      = static_cast<size_t>(c.biasRow) * kHidden;
    d_out.zero();
    using Kernel = void (*)(rccl_ep::bf16_t*, const rccl_ep::bf16_t*, const int32_t*, int, const rccl_ep::bf16_t*,
                            const rccl_ep::bf16_t*, size_t, int);
    const Kernel kernel = c.maxRows == kFast ? kernelReduceRows<kFast> : kernelReduceRows<0>;
    hipLaunchKernelGGL(kernel, dim3(1), dim3(rccl_ep::kWarpSize), 0, 0, d_out.ptr, d_src.ptr, d_rows.ptr, c.nrow, b0,
                       b1, biasOff, kHidden);
    syncAndCheck();
    expectContract(kRows, c.nrow, c.biases, biasOff);
  }

  // reduce_rows_ballot takes the rows of the mask's set lanes in ascending lane order: 12 of lanes
  // 0..15, with gaps, each lane offering a different row, plus token 2's biases.
  {
    SCOPED_TRACE("reduce_rows_ballot, 12 of 16 lanes");
    constexpr unsigned long long kMask = 0xDBBBull;  // lanes 0 1 3 4 5 7 8 9 11 12 14 15
    std::vector<int32_t> laneRows(rccl_ep::kWarpSize), packed;
    for (int l = 0; l < rccl_ep::kWarpSize; ++l) {
      laneRows[l] = (l * 5 + 3) % kSrcRows;
      if ((kMask >> l) & 1) packed.push_back(laneRows[l]);
    }
    ASSERT_EQ(packed.size(), 12u);
    DeviceBuffer<int32_t> d_laneRows(rccl_ep::kWarpSize);
    d_laneRows.copyFrom(laneRows);
    d_out.zero();
    kernelReduceRowsBallot<<<1, rccl_ep::kWarpSize>>>(d_out.ptr, d_src.ptr, d_laneRows.ptr, kMask, d_b0.ptr, d_b1.ptr,
                                                      2 * kHidden, kHidden);
    syncAndCheck();
    expectContract(packed.data(), 12, true, 2 * kHidden);
    // The rows make the packing order visible: in descending lane order the sum differs somewhere.
    const std::vector<int32_t> descending(packed.rbegin(), packed.rend());
    int orderSensitive = 0;
    for (int h = 0; h < kHidden; ++h) {
      orderSensitive += reduceReference(h_src, packed.data(), 12, &h_b0, &h_b1, 2 * kHidden, h, kHidden) !=
                        reduceReference(h_src, descending.data(), 12, &h_b0, &h_b1, 2 * kHidden, h, kHidden);
    }
    EXPECT_GT(orderSensitive, 0);
  }

  // The data does tell orders apart: the eight rows summed in reverse differ from the contract's
  // result somewhere.
  int reversed = 0;
  const int32_t kReversed[8] = {kRows[7], kRows[6], kRows[5], kRows[4], kRows[3], kRows[2], kRows[1], kRows[0]};
  for (int h = 0; h < kHidden; ++h) {
    reversed += reduceReference(h_src, kRows, 8, nullptr, nullptr, 0, h, kHidden) !=
                reduceReference(h_src, kReversed, 8, nullptr, nullptr, 0, h, kHidden);
  }
  EXPECT_GT(reversed, 0);
  // And biases tell tokens apart: token 2's differ from token 0's at every element.
  int sameBias = 0;
  for (int h = 0; h < kHidden; ++h) {
    sameBias += reduceReference(h_src, kRows, 0, &h_b0, &h_b1, 2 * kHidden, h, kHidden) ==
                reduceReference(h_src, kRows, 0, &h_b0, &h_b1, 0, h, kHidden);
  }
  EXPECT_EQ(sameBias, 0);
}

__global__ void kernelWaveCopy(const rccl_ep::bf16_t* __restrict__ src, rccl_ep::bf16_t* __restrict__ dst,
                               int n) {
  rccl_ep::wave_copy(dst, src, n);
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

template <bool kNtLoad, bool kNtStore>
__global__ void kernelWaveCopyLocal(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int n) {
  rccl_ep::wave_copy_local<kNtLoad, kNtStore, 4>(dst, src, n);
}

TEST_F(RcclEpWaveTest, WaveCopyLocalMovesEveryByte) {
  // Unrolled 4 deep, one wave moves 4 x 64 dwordx4 = 4096 B per unrolled trip. 5 B is the byte tail
  // alone, 16 B one vector of the remainder loop, 4096 B one unrolled trip, 14336 B (a 7168-element
  // bf16 row) three trips and 128 remainder vectors, and 14337 B adds a byte of tail.
  const std::vector<int> sizes = {5, 16, 4096, 14336, 14337};
  constexpr int kGuard = 64;  // bytes past the end that must stay untouched
  constexpr uint8_t kFill = 0xA5;
  using Kernel = void (*)(const uint8_t*, uint8_t*, int);
  const struct {
    const char* what;
    Kernel kernel;
  } hints[] = {
      {"plain", kernelWaveCopyLocal<false, false>},
      {"nontemporal loads", kernelWaveCopyLocal<true, false>},
      {"nontemporal stores", kernelWaveCopyLocal<false, true>},
      {"nontemporal loads and stores", kernelWaveCopyLocal<true, true>},
  };

  for (const auto& hint : hints) {
    for (int n : sizes) {
      SCOPED_TRACE(testing::Message() << hint.what << ", n = " << n);
      // Each 4-byte word holds its own index, so a vector copied to the wrong place shows.
      std::vector<uint8_t> h_src(n);
      for (int i = 0; i < n; ++i) h_src[i] = static_cast<uint8_t>((i / 4) >> (8 * (i % 4)));
      DeviceBuffer<uint8_t> d_src(n), d_dst(n + kGuard);
      d_src.copyFrom(h_src);
      d_dst.copyFrom(std::vector<uint8_t>(n + kGuard, kFill));
      hipLaunchKernelGGL(hint.kernel, dim3(1), dim3(rccl_ep::kWarpSize), 0, 0, d_src.ptr, d_dst.ptr, n);
      syncAndCheck();
      const std::vector<uint8_t> h_dst = d_dst.copyTo();
      int mismatches = 0;
      for (int i = 0; i < n + kGuard; ++i) {
        const uint8_t want = i < n ? h_src[i] : kFill;
        if (h_dst[i] != want && mismatches++ < 8) {
          ADD_FAILURE() << "byte " << i << ": got " << int(h_dst[i]) << ", expected " << int(want);
        }
      }
      EXPECT_EQ(mismatches, 0);
    }
  }
}

TEST_F(RcclEpWaveTest, PublishArchWritesOwnSlotInEveryPeerRegion) {
  constexpr int kRanks = 3;
  constexpr uint8_t kFill = 0xA5;

  rccl_ep::EpConfig c{};
  c.num_ranks               = kRanks;
  c.num_experts             = kRanks * 2;
  c.num_topk                = 2;
  c.hidden                  = 128;
  c.num_max_tokens_per_rank = 2;
  const rccl_ep::EpWindowLayout l = rccl_ep::EpWindowLayout::make(c, sizeof(uint16_t));

  std::vector<DeviceBuffer<uint8_t>> windows;
  windows.reserve(kRanks);
  std::vector<rccl_ep::WindowView> h_views(kRanks);
  for (int p = 0; p < kRanks; ++p) {
    windows.emplace_back(l.total_bytes);
    windows[p].copyFrom(std::vector<uint8_t>(l.total_bytes, kFill));
    h_views[p].base = windows[p].ptr;
    h_views[p].l    = l;
  }
  DeviceBuffer<rccl_ep::WindowView> d_views(kRanks);
  d_views.copyFrom(h_views);

  // Distinct NUL-padded strings per rank; the padding (0x00) differs from the fill, so a short copy shows.
  const int publishers[2]  = {1, 2};
  const char* const arch[] = {"gfx942:sramecc+:xnack-", "gfx950"};
  std::vector<std::vector<char>> strs(2, std::vector<char>(rccl_ep::kArchLen, '\0'));
  for (int i = 0; i < 2; ++i) memcpy(strs[i].data(), arch[i], strlen(arch[i]));

  for (int i = 0; i < 2; ++i) {
    rccl_ep::EpConfig cr = c;
    cr.rank              = publishers[i];
    DeviceBuffer<char> d_mine(rccl_ep::kArchLen);
    d_mine.copyFrom(strs[i]);
    rccl_ep::k_ep_publish_arch<<<kRanks, rccl_ep::kArchLen>>>(cr, d_views.ptr, d_mine.ptr);
    syncAndCheck();
  }

  for (int p = 0; p < kRanks; ++p) {
    const std::vector<uint8_t> h = windows[p].copyTo();
    ASSERT_EQ(h.size(), l.total_bytes);
    for (size_t b = 0; b < h.size(); ++b) {
      uint8_t expected = kFill;
      for (int i = 0; i < 2; ++i) {
        const size_t begin = l.off_arch + static_cast<size_t>(publishers[i]) * rccl_ep::kArchLen;
        if (b >= begin && b < begin + rccl_ep::kArchLen) expected = static_cast<uint8_t>(strs[i][b - begin]);
      }
      ASSERT_EQ(h[b], expected) << "window " << p << " byte " << b << " (off_arch " << l.off_arch << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Expert ids outside this rank's experts
// ---------------------------------------------------------------------------

// The dispatch epilogue on rank 1 of 2, with 8 experts (rank 1's are 4..7), top-4 and one token
// staged by each source. It copies each staged row out and rebases its ids into local ones, and any
// id outside 4..7 becomes -1, whatever the sender wrote, so no local id reaches experts_per_rank.
TEST_F(RcclEpWaveTest, CopyEpilogueDropsIdsOutsideOwnExperts) {
  constexpr int kTopk = 4, kHidden = 128, kSources = 2;
  rccl_ep::EpConfig c{};
  c.num_ranks               = kSources;
  c.rank                    = 1;
  c.num_experts             = 8;
  c.num_topk                = kTopk;
  c.hidden                  = kHidden;
  c.num_max_tokens_per_rank = 2;
  const rccl_ep::EpWindowLayout l = rccl_ep::EpWindowLayout::make(c, sizeof(rccl_ep::bf16_t));

  // Source 0's token names rank 0's expert 0, expert 7, an id past the last expert and -1; source 1's
  // names 4, rank 0's expert 3, another id past the last, and 5.
  const int32_t staged[kSources][kTopk] = {{0, 7, 8, -1}, {4, 3, 12, 5}};
  const int32_t want[kSources][kTopk]   = {{-1, 3, -1, -1}, {0, -1, -1, 1}};
  float weights[kSources][kTopk];
  int32_t srcIdx[kSources];
  std::vector<rccl_ep::bf16_t> h_x(kSources * kHidden);
  std::vector<uint8_t> h_win(l.total_bytes, 0);
  auto put = [&](size_t off, const void* v, size_t bytes) { memcpy(h_win.data() + off, v, bytes); };
  for (int src = 0; src < kSources; ++src) {
    const size_t s    = rccl_ep::EpWindowLayout::slot(c, src, 0);
    const int32_t one = 1;
    srcIdx[src]       = src * c.num_max_tokens_per_rank;
    for (int k = 0; k < kTopk; ++k) weights[src][k] = 0.25f * static_cast<float>(src * kTopk + k + 1);
    for (int h = 0; h < kHidden; ++h) h_x[src * kHidden + h] = __float2bfloat16(static_cast<float>(src * 128 + h));
    put(l.off_counts + src * sizeof(int32_t), &one, sizeof(one));
    put(l.off_x + s * kHidden * sizeof(rccl_ep::bf16_t), &h_x[src * kHidden], kHidden * sizeof(rccl_ep::bf16_t));
    put(l.off_topk_idx + s * kTopk * sizeof(int32_t), staged[src], sizeof(staged[src]));
    put(l.off_topk_w + s * kTopk * sizeof(float), weights[src], sizeof(weights[src]));
    put(l.off_src_idx + s * sizeof(int32_t), &srcIdx[src], sizeof(int32_t));
  }
  DeviceBuffer<uint8_t> d_win(l.total_bytes);
  d_win.copyFrom(h_win);
  rccl_ep::WindowView view{};
  view.base = d_win.ptr;
  view.l    = l;

  DeviceBuffer<int32_t> d_offsets(kSources), d_topk(kSources * kTopk), d_src(kSources);
  DeviceBuffer<float> d_w(kSources * kTopk);
  DeviceBuffer<rccl_ep::bf16_t> d_x(kSources * kHidden);
  d_offsets.copyFrom(std::vector<int32_t>{0, 1});
  d_topk.copyFrom(std::vector<int32_t>(kSources * kTopk, 99));
  rccl_ep::dispatch_copy_epilogue_impl<rccl_ep::bf16_t><<<dim3(kSources, 1), rccl_ep::kWarpSize>>>(
      c, view, d_offsets.ptr, d_x.ptr, nullptr, d_topk.ptr, d_w.ptr, d_src.ptr);
  syncAndCheck();

  const std::vector<int32_t> gotTopk      = d_topk.copyTo();
  const std::vector<float> gotW           = d_w.copyTo();
  const std::vector<int32_t> gotSrc       = d_src.copyTo();
  const std::vector<rccl_ep::bf16_t> gotX = d_x.copyTo();
  for (int i = 0; i < kSources; ++i) {
    for (int k = 0; k < kTopk; ++k) {
      EXPECT_EQ(gotTopk[i * kTopk + k], want[i][k]) << "row " << i << " slot " << k << " (id " << staged[i][k] << ")";
      EXPECT_EQ(gotW[i * kTopk + k], weights[i][k]) << "row " << i << " slot " << k;
    }
    EXPECT_EQ(gotSrc[i], srcIdx[i]) << "row " << i;
    EXPECT_EQ(memcmp(&gotX[i * kHidden], &h_x[i * kHidden], kHidden * sizeof(rccl_ep::bf16_t)), 0) << "row " << i;
  }
}

// ep_expand_build's kernels, launched as it launches them, on local ids from its caller with 4
// experts per rank: ids 4 and 7 are past the last expert. They select no expert, like -1 and -2:
// the row map holds -1 at their slots and the valid slots get the rows they would get without
// them, and k_expand_hist writes nothing past its rows. offsets and hist carry a poisoned tail
// past their ends, so an id used as an index there shows.
TEST_F(RcclEpWaveTest, ExpandIgnoresLocalIdsPastLastExpert) {
  constexpr int kTopk = 4, kRecv = 2, kEpr = 4, kTail = 2 * kEpr;
  constexpr int32_t kPoison = 1 << 20;
  rccl_ep::EpConfig c{};
  c.num_ranks               = 2;
  c.rank                    = 0;
  c.num_experts             = 2 * kEpr;
  c.num_topk                = kTopk;
  c.hidden                  = 128;
  c.num_max_tokens_per_rank = kRecv;
  const std::vector<int32_t> h_topk = {0, kEpr, 1, -1, kEpr + 3, 2, 2, -2};
  // Experts 0..3 have 1, 1, 2 and 0 slots; at alignment 1 their rows start at 0, 1, 2 and 4.
  const std::vector<int32_t> wantCounts = {1, 1, 2, 0};
  const std::vector<int32_t> wantMap    = {0, -1, 1, -1, -1, 2, 3, -1};

  DeviceBuffer<int32_t> d_topk(h_topk.size()), d_counts(kEpr), d_offsets(kEpr + kTail), d_psum(kEpr), d_total(1),
      d_hist(kRecv * kEpr + kTail), d_map(kRecv * kTopk);
  d_topk.copyFrom(h_topk);
  d_offsets.copyFrom(std::vector<int32_t>(kEpr + kTail, kPoison));
  d_hist.copyFrom(std::vector<int32_t>(kRecv * kEpr + kTail, kPoison));
  rccl_ep::k_expand_count<<<kEpr, 256>>>(c, d_topk.ptr, kRecv, d_counts.ptr);
  rccl_ep::k_expand_scan<<<1, 1>>>(c, 1, d_counts.ptr, d_offsets.ptr, d_psum.ptr, d_total.ptr);
  rccl_ep::k_expand_hist<<<1, 256>>>(c, d_topk.ptr, kRecv, d_hist.ptr);
  rccl_ep::k_expand_base<<<kEpr, rccl_ep::kWarpSize>>>(c, kRecv, d_hist.ptr);
  rccl_ep::k_expand_map<<<1, 256>>>(c, d_topk.ptr, kRecv, d_offsets.ptr, d_hist.ptr, d_map.ptr);
  syncAndCheck();

  EXPECT_EQ(d_counts.copyTo(), wantCounts);
  EXPECT_EQ(d_total.download(), 4);
  EXPECT_EQ(d_map.copyTo(), wantMap);
  const std::vector<int32_t> hist = d_hist.copyTo();
  for (int j = kRecv * kEpr; j < kRecv * kEpr + kTail; ++j) EXPECT_EQ(hist[j], kPoison) << "hist entry " << j;
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

TEST(RcclEpLayoutTest, OwnerOfBoundsTheExpertId) {
  const rccl_ep::EpConfig c = makeConfig(8, 128, 1024, 4);  // 256 experts, 32 per rank

  EXPECT_EQ(rccl_ep::owner_of(c, 0), 0);
  EXPECT_EQ(rccl_ep::owner_of(c, 31), 0);
  EXPECT_EQ(rccl_ep::owner_of(c, 32), 1);
  EXPECT_EQ(rccl_ep::owner_of(c, 255), 7);
  // Out of range on either side: no owner, rather than a rank past the last.
  EXPECT_EQ(rccl_ep::owner_of(c, 256), -1);
  EXPECT_EQ(rccl_ep::owner_of(c, 1 << 30), -1);
  EXPECT_EQ(rccl_ep::owner_of(c, -1), -1);
  EXPECT_EQ(rccl_ep::owner_of(c, -33), -1);
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

  EXPECT_EQ(c.experts_per_rank(), 32);
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
