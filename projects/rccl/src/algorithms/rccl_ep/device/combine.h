/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Combine: push locally-owned expert outputs back to the origin, and
 * accumulate them there in strict top-k order in fp32.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_COMBINE_H_
#define RCCL_EP_COMBINE_H_

#include "device/ep_common.h"

namespace rccl_ep {

// Combine, non-hybrid intranode. The sum along num_topk runs in STRICT index
// order in float32, which is why the staging region is per (slot, topk) rather
// than pre-reduced per rank: with A owning slots {1,3} and B {0,2} the order is
// B0, A1, B2, A3, interleaved across ranks. Pre-reducing reorders that, and
// float addition is not associative, so bit-exactness fails.

// The combine reduce (reduce_rows), shared by the sender's grouped partials and the origin's
// epilogue so the two agree bit for bit: fp32 from 0.f, then bias0, bias1 and the rows in the order
// listed, rounded to bf16 once; reordering the terms changes the result. One dwordx4 (kVecElems
// bf16) per lane; needs hidden % kVecElems == 0, which ep_create enforces.
constexpr int kVecElems = 8;
static_assert(kVecElems * sizeof(bf16_t) == sizeof(uint4), "one vector of a row is one dwordx4");
// Rows of a token the register fast path takes; a token with more goes to reduce_rows_ballot.
constexpr int kFastRows = 8;

__device__ __forceinline__ void add_vec(float (&acc)[kVecElems], const uint4& y) {
  const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
#pragma unroll
  for (int u = 0; u < kVecElems; ++u) acc[u] += __bfloat162float(yy[u]);
}

// Rounds acc to bf16, the contract's one rounding, and stores it as one dwordx4 at p.
__device__ __forceinline__ void store_vec(bf16_t* p, const float (&acc)[kVecElems]) {
  uint4 o;
  bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
  for (int u = 0; u < kVecElems; ++u) oo[u] = __float2bfloat16(acc[u]);
  *reinterpret_cast<uint4*>(p) = o;
}

// The contract for one row: dst = bf16(bias0 + bias1 + the nrow rows src + rows[j] * hidden), one
// vector per lane per trip. dst addresses the output row; bias0 and bias1 are bias TENSORS, either
// may be null, and the token's bias row starts bias_off elements in.
//
// kMaxRows > 0: at most kMaxRows rows, in fixed-size arrays the caller fills, which stay in
// registers, and every row load of a vector is issued before the first add rather than one at a
// time. Unrolling the trip loop would double the live row loads and push a 1024-thread block past
// its 128-VGPR budget into spills. kMaxRows == 0: any number of rows, loaded one at a time, for
// reduce_rows_ballot.
template <int kMaxRows>
__device__ __forceinline__ void reduce_rows(bf16_t* dst, const bf16_t* __restrict__ src, const int32_t* rows, int nrow,
                                            const bf16_t* __restrict__ bias0, const bf16_t* __restrict__ bias1,
                                            size_t bias_off, int hidden, int lane) {
  EP_DEBUG_ASSERT(hidden % kVecElems == 0);
  const int nvec = hidden / kVecElems;
#pragma unroll 1
  for (int v = lane; v < nvec; v += kWarpSize) {
    const int h0 = v * kVecElems;
    uint4 y[kMaxRows > 0 ? kMaxRows : 1];
    if constexpr (kMaxRows > 0) {
#pragma unroll
      for (int j = 0; j < kMaxRows; ++j) {
        if (j < nrow) y[j] = *reinterpret_cast<const uint4*>(src + (size_t)rows[j] * hidden + h0);
      }
    }
    float acc[kVecElems];
#pragma unroll
    for (int u = 0; u < kVecElems; ++u) acc[u] = 0.f;
    if (bias0) add_vec(acc, *reinterpret_cast<const uint4*>(bias0 + bias_off + h0));
    if (bias1) add_vec(acc, *reinterpret_cast<const uint4*>(bias1 + bias_off + h0));
    if constexpr (kMaxRows > 0) {
#pragma unroll
      for (int j = 0; j < kMaxRows; ++j) {
        if (j < nrow) add_vec(acc, y[j]);
      }
    } else {
      for (int j = 0; j < nrow; ++j) add_vec(acc, *reinterpret_cast<const uint4*>(src + (size_t)rows[j] * hidden + h0));
    }
    store_vec(dst + h0, acc);
  }
}

// The grouped push's ownership rule for one token. Lane k (k < num_topk) holds slot k's local
// expert `e`, -1 for another rank's. owns_any_slot: this rank owns a slot of the token, so the
// origin reads this rank's row, which is therefore written even when no slot resolved to an input
// row (as zeros).
__device__ __forceinline__ bool owns_any_slot(int e) {
  return __ballot(e >= 0) != 0;
}

// reduce_rows for a token with more than kFastRows rows, on either side: lane k holds slot k's row
// in lane_row, and the set bits of `mask` (a ballot) pick the slots, reduced in ascending k. Out of
// line so its runtime-indexed row array does not count against the fast path's register budget.
// The whole wave calls it, since it reads other lanes' rows.
__device__ __noinline__ void reduce_rows_ballot(bf16_t* dst, const bf16_t* __restrict__ src, unsigned long long mask,
                                                int32_t lane_row, const bf16_t* __restrict__ bias0,
                                                const bf16_t* __restrict__ bias1, size_t bias_off, int hidden,
                                                int lane) {
  int32_t row[kMaxTopk];
  int nrow = 0;
  for (unsigned long long m = mask; m != 0; m &= m - 1) row[nrow++] = __shfl(lane_row, __builtin_ctzll(m), kWarpSize);
  reduce_rows<0>(dst, src, row, nrow, bias0, bias1, bias_off, hidden, lane);
}

// Expert side: push locally-owned values back to the origin, preserving slot k
// so the origin accumulates in order. Input layout is selected by `grouped` and
// `row_map`: grouped=1 is one already-reduced row per token pushed to k_last;
// grouped=0 is [nrecv, num_topk, hidden] at row i*K+k, or the expanded layout
// addressed through row_map.
__global__ void combine_impl(EpConfig cfg,
                             const bf16_t* __restrict__ in_y,        // [nrecv, hidden] or [rows, hidden]
                             const float* __restrict__ in_w,         // weights or nullptr
                             const int32_t* __restrict__ row_map,    // [nrecv, num_topk] or nullptr
                             const int32_t* __restrict__ recv_topk,  // [nrecv, num_topk] local ids, -1 outside
                             const int32_t* __restrict__ src_idx,    // [nrecv]
                             int num_recv, int grouped, WindowView self, WindowView* __restrict__ peer_views) {
  // Grid mirrors dispatch: block x owns one SOURCE rank, so the peer view is
  // loaded once per CTA and every write goes to a single peer. Parallelism is
  // gridDim.y * nwarps and only gridDim.y follows the budget; see kEpWaves.
  const int src_rank = blockIdx.x;
  if (src_rank >= cfg.num_ranks) return;  // block-uniform, so the barrier below is safe
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk;

  // counts() lives in the peer window, so each term of this prefix sum is a
  // global load. Every thread needs the same two scalars, and the block is
  // kEpWaves wide, so computing it once and broadcasting through LDS trades
  // blockDim x src_rank loads for one barrier.
  __shared__ int s_begin, s_end;
  if (threadIdx.x == 0) {
    int b = 0;
    for (int r = 0; r < src_rank; ++r) b += self.counts()[r];
    s_begin = b;
    s_end = b + self.counts()[src_rank];
  }
  __syncthreads();
  const int begin = s_begin, end = s_end;

  const WindowView w = peer_views[src_rank];
  const int stride = nwarps * gridDim.y;

  for (int i = begin + blockIdx.y * nwarps + warp; i < end && i < num_recv; i += stride) {
    const int g = src_idx[i];
    const int src_tok = g % cfg.num_max_tokens_per_rank;
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, src_tok);

    if (grouped && row_map != nullptr) {
      // Grouped reduction over an EXPANDED input: form the per-rank partial
      // here, summing this rank's owned rows in ascending k into one row. The
      // origin folds that row at the last top-k position this rank owns. fp32
      // with a single rounding at the end; bf16 accumulation would not match.
      // Lane k resolves slot k, and its rank among the valid lanes is its
      // position in that order.
      int e = -1, r = -1;
      if (lane < K) {
        e = recv_topk[(size_t)i * K + lane];
        if (e >= 0) r = row_map[(size_t)i * K + lane];
      }
      const bool valid = r >= 0;  // r >= 0 implies lane < K and e >= 0
      const unsigned long long vm = __ballot(valid);
      const int nrow = __popcll(vm);
      const bool owns = owns_any_slot(e);  // false implies nrow == 0
      if (owns && nrow <= kFastRows) {
        const int pos = __popcll(vm & __lanemask_lt());
        int32_t rows[kFastRows];
#pragma unroll
        for (int j = 0; j < kFastRows; ++j) {
          // src is wave-uniform (from a ballot), so readlane leaves the row in a scalar
          // register, off the vector budget the reduce needs for its loads.
          const unsigned long long mj = __ballot(valid && pos == j);
          const int src = mj ? __builtin_ctzll(mj) : 0;
          rows[j] = __builtin_amdgcn_readlane(r, src);
        }
        reduce_rows<kFastRows>(w.y() + slot * cfg.hidden, in_y, rows, nrow, nullptr, nullptr, 0, cfg.hidden, lane);
      } else if (owns) {
        reduce_rows_ballot(w.y() + slot * cfg.hidden, in_y, vm, r, nullptr, nullptr, 0, cfg.hidden, lane);
      }
      if (in_w && valid) w.cw()[slot * K + lane] = in_w[r];
    } else if (grouped) {
      // Grouped mode sends ONE row per token, so index by slot rather than by
      // (slot, k). The expanded layout is K times larger, and scattering single
      // rows across it is far slower than a dense write.
      if (owns_any_slot(lane < K ? recv_topk[(size_t)i * K + lane] : -1)) {
        wave_copy(w.y() + slot * cfg.hidden, in_y + (size_t)i * cfg.hidden, cfg.hidden);
      }
      // The origin needs its own top-k weights back; every rank that received
      // this token holds an identical copy of the full row, so whichever
      // arrives last wins and the result is the same either way.
      if (in_w) {
        for (int k = lane; k < K; k += kWarpSize) w.cw()[slot * K + k] = in_w[(size_t)i * K + k];
      }
    } else {
      for (int k = 0; k < K; ++k) {
        if (recv_topk[(size_t)i * K + k] < 0) continue;  // not ours
        const int row = row_map ? row_map[(size_t)i * K + k] : (int)(i * K + k);
        if (row < 0) continue;
        wave_copy(w.y() + (slot * K + k) * cfg.hidden, in_y + (size_t)row * cfg.hidden, cfg.hidden);
        if (in_w && lane == 0) w.cw()[slot * K + k] = in_w[row];
      }
    }
  }
  // No completion flag: the caller barriers between send and receive, and every
  // slot the receiver reads is one some rank is guaranteed to have written --
  // a token only reaches a rank that owns one of its experts.
}

// Origin side for token t: accumulate across topk slots in strict order, in float32.
// bias0/bias1 are the accumulator's initial value and must seed it BEFORE the
// topk sum so they share the rounding sequence.
// `grouped` mirrors the sender: only the owner's last slot is populated.
__device__ __forceinline__ void combine_reduce_token(const EpConfig& cfg, const WindowView& self,
                                                     const int32_t* __restrict__ topk_idx, int t, int grouped,
                                                     const bf16_t* __restrict__ bias0, const bf16_t* __restrict__ bias1,
                                                     bf16_t* __restrict__ out, float* __restrict__ out_w, int lane) {
  const int K = cfg.num_topk;
  const size_t tbase = (size_t)t * cfg.hidden;

  // Lane k decides slot k: the expert id is in range and, when grouped, no LATER
  // slot has the same owner (the sender pushed that owner's row to its last slot).
  // The ballot keeps ascending k.
  const int own = owner_of(cfg, lane < K ? topk_idx[(size_t)t * K + lane] : -1);
  bool take_k = own >= 0;
  if (grouped) {
    for (int d = 1; d < K; ++d) {
      const int later = __shfl(own, lane + d, kWarpSize);
      if (lane + d < K && later == own) take_k = false;
    }
  }
  const size_t my_slot = take_k ? EpWindowLayout::slot(cfg, own, t) : 0;
  // The sender's layout: one row per slot when grouped, one per (slot, k) otherwise.
  const int32_t my_row = (int32_t)(grouped ? my_slot : (my_slot * K + lane));
  unsigned long long pending = __ballot(take_k);
  const int nrow = __popcll(pending);
  if (nrow <= kFastRows) {
    int32_t rows[kFastRows];
#pragma unroll
    for (int j = 0; j < kFastRows; ++j) {
      rows[j] = __shfl(my_row, pending ? __builtin_ctzll(pending) : 0, kWarpSize);
      pending &= pending - 1;
    }
    reduce_rows<kFastRows>(out + tbase, self.y(), rows, nrow, bias0, bias1, tbase, cfg.hidden, lane);
  } else {
    reduce_rows_ballot(out + tbase, self.y(), pending, my_row, bias0, bias1, tbase, cfg.hidden, lane);
  }

  if (out_w) {
    for (int k = lane; k < K; k += kWarpSize) {
      // Any owner of this token has the full weight row; a -1 or out-of-range slot was
      // never sent anywhere, so its weight is written as zero.
      const int owner = owner_of(cfg, topk_idx[(size_t)t * K + k]);
      out_w[(size_t)t * K + k] = (owner >= 0) ? self.cw()[EpWindowLayout::slot(cfg, owner, t) * K + k] : 0.f;
    }
  }
}

__global__ void combine_reduce_epilogue_impl(EpConfig cfg, WindowView self,
                                             const int32_t* __restrict__ topk_idx, // [ntok, num_topk] GLOBAL expert ids
                                             int num_tokens,
                                             const bf16_t* __restrict__ bias0, // [ntok, hidden] or nullptr
                                             const bf16_t* __restrict__ bias1, // [ntok, hidden] or nullptr
                                             int grouped,
                                             bf16_t* __restrict__ out, // [ntok, hidden]
                                             float* __restrict__ out_w) { // [ntok, num_topk] or nullptr
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int stride = nwarps * gridDim.x;
  // One wave per token, so gridDim.x * nwarps waves chase num_tokens, and this
  // loop is sensitive in BOTH directions: too few waves and each serialises
  // many tokens, too many and most retire without work while the block still
  // pays for them. Launched at kEpWaves, which is sized for the many-token
  // case; python/rccl_ep_capi.hip records what that costs when tokens are few.
  for (int t = blockIdx.x * nwarps + warp; t < num_tokens; t += stride) {
    combine_reduce_token(cfg, self, topk_idx, t, grouped, bias0, bias1, out, out_w, lane);
  }
}

} // namespace rccl_ep

#endif // RCCL_EP_COMBINE_H_
