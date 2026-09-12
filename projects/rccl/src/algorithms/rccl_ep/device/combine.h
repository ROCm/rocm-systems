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

// Sum `nrow` rows of `src`, named by `rows` and applied in the order given,
// into the single row `dst`, seeded by up to two optional bias rows. All four
// pointers address a ROW, not a tensor -- the caller has already applied its
// own token stride.
//
// Both combine paths reduce exactly this way and under the same contract:
// accumulate in fp32, round once at the end, and apply the terms as bias0,
// bias1, then rows in ascending k. That order is the bit-exactness guarantee,
// so it is defined here once rather than in each path.
//
// Vectorised at dwordx4: scalar 2-byte loads are an 8x loss of load width, and
// the split changes only which lane owns which element, never the order of
// additions applied to any single element.
__device__ __forceinline__ void reduce_rows_bf16(bf16_t* dst,
                                                 const bf16_t* __restrict__ src,
                                                 const int32_t* rows, int nrow,
                                                 const bf16_t* __restrict__ bias0,
                                                 const bf16_t* __restrict__ bias1,
                                                 int hidden, int lane) {
  constexpr int kPerVec = 8; // 8 bf16 = 16 B
  const int nvec = hidden / kPerVec;
  for (int v = lane; v < nvec; v += kWarpSize) {
    float acc[kPerVec];
    const int h0 = v * kPerVec;
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
    if (bias0) {
      const uint4 b = *reinterpret_cast<const uint4*>(bias0 + h0);
      const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
    }
    if (bias1) {
      const uint4 b = *reinterpret_cast<const uint4*>(bias1 + h0);
      const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
    }
    for (int j = 0; j < nrow; ++j) { // ascending k, as strict order requires
      const uint4 y = *reinterpret_cast<const uint4*>(src + (size_t)rows[j] * hidden + h0);
      const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(yy[u]);
    }
    uint4 o;
    bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
    *reinterpret_cast<uint4*>(dst + h0) = o;
  }
  // scalar tail when hidden is not a multiple of 8
  for (int h = nvec * kPerVec + lane; h < hidden; h += kWarpSize) {
    float acc = 0.f;
    if (bias0) acc += __bfloat162float(bias0[h]);
    if (bias1) acc += __bfloat162float(bias1[h]);
    for (int j = 0; j < nrow; ++j) acc += __bfloat162float(src[(size_t)rows[j] * hidden + h]);
    dst[h] = __float2bfloat16(acc);
  }
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
  if (src_rank >= cfg.num_ranks) return;   // block-uniform, so the barrier below is safe
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
      const int kl = last_local_slot(recv_topk, i, K);
      if (kl >= 0) {
        // Row bases hoisted out of the element loop.
        int32_t row[kMaxTopk];
        int nrow = 0;
        for (int k = 0; k < K; ++k) {
          if (recv_topk[(size_t)i * K + k] < 0) continue;
          const int r = row_map[(size_t)i * K + k];
          if (r >= 0) row[nrow++] = r;
        }
        // No bias on this side; the origin seeds its own. The store goes to a
        // PEER, where narrow stores cost more than against local HBM, which is
        // the other reason reduce_rows_bf16 keeps them at dwordx4.
        reduce_rows_bf16(w.y() + slot * cfg.hidden, in_y, row, nrow,
                         /*bias0=*/nullptr, /*bias1=*/nullptr, cfg.hidden, lane);
      }
      if (in_w) {
        for (int k = 0; k < K; ++k) {
          if (recv_topk[(size_t)i * K + k] < 0) continue;
          const int row = row_map[(size_t)i * K + k];
          if (row >= 0 && lane == 0) w.cw()[slot * K + k] = in_w[row];
        }
      }
    } else if (grouped) {
      // Grouped mode sends ONE row per token, so index by slot rather than by
      // (slot, k). The expanded layout is K times larger, and scattering single
      // rows across it is far slower than a dense write.
      const int kl = last_local_slot(recv_topk, i, K);
      if (kl >= 0) wave_copy_bf16(w.y() + slot * cfg.hidden, in_y + (size_t)i * cfg.hidden, cfg.hidden);
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
        wave_copy_bf16(w.y() + (slot * K + k) * cfg.hidden, in_y + (size_t)row * cfg.hidden, cfg.hidden);
        if (in_w && lane == 0) w.cw()[slot * K + k] = in_w[row];
      }
    }
  }
  // No completion flag: the caller barriers between send and receive, and every
  // slot the receiver reads is one some rank is guaranteed to have written --
  // a token only reaches a rank that owns one of its experts.
}

// Origin side: accumulate across topk slots in strict order, in float32.
// bias0/bias1 are the accumulator's initial value and must seed it BEFORE the
// topk sum so they share the rounding sequence.
// `grouped` mirrors the sender: only the owner's last slot is populated.
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
  const int epr = cfg.experts_per_rank();
  const int K = cfg.num_topk;

  // One wave per token, so gridDim.x * nwarps waves chase num_tokens, and this
  // loop is sensitive in BOTH directions: too few waves and each serialises
  // many tokens, too many and most retire without work while the block still
  // pays for them. Launched at kEpWaves, which is sized for the many-token
  // case; python/rccl_ep_capi.hip records what that costs when tokens are few.
  for (int t = blockIdx.x * nwarps + warp; t < num_tokens; t += stride) {
    // Decide once per token which slots carry a value, rather than re-deriving
    // it for every element of hidden.
    bool take[kMaxTopk];
    for (int k = 0; k < K; ++k) {
      const int e = topk_idx[(size_t)t * K + k];
      // Upper bound as well as lower: owner = e / epr reaches num_ranks for an
      // out-of-range id, and slot() would then index a whole rank-region past
      // the end of the window.
      take[k] = (e >= 0 && e < cfg.num_experts);
      if (take[k] && grouped) {
        const int owner = e / epr;
        for (int k2 = k + 1; k2 < K; ++k2) {
          const int e2 = topk_idx[(size_t)t * K + k2];
          if (e2 >= 0 && e2 / epr == owner) {
            take[k] = false;
            break;
          }
        }
      }
    }

    // Row base for each contributing slot, hoisted out of the element loop.
    // Recomputing owner and slot per element cost a topk_idx load and an
    // integer divide for every one of hidden x K accesses. Packed rather than
    // indexed by k, so the reduction carries no per-element branch; ascending k
    // is preserved, which is the part the bit-exactness contract cares about.
    int32_t row[kMaxTopk];
    int nrow = 0;
    for (int k = 0; k < K; ++k) {
      if (!take[k]) continue;
      const int owner = topk_idx[(size_t)t * K + k] / epr;
      const size_t slot = EpWindowLayout::slot(cfg, owner, t);
      // Must mirror the sender's layout: one row per slot when grouped, one
      // per (slot, k) when reading the expanded layout.
      row[nrow++] = (int32_t)(grouped ? slot : (slot * K + k));
    }

    const size_t tbase = (size_t)t * cfg.hidden;
    reduce_rows_bf16(out + tbase, self.y(), row, nrow,
                     bias0 ? bias0 + tbase : nullptr,
                     bias1 ? bias1 + tbase : nullptr, cfg.hidden, lane);

    if (out_w) {
      for (int k = lane; k < K; k += kWarpSize) {
        const int e = topk_idx[(size_t)t * K + k];
        // Any owner of this token has the full weight row; -1 slots were never
        // sent anywhere, so their weights are written as zero.
        const int owner = (e >= 0) ? (e / epr) : -1;
        out_w[(size_t)t * K + k] = (owner >= 0) ? self.cw()[EpWindowLayout::slot(cfg, owner, t) * K + k] : 0.f;
      }
    }
  }
}

} // namespace rccl_ep

#endif // RCCL_EP_COMBINE_H_
