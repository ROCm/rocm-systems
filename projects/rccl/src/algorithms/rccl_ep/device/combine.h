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
  // loaded once per CTA and every write goes to a single peer. The BF16 path
  // uses the same 32 token groups per CTA as dispatch.
  const int src_rank = blockIdx.x;
  if (src_rank >= cfg.num_ranks) return;   // block-uniform, so the barrier below is safe
  constexpr int kGroupSize = kBf16TokenGroupSize;
  constexpr int kGroupsPerWave = kWarpSize / kGroupSize;
  const int lane = get_lane_idx();
  const int group_lane = lane % kGroupSize;
  const int group = get_warp_idx() * kGroupsPerWave + lane / kGroupSize;
  const int ngroups = get_num_warps() * kGroupsPerWave;
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
  const int stride = ngroups * gridDim.y;

  for (int i = begin + blockIdx.y * ngroups + group; i < end && i < num_recv; i += stride) {
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
        bf16_t* dst = w.y() + slot * cfg.hidden;
        // Row bases hoisted and vectorised at dwordx4. This writes to a PEER,
        // where narrow stores cost more than against local HBM, so keep it wide.
        int32_t row[kMaxTopk];
        int nrow = 0;
        for (int k = 0; k < K; ++k) {
          if (recv_topk[(size_t)i * K + k] < 0) continue;
          const int r = row_map[(size_t)i * K + k];
          if (r >= 0) row[nrow++] = r;
        }
        constexpr int kPerVec = 8;
        const int nvec = cfg.hidden / kPerVec;
        for (int v = group_lane; v < nvec; v += kGroupSize) {
          float acc[kPerVec];
          const int h0 = v * kPerVec;
#pragma unroll
          for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
          for (int j = 0; j < nrow; ++j) {   // ascending k, as strict order requires
            const uint4 y = *reinterpret_cast<const uint4*>(in_y + (size_t)row[j] * cfg.hidden + h0);
            const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
#pragma unroll
            for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(yy[u]);
          }
          uint4 o;
          bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
          for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
          store16_peer(reinterpret_cast<uint4*>(dst + h0), o);
        }
        for (int h = nvec * kPerVec + group_lane; h < cfg.hidden; h += kGroupSize) {
          float acc = 0.f;
          for (int j = 0; j < nrow; ++j) acc += __bfloat162float(in_y[(size_t)row[j] * cfg.hidden + h]);
          dst[h] = __float2bfloat16(acc);
        }
      }
      if (in_w) {
        for (int k = 0; k < K; ++k) {
          if (recv_topk[(size_t)i * K + k] < 0) continue;
          const int row = row_map[(size_t)i * K + k];
          if (row >= 0 && group_lane == 0) w.cw()[slot * K + k] = in_w[row];
        }
      }
    } else if (grouped) {
      // Grouped mode sends ONE row per token, so index by slot rather than by
      // (slot, k). The expanded layout is K times larger, and scattering single
      // rows across it is far slower than a dense write.
      const int kl = last_local_slot(recv_topk, i, K);
      if (kl >= 0) {
        group_copy_bf16_peer<kGroupSize>(
            w.y() + slot * cfg.hidden, in_y + (size_t)i * cfg.hidden,
            cfg.hidden, group_lane);
      }
      // The origin needs its own top-k weights back; every rank that received
      // this token holds an identical copy of the full row, so whichever
      // arrives last wins and the result is the same either way.
      if (in_w) {
        for (int k = group_lane; k < K; k += kGroupSize) {
          w.cw()[slot * K + k] = in_w[(size_t)i * K + k];
        }
      }
    } else {
      for (int k = 0; k < K; ++k) {
        if (recv_topk[(size_t)i * K + k] < 0) continue;  // not ours
        const int row = row_map ? row_map[(size_t)i * K + k] : (int)(i * K + k);
        if (row < 0) continue;
        group_copy_bf16_peer<kGroupSize>(
            w.y() + (slot * K + k) * cfg.hidden,
            in_y + (size_t)row * cfg.hidden, cfg.hidden, group_lane);
        if (in_w && group_lane == 0) w.cw()[slot * K + k] = in_w[row];
      }
    }
  }
  // No completion flag: the caller barriers between send and receive, and every
  // slot the receiver reads is one some rank is guaranteed to have written --
  // a token only reaches a rank that owns one of its experts.
}

// Production BF16 path. The target workload's forward combine is grouped and
// unexpanded at hidden=2048/topk=8, so every received token contributes one dense
// row. Specializing those invariants removes the generic row-map and top-k
// scans while preserving the same 32 token groups per 256-thread CTA.
__global__ void combine_grouped_bf16_8x2048_impl(
    EpConfig cfg,
    const bf16_t* __restrict__ in_y,
    const float* __restrict__ in_w,
    const int32_t* __restrict__ src_idx,
    int num_recv,
    WindowView self,
    WindowView* __restrict__ peer_views) {
  constexpr int kNumTopk = 8;
  constexpr int kHidden = 2048;
  constexpr int kGroupSize = kBf16TokenGroupSize;
  constexpr int kGroupsPerWave = kWarpSize / kGroupSize;
  const int src_rank = blockIdx.x;
  if (src_rank >= cfg.num_ranks) return;

  const int lane = get_lane_idx();
  const int group_lane = lane % kGroupSize;
  const int group = get_warp_idx() * kGroupsPerWave + lane / kGroupSize;
  const int ngroups = get_num_warps() * kGroupsPerWave;

  __shared__ int s_begin, s_end;
  if (threadIdx.x == 0) {
    int begin = 0;
    for (int rank = 0; rank < src_rank; ++rank) {
      begin += self.counts()[rank];
    }
    s_begin = begin;
    s_end = begin + self.counts()[src_rank];
  }
  __syncthreads();

  const WindowView peer = peer_views[src_rank];
  const int stride = ngroups * gridDim.y;
  for (int i = s_begin + blockIdx.y * ngroups + group;
       i < s_end && i < num_recv;
       i += stride) {
    const int global_src_token = src_idx[i];
    const int src_token = global_src_token % cfg.num_max_tokens_per_rank;
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, src_token);
    group_copy_bf16_peer<kGroupSize>(
        peer.y() + slot * kHidden,
        in_y + static_cast<size_t>(i) * kHidden,
        kHidden,
        group_lane);
    if (in_w != nullptr) {
      peer.cw()[slot * kNumTopk + group_lane] =
          in_w[static_cast<size_t>(i) * kNumTopk + group_lane];
    }
  }
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
  constexpr int kGroupSize = kBf16TokenGroupSize;
  constexpr int kGroupsPerWave = kWarpSize / kGroupSize;
  const int lane = get_lane_idx();
  const int group_lane = lane % kGroupSize;
  const int group = get_warp_idx() * kGroupsPerWave + lane / kGroupSize;
  const int ngroups = get_num_warps() * kGroupsPerWave;
  const int stride = ngroups * gridDim.x;
  const int epr = cfg.experts_per_rank();
  const int K = cfg.num_topk;

  // One lane group per token keeps DeepEP V2's 32-way token concurrency while
  // retaining RCCL's 256-thread CTA footprint.
  for (int t = blockIdx.x * ngroups + group; t < num_tokens; t += stride) {
    // Decide once per token which slots carry a value, rather than re-deriving
    // it for every element of hidden.
    bool take[kMaxTopk];
    for (int k = 0; k < K; ++k) {
      const int e = topk_idx[(size_t)t * K + k];
      take[k] = (e >= 0);
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
    // integer divide for every one of hidden x K accesses.
    int32_t row[kMaxTopk];
    for (int k = 0; k < K; ++k) {
      if (!take[k]) {
        row[k] = -1;
        continue;
      }
      const int owner = topk_idx[(size_t)t * K + k] / epr;
      const size_t slot = EpWindowLayout::slot(cfg, owner, t);
      // Must mirror the sender's layout: one row per slot when grouped, one
      // per (slot, k) when reading the expanded layout.
      row[k] = (int32_t)(grouped ? slot : (slot * K + k));
    }

    // Vectorised at dwordx4, matching the dispatch path; scalar 2-byte loads
    // here are an 8x loss of load width.
    //
    // Bit-exactness is preserved: this changes only which lane owns which
    // element, never the order of additions applied to any single element,
    // which stays bias then ascending k.
    constexpr int kPerVec = 8; // 8 bf16 = 16 B
    constexpr int kVecUnroll = 4;
    constexpr int kVecStride = kGroupSize * kVecUnroll;
    const int nvec = cfg.hidden / kPerVec;
    const int bulk_vecs = (nvec / kVecStride) * kVecStride;
    const bf16_t* ybase = self.y();
    // Reduction input is reused across the top-k fold. Keep these loads as
    // cacheable 16-byte operations; load16_peer lowers to four scalar
    // non-temporal loads on gfx950 and increases scratch pressure.
    for (int base = group_lane; base < bulk_vecs; base += kVecStride) {
      float acc[kVecUnroll][kPerVec];
#pragma unroll
      for (int j = 0; j < kVecUnroll; ++j) {
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) acc[j][u] = 0.f;
      }
      if (bias0) {
#pragma unroll
        for (int j = 0; j < kVecUnroll; ++j) {
          const int h0 = (base + j * kGroupSize) * kPerVec;
          const uint4 b = *reinterpret_cast<const uint4*>(
              bias0 + (size_t)t * cfg.hidden + h0);
          const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
          for (int u = 0; u < kPerVec; ++u) {
            acc[j][u] += __bfloat162float(bb[u]);
          }
        }
      }
      if (bias1) {
#pragma unroll
        for (int j = 0; j < kVecUnroll; ++j) {
          const int h0 = (base + j * kGroupSize) * kPerVec;
          const uint4 b = *reinterpret_cast<const uint4*>(
              bias1 + (size_t)t * cfg.hidden + h0);
          const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
          for (int u = 0; u < kPerVec; ++u) {
            acc[j][u] += __bfloat162float(bb[u]);
          }
        }
      }
      for (int k = 0; k < K; ++k) { // strict k order
        if (row[k] < 0) continue;
        uint4 values[kVecUnroll];
#pragma unroll
        for (int j = 0; j < kVecUnroll; ++j) {
          const int h0 = (base + j * kGroupSize) * kPerVec;
          values[j] = *reinterpret_cast<const uint4*>(
              ybase + (size_t)row[k] * cfg.hidden + h0);
        }
#pragma unroll
        for (int j = 0; j < kVecUnroll; ++j) {
          const bf16_t* yy = reinterpret_cast<const bf16_t*>(&values[j]);
#pragma unroll
          for (int u = 0; u < kPerVec; ++u) {
            acc[j][u] += __bfloat162float(yy[u]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kVecUnroll; ++j) {
        uint4 value;
        bf16_t* output = reinterpret_cast<bf16_t*>(&value);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) {
          output[u] = __float2bfloat16(acc[j][u]);
        }
        const int h0 = (base + j * kGroupSize) * kPerVec;
        *reinterpret_cast<uint4*>(out + (size_t)t * cfg.hidden + h0) = value;
      }
    }

    for (int v = bulk_vecs + group_lane; v < nvec; v += kGroupSize) {
      float acc[kPerVec];
      const int h0 = v * kPerVec;
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
      if (bias0) {
        const uint4 b = *reinterpret_cast<const uint4*>(bias0 + (size_t)t * cfg.hidden + h0);
        const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
      }
      if (bias1) {
        const uint4 b = *reinterpret_cast<const uint4*>(bias1 + (size_t)t * cfg.hidden + h0);
        const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
      }
      for (int k = 0; k < K; ++k) { // strict k order
        if (row[k] < 0) continue;
        const uint4 y = *reinterpret_cast<const uint4*>(
            ybase + (size_t)row[k] * cfg.hidden + h0);
        const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(yy[u]);
      }
      uint4 o;
      bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
      *reinterpret_cast<uint4*>(out + (size_t)t * cfg.hidden + h0) = o;
    }

    // scalar tail when hidden is not a multiple of 8
    for (int h = nvec * kPerVec + group_lane; h < cfg.hidden; h += kGroupSize) {
      float acc = 0.f;
      if (bias0) acc += __bfloat162float(bias0[(size_t)t * cfg.hidden + h]);
      if (bias1) acc += __bfloat162float(bias1[(size_t)t * cfg.hidden + h]);
      for (int k = 0; k < K; ++k) {
        if (row[k] < 0) continue;
        acc += __bfloat162float(ybase[(size_t)row[k] * cfg.hidden + h]);
      }
      out[(size_t)t * cfg.hidden + h] = __float2bfloat16(acc);
    }

    if (out_w) {
      for (int k = group_lane; k < K; k += kGroupSize) {
        const int e = topk_idx[(size_t)t * K + k];
        // Any owner of this token has the full weight row; -1 slots were never
        // sent anywhere, so their weights are written as zero.
        const int owner = (e >= 0) ? (e / epr) : -1;
        out_w[(size_t)t * K + k] = (owner >= 0) ? self.cw()[EpWindowLayout::slot(cfg, owner, t) * K + k] : 0.f;
      }
    }
  }
}

// Production specialization of the origin-side grouped reduction. Each lane
// owns one top-k entry, then shares its resolved row with the other seven lanes
// in the token group. This follows DeepEP V2's lane-distributed metadata
// pattern and avoids dynamically indexed per-thread row/take arrays.
__global__ void combine_reduce_grouped_bf16_8x2048_impl(
    EpConfig cfg,
    WindowView self,
    const int32_t* __restrict__ topk_idx,
    int num_tokens,
    bf16_t* __restrict__ out,
    float* __restrict__ out_w) {
  constexpr int kNumTopk = 8;
  constexpr int kHidden = 2048;
  constexpr int kGroupSize = kBf16TokenGroupSize;
  constexpr int kGroupsPerWave = kWarpSize / kGroupSize;
  constexpr int kPerVec = 8;
  constexpr int kVecUnroll = 4;
  constexpr int kVecStride = kGroupSize * kVecUnroll;
  constexpr int kNumVec = kHidden / kPerVec;

  const int lane = get_lane_idx();
  const int group_lane = lane % kGroupSize;
  const int group = get_warp_idx() * kGroupsPerWave + lane / kGroupSize;
  const int ngroups = get_num_warps() * kGroupsPerWave;
  const int stride = ngroups * gridDim.x;
  const int experts_per_rank = cfg.experts_per_rank();
  const bf16_t* ybase = self.y();

  for (int token = blockIdx.x * ngroups + group;
       token < num_tokens;
       token += stride) {
    const int expert = topk_idx[static_cast<size_t>(token) * kNumTopk + group_lane];
    const int owner = expert >= 0 ? expert / experts_per_rank : -1;

    bool take = owner >= 0;
#pragma unroll
    for (int other_lane = 0; other_lane < kNumTopk; ++other_lane) {
      const int other_owner = __shfl(owner, other_lane, kGroupSize);
      if (other_lane > group_lane && other_owner == owner) {
        take = false;
      }
    }
    const int lane_row = take
        ? static_cast<int>(EpWindowLayout::slot(cfg, owner, token))
        : -1;
    int rows[kNumTopk];
#pragma unroll
    for (int k = 0; k < kNumTopk; ++k) {
      rows[k] = __shfl(lane_row, k, kGroupSize);
    }

    for (int base = group_lane; base < kNumVec; base += kVecStride) {
      float acc[kVecUnroll][kPerVec] = {};
#pragma unroll
      for (int k = 0; k < kNumTopk; ++k) {
        if (rows[k] < 0) continue;
        uint4 values[kVecUnroll];
#pragma unroll
        for (int j = 0; j < kVecUnroll; ++j) {
          const int h0 = (base + j * kGroupSize) * kPerVec;
          values[j] = *reinterpret_cast<const uint4*>(
              ybase + static_cast<size_t>(rows[k]) * kHidden + h0);
        }
#pragma unroll
        for (int j = 0; j < kVecUnroll; ++j) {
          const bf16_t* values_bf16 =
              reinterpret_cast<const bf16_t*>(&values[j]);
#pragma unroll
          for (int u = 0; u < kPerVec; ++u) {
            acc[j][u] += __bfloat162float(values_bf16[u]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kVecUnroll; ++j) {
        uint4 value;
        bf16_t* output = reinterpret_cast<bf16_t*>(&value);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) {
          output[u] = __float2bfloat16(acc[j][u]);
        }
        const int h0 = (base + j * kGroupSize) * kPerVec;
        *reinterpret_cast<uint4*>(
            out + static_cast<size_t>(token) * kHidden + h0) = value;
      }
    }

    if (out_w != nullptr) {
      out_w[static_cast<size_t>(token) * kNumTopk + group_lane] =
          owner >= 0
          ? self.cw()[EpWindowLayout::slot(cfg, owner, token) * kNumTopk +
                      group_lane]
          : 0.f;
    }
  }
}

} // namespace rccl_ep

#endif // RCCL_EP_COMBINE_H_
