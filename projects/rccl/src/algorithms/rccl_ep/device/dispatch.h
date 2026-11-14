/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Dispatch: routing scan, the push of token payload into peer windows, and the
 * epilogue that compacts the staged per-source regions into the contiguous
 * output.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_DISPATCH_H_
#define RCCL_EP_DISPATCH_H_

#include "device/ep_common.h"

namespace rccl_ep {

// Phase 1: per (token, dst rank), whether the token is sent and at which slot.
// Slot order must follow token order so the receiver's concatenation stays in
// token order. One CTA per destination; each wave counts the sends in its
// contiguous token range, an LDS scan gives each wave its first slot, and a
// second pass assigns slots in token order. All top-k ids are loaded before any
// is tested, so the loads overlap.
constexpr int kPlanMaxWaves = 32;

__global__ void k_ep_plan(EpConfig cfg,
                          const int32_t* __restrict__ topk_idx,  // [ntok, num_topk]
                          int num_tokens,
                          int32_t* __restrict__ slot_of,         // [num_ranks, ntok] -1 = not sent
                          int32_t* __restrict__ send_list,       // [num_ranks, ntok] slot -> token
                          int32_t* __restrict__ send_counts) {   // [num_ranks]
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;
  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int K = cfg.num_topk;
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  // Whole 64-token rounds per wave, so each round's scan covers full lanes.
  const int per_wave = ((num_tokens + nwarps - 1) / nwarps + kWarpSize - 1) / kWarpSize * kWarpSize;
  const int t_begin = min(num_tokens, warp * per_wave);
  const int t_end = min(num_tokens, t_begin + per_wave);

  auto sends = [&](int t) -> int {
    int send = 0;
    if (t < t_end) {
      int e[kMaxTopk];
#pragma unroll
      for (int k = 0; k < kMaxTopk; ++k) e[k] = k < K ? topk_idx[(size_t)t * K + k] : -1;
#pragma unroll
      for (int k = 0; k < kMaxTopk; ++k) send |= (e[k] >= lo && e[k] < hi) ? 1 : 0;
    }
    return send;
  };

  // Pass 1: this wave's send count.
  int wave_total = 0;
  for (int base = t_begin; base < t_end; base += kWarpSize) wave_total += reduce_add<int>(sends(base + lane));

  // Exclusive scan over the waves, in wave order = token order.
  __shared__ int wave_base[kPlanMaxWaves];
  if (lane == 0) wave_base[warp] = wave_total;
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int w = 0; w < nwarps; ++w) {
      const int v = wave_base[w];
      wave_base[w] = acc;
      acc += v;
    }
    send_counts[dst] = acc;
  }
  __syncthreads();

  // Pass 2: slots, in token order within the wave, from the wave's base.
  int running = wave_base[warp];
  for (int base = t_begin; base < t_end; base += kWarpSize) {
    const int t = base + lane;
    const int send = sends(t);
    const int excl = warp_exclusive_sum<int>(send);
    if (t < t_end) {
      const int slot = send ? (running + excl) : -1;
      slot_of[(size_t)dst * num_tokens + t] = slot;
      // The inverse map, so the send kernel iterates only the tokens it will
      // actually move. With topk 6 over 256 experts on 8 ranks only ~55% of
      // tokens go to any given peer, so scanning all of them wastes ~45% of wave
      // iterations.
      if (send) send_list[(size_t)dst * num_tokens + slot] = t;
    }
    running += reduce_add<int>(send);
  }
}

// Phase 2: push payload and metadata into each destination's window. T is the payload type:
// bf16_t, or fp8_t with its per-128-element scales `sf`. Grid is (num_ranks, ctas_per_rank), one
// wave per token, over the DENSE list of tokens bound for that peer (see k_ep_plan for why the
// list is prebuilt). Slot order follows the list, which follows token order, so the receiver's
// concatenation stays in token order. Only gridDim.y follows the caller's budget, so the block
// must be wide; see kEpWaves.
template <typename T>
__global__ void dispatch_impl(EpConfig cfg,
                              const T* __restrict__ x,               // [ntok, hidden]
                              const float* __restrict__ sf,          // [ntok, hidden_sf], fp8 only
                              const int32_t* __restrict__ topk_idx,  // [ntok, num_topk]
                              const float* __restrict__ topk_w,      // [ntok, num_topk]
                              int num_tokens, const int32_t* __restrict__ send_list,
                              const int32_t* __restrict__ send_counts,
                              WindowView* __restrict__ peer_views) {  // [num_ranks]
  constexpr bool kFp8 = kIsFp8<T>;
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;
  const WindowView w = peer_views[dst];
  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int nsf = cfg.hidden_sf();
  const int n = send_counts[dst];
  const int K = cfg.num_topk, stride = nwarps * gridDim.y, start = blockIdx.y * nwarps + warp;
  const int32_t* list = send_list + (size_t)dst * num_tokens;

  // Row i of the list is token t: payload (and fp8 scales), then the metadata masked to the
  // destination's experts. Lane k holds slot k's expert id and weight; num_topk <= kMaxTopk <
  // kWarpSize. The metadata is loaded before the payload copy, so its load does not sit between
  // two copies.
  auto push_row = [&](int i, int t) {
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, i);
    int e = -1;
    float tw = 0.f;
    if (lane < K) {
      e = topk_idx[(size_t)t * K + lane];
      tw = topk_w[(size_t)t * K + lane];
    }
    wave_copy(w.payload<T>() + slot * cfg.hidden, x + (size_t)t * cfg.hidden, cfg.hidden);
    if constexpr (kFp8) {
      for (int j = lane; j < nsf; j += kWarpSize) w.sf()[slot * nsf + j] = sf[(size_t)t * nsf + j];
    }
    if (lane < K) {
      w.topk_idx()[slot * K + lane] = (e >= lo && e < hi) ? e : -1;
      w.topk_w()[slot * K + lane] = tw;
    }
    if (lane == 0) w.src_idx()[slot] = cfg.rank * cfg.num_max_tokens_per_rank + t;
  };

  // Prefetch the next token's index so its load does not sit between two copies.
  int t = start < n ? list[start] : 0;
  for (int i = start; i < n; i += stride) {
    const int t_next = i + stride < n ? list[i + stride] : 0;
    push_row(i, t);
    t = t_next;
  }
  // The first thread of each CTA publishes the count. The receiver reads it only after barrier B,
  // which also makes the payload above visible.
  __syncthreads();
  if (warp == 0 && lane == 0) {
    st_release_sys<int32_t>(&w.counts()[cfg.rank], send_counts[dst]);
  }
}

// Phase 3: compact the staged regions into the contiguous output. Source-rank
// order, token order preserved within a region -- the required output layout.
// A pure local copy, the stage most sensitive to block width; see kEpWaves.
// Grid (num_ranks, CTAs per source); recv_offsets is the exclusive per-source prefix.
template <typename T>
__global__ void dispatch_copy_epilogue_impl(EpConfig cfg, WindowView self,
                                            const int32_t* __restrict__ recv_offsets,  // [num_ranks] exclusive prefix
                                            T* __restrict__ out_x,
                                            float* __restrict__ out_sf,  // fp8 only
                                            int32_t* __restrict__ out_topk_idx, float* __restrict__ out_topk_w,
                                            int32_t* __restrict__ out_src_idx) {
  constexpr bool kFp8 = kIsFp8<T>;
  const int src = blockIdx.x;
  if (src >= cfg.num_ranks) return;

  const int n = self.counts()[src];
  const int dst_base = recv_offsets[src];
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, my_lo = cfg.expert_begin(cfg.rank), nsf = cfg.hidden_sf();
  static_assert(kMaxTopk <= kWarpSize, "lane k holds top-k slot k");

  for (int i = blockIdx.y * nwarps + warp; i < n; i += nwarps * gridDim.y) {
    const size_t s = EpWindowLayout::slot(cfg, src, i);
    const size_t d = dst_base + i;
    wave_copy_local<true, 4>(out_x + d * cfg.hidden, self.payload<T>() + s * cfg.hidden, cfg.hidden);
    if constexpr (kFp8) {
      for (int j = lane; j < nsf; j += kWarpSize) out_sf[d * nsf + j] = self.sf()[s * nsf + j];
    }
    // The top-k ids rebased into this rank's local experts (the sender sent -1 for any other
    // rank's expert, which stays -1), the weights and the source index.
    if (lane < K) {
      const int e = self.topk_idx()[s * K + lane];
      out_topk_idx[d * K + lane] = (e >= 0) ? (e - my_lo) : -1;
      out_topk_w[d * K + lane] = self.topk_w()[s * K + lane];
    }
    if (lane == 0) out_src_idx[d] = self.src_idx()[s];
  }
}

}  // namespace rccl_ep

#endif  // RCCL_EP_DISPATCH_H_
