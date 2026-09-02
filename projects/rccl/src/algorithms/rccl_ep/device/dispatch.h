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

// Phase 1 deterministic path: per (token, dst rank), whether the token is sent
// and at which slot. One CTA per destination; a single wave preserves token
// order for callers that explicitly request deterministic routing.
__global__ void k_ep_plan(EpConfig cfg,
                          const int32_t* __restrict__ topk_idx,  // [ntok, num_topk]
                          int num_tokens,
                          int32_t* __restrict__ slot_of,         // [num_ranks, ntok] -1 = not sent
                          int32_t* __restrict__ send_list,       // [num_ranks, ntok] slot -> token
                          int32_t* __restrict__ send_counts,     // [num_ranks]
                          int32_t* __restrict__ expert_counts) { // [num_experts]
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;
  if (threadIdx.x >= kWarpSize) return;  // one wave does the ordered scan

  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int lane = get_lane_idx();
  int running = 0;

  for (int base = 0; base < num_tokens; base += kWarpSize) {
    const int t = base + lane;
    int send = 0;
    if (t < num_tokens) {
      for (int k = 0; k < cfg.num_topk; ++k) {
        const int e = topk_idx[(size_t)t * cfg.num_topk + k];
        if (dst == 0 && e >= 0 && e < cfg.num_experts) atomicAdd(expert_counts + e, 1);
        if (e >= lo && e < hi) {
          send = 1;
          break;
        }
      }
    }
    const int excl = warp_exclusive_sum<int>(send);
    if (t < num_tokens) {
      const int slot = send ? (running + excl) : -1;
      slot_of[(size_t)dst * num_tokens + t] = slot;
      // The inverse map, so the send kernel iterates only the tokens it will
      // actually move. With topk 6 over 256 experts only ~55% of tokens go to
      // any given peer, so scanning all of them wastes ~45% of wave iterations.
      if (send) send_list[(size_t)dst * num_tokens + slot] = t;
    }
    running += reduce_add<int>(send);
  }
  if (lane == 0) send_counts[dst] = running;
}

// DeepEP V2's normal path assigns destination slots in parallel with atomic
// sender counters. Keep the ordered scan above for deterministic mode, but do
// not serialize the default path to one wave per destination.
__global__ void k_ep_plan_parallel(EpConfig cfg,
                                   const int32_t* __restrict__ topk_idx,  // [ntok, num_topk]
                                   int num_tokens,
                                   int32_t* __restrict__ slot_of,         // [num_ranks, ntok] -1 = not sent
                                   int32_t* __restrict__ send_list,       // [num_ranks, ntok] slot -> token
                                   int32_t* __restrict__ send_counts,     // [num_ranks]
                                   int32_t* __restrict__ expert_counts) { // [num_experts]
  const int warp = get_warp_idx();
  const int nwarps = get_num_warps();
  const int lane = get_lane_idx();
  const int stride = gridDim.x * nwarps;

  for (int t = blockIdx.x * nwarps + warp; t < num_tokens; t += stride) {
    if (lane < cfg.num_topk) {
      const int expert = topk_idx[(size_t)t * cfg.num_topk + lane];
      if (expert >= 0 && expert < cfg.num_experts) atomicAdd(expert_counts + expert, 1);
    }
    if (lane >= cfg.num_ranks) continue;

    const int dst = lane;
    const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
    bool send = false;
    for (int k = 0; k < cfg.num_topk; ++k) {
      const int e = topk_idx[(size_t)t * cfg.num_topk + k];
      if (e >= lo && e < hi) {
        send = true;
        break;
      }
    }

    int slot = -1;
    if (send) {
      slot = atomicAdd(send_counts + dst, 1);
      send_list[(size_t)dst * num_tokens + slot] = t;
    }
    slot_of[(size_t)dst * num_tokens + t] = slot;
  }
}

// Phase 2: push payload and metadata into each destination's window. Grid is
// (num_ranks, ctas_per_rank), with independent lane groups moving tokens.
// Only gridDim.y follows the caller's budget; subdividing AMD waves preserves
// DeepEP V2's 32 token channels per CTA at a 256-thread block size.
__global__ void dispatch_impl(EpConfig cfg,
                              const bf16_t* __restrict__ x,          // [ntok, hidden]
                              const int32_t* __restrict__ topk_idx,  // [ntok, num_topk]
                              const float* __restrict__ topk_w,      // [ntok, num_topk]
                              int num_tokens, const int32_t* __restrict__ send_list,
                              const int32_t* __restrict__ send_counts,
                              WindowView* __restrict__ peer_views) { // [num_ranks]
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;

  const WindowView w = peer_views[dst];
  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  constexpr int kGroupSize = kBf16TokenGroupSize;
  constexpr int kGroupsPerWave = kWarpSize / kGroupSize;
  const int lane = get_lane_idx();
  const int group_lane = lane % kGroupSize;
  const int group = get_warp_idx() * kGroupsPerWave + lane / kGroupSize;
  const int ngroups = get_num_warps() * kGroupsPerWave;

  // One lane group per token, over the DENSE list bound for this peer;
  // see the note in k_ep_plan for why the list is prebuilt.
  const int n = send_counts[dst];
  const int stride = ngroups * gridDim.y;
  const int start = blockIdx.y * ngroups + group;

  for (int i = start; i < n; i += stride) {
    const int t = send_list[(size_t)dst * num_tokens + i];
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, i);

    // payload
    group_copy_bf16_peer<kGroupSize>(
        w.x() + slot * cfg.hidden, x + (size_t)t * cfg.hidden, cfg.hidden,
        group_lane);

    // metadata: mask topk_idx to the destination's expert range before sending
    for (int k = group_lane; k < cfg.num_topk; k += kGroupSize) {
      const int e = topk_idx[(size_t)t * cfg.num_topk + k];
      w.topk_idx()[slot * cfg.num_topk + k] = (e >= lo && e < hi) ? e : -1;
      w.topk_w()[slot * cfg.num_topk + k] = topk_w[(size_t)t * cfg.num_topk + k];
    }
    if (group_lane == 0) {
      w.src_idx()[slot] = cfg.rank * cfg.num_max_tokens_per_rank + t;
    }
  }

  // one wave per destination publishes the count, release-ordered so the
  // payload above is visible before the count that advertises it
  __syncthreads();
  if (threadIdx.x == 0) {
    st_release_sys<int32_t>(&w.counts()[cfg.rank], send_counts[dst]);
    st_release_sys<uint32_t>(&w.flags()[cfg.rank], 1u);
  }
}

// Production BF16 specialization. At hidden=2048/topk=8 one eight-lane group
// owns a token, so both the payload loop bounds and metadata assignment are
// compile-time constants while retaining the generic kernel's launch shape.
__global__ void dispatch_bf16_8x2048_impl(
    EpConfig cfg,
    const bf16_t* __restrict__ x,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    int num_tokens,
    const int32_t* __restrict__ send_list,
    const int32_t* __restrict__ send_counts,
    WindowView* __restrict__ peer_views) {
  constexpr int kNumTopk = 8;
  constexpr int kHidden = 2048;
  constexpr int kGroupSize = kBf16TokenGroupSize;
  constexpr int kGroupsPerWave = kWarpSize / kGroupSize;
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;

  const WindowView peer = peer_views[dst];
  const int expert_begin = cfg.expert_begin(dst);
  const int expert_end = cfg.expert_end(dst);
  const int lane = get_lane_idx();
  const int group_lane = lane % kGroupSize;
  const int group = get_warp_idx() * kGroupsPerWave + lane / kGroupSize;
  const int ngroups = get_num_warps() * kGroupsPerWave;
  const int count = send_counts[dst];
  const int stride = ngroups * gridDim.y;

  for (int i = blockIdx.y * ngroups + group; i < count; i += stride) {
    const int token = send_list[static_cast<size_t>(dst) * num_tokens + i];
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, i);
    group_copy_bf16_peer<kGroupSize>(
        peer.x() + slot * kHidden,
        x + static_cast<size_t>(token) * kHidden,
        kHidden,
        group_lane);

    const int expert =
        topk_idx[static_cast<size_t>(token) * kNumTopk + group_lane];
    peer.topk_idx()[slot * kNumTopk + group_lane] =
        expert >= expert_begin && expert < expert_end ? expert : -1;
    peer.topk_w()[slot * kNumTopk + group_lane] =
        topk_w[static_cast<size_t>(token) * kNumTopk + group_lane];
    if (group_lane == 0) {
      peer.src_idx()[slot] = cfg.rank * cfg.num_max_tokens_per_rank + token;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    st_release_sys<int32_t>(&peer.counts()[cfg.rank], send_counts[dst]);
    st_release_sys<uint32_t>(&peer.flags()[cfg.rank], 1u);
  }
}

__global__ void dispatch_impl(EpConfig cfg, const fp8_t* __restrict__ x,
                              const float* __restrict__ sf,          // [ntok, hidden_sf]
                              const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_w, int num_tokens,
                              const int32_t* __restrict__ send_list, const int32_t* __restrict__ send_counts,
                              WindowView* __restrict__ peer_views) {
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;
  const WindowView w = peer_views[dst];
  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int nsf = cfg.hidden_sf();
  const int n = send_counts[dst];

  for (int i = blockIdx.y * nwarps + warp; i < n; i += nwarps * gridDim.y) {
    const int t = send_list[(size_t)dst * num_tokens + i];
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, i);

    wave_copy_bytes_peer((uint8_t*)(w.x_fp8() + slot * cfg.hidden),
                         (const uint8_t*)(x + (size_t)t * cfg.hidden), cfg.hidden);
    for (int i = lane; i < nsf; i += kWarpSize) w.sf()[slot * nsf + i] = sf[(size_t)t * nsf + i];
    for (int k = lane; k < cfg.num_topk; k += kWarpSize) {
      const int e = topk_idx[(size_t)t * cfg.num_topk + k];
      w.topk_idx()[slot * cfg.num_topk + k] = (e >= lo && e < hi) ? e : -1;
      w.topk_w()[slot * cfg.num_topk + k] = topk_w[(size_t)t * cfg.num_topk + k];
    }
    if (lane == 0) w.src_idx()[slot] = cfg.rank * cfg.num_max_tokens_per_rank + t;
  }
  __syncthreads();
  if (warp == 0 && lane == 0) {
    st_release_sys<int32_t>(&w.counts()[cfg.rank], send_counts[dst]);
    st_release_sys<uint32_t>(&w.flags()[cfg.rank], 1u);
  }
}

// Phase 3: compact the staged regions into the contiguous output. Source-rank
// order, token order preserved within a region -- the required output layout.
// A pure local copy, the stage most sensitive to block width; see kEpWaves.
__global__ void dispatch_copy_epilogue_impl(EpConfig cfg, WindowView self,
                                            const int32_t* __restrict__ recv_offsets,  // [num_ranks] exclusive prefix
                                            bf16_t* __restrict__ out_x, int32_t* __restrict__ out_topk_idx,
                                            float* __restrict__ out_topk_w, int32_t* __restrict__ out_src_idx,
                                            bool copy_payload) {
  const int src = blockIdx.x;
  if (src >= cfg.num_ranks) return;

  const int n = self.counts()[src];
  const int dst_base = recv_offsets[src];
  constexpr int kGroupSize = kBf16TokenGroupSize;
  constexpr int kGroupsPerWave = kWarpSize / kGroupSize;
  const int lane = get_lane_idx();
  const int group_lane = lane % kGroupSize;
  const int group = get_warp_idx() * kGroupsPerWave + lane / kGroupSize;
  const int ngroups = get_num_warps() * kGroupsPerWave;
  const int my_lo = cfg.expert_begin(cfg.rank);

  for (int i = blockIdx.y * ngroups + group; i < n; i += ngroups * gridDim.y) {
    const size_t s = EpWindowLayout::slot(cfg, src, i);
    const size_t d = dst_base + i;

    if (copy_payload) {
      group_copy_bf16_from_peer<kGroupSize>(
          out_x + d * cfg.hidden, self.x() + s * cfg.hidden, cfg.hidden,
          group_lane);
    }

    for (int k = group_lane; k < cfg.num_topk; k += kGroupSize) {
      const int e = self.topk_idx()[s * cfg.num_topk + k];
      // rebase into local expert space; anything outside becomes -1
      out_topk_idx[d * cfg.num_topk + k] = (e >= 0) ? (e - my_lo) : -1;
      out_topk_w[d * cfg.num_topk + k] = self.topk_w()[s * cfg.num_topk + k];
    }
    if (group_lane == 0) out_src_idx[d] = self.src_idx()[s];
  }
}

// Cached BF16 replay moves the large payload with the copy engine. Use every
// thread here for the remaining metadata instead of assigning a whole wave to
// one token, which would leave 56 of 64 lanes idle at top-k 8.
__global__ void dispatch_metadata_epilogue_impl(
    EpConfig cfg,
    WindowView self,
    const int32_t* __restrict__ recv_offsets,
    int32_t* __restrict__ out_topk_idx,
    float* __restrict__ out_topk_w,
    int32_t* __restrict__ out_src_idx) {
  const int src = blockIdx.x;
  if (src >= cfg.num_ranks) return;

  const int n = self.counts()[src];
  const int dst_base = recv_offsets[src];
  const int thread = blockIdx.y * blockDim.x + threadIdx.x;
  const int stride = gridDim.y * blockDim.x;
  const int my_lo = cfg.expert_begin(cfg.rank);

  for (int i = thread; i < n; i += stride) {
    const size_t s = EpWindowLayout::slot(cfg, src, i);
    const size_t d = dst_base + i;
    for (int k = 0; k < cfg.num_topk; ++k) {
      const int e = self.topk_idx()[s * cfg.num_topk + k];
      out_topk_idx[d * cfg.num_topk + k] = (e >= 0) ? (e - my_lo) : -1;
      out_topk_w[d * cfg.num_topk + k] = self.topk_w()[s * cfg.num_topk + k];
    }
    out_src_idx[d] = self.src_idx()[s];
  }
}

__global__ void dispatch_copy_epilogue_impl(EpConfig cfg, WindowView self, const int32_t* __restrict__ recv_offsets,
                                            fp8_t* __restrict__ out_x, float* __restrict__ out_sf,
                                            int32_t* __restrict__ out_topk_idx, float* __restrict__ out_topk_w,
                                            int32_t* __restrict__ out_src_idx, bool copy_payload) {
  const int src = blockIdx.x;
  if (src >= cfg.num_ranks) return;
  const int n = self.counts()[src];
  const int dst_base = recv_offsets[src];
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int my_lo = cfg.expert_begin(cfg.rank), nsf = cfg.hidden_sf();

  for (int i = blockIdx.y * nwarps + warp; i < n; i += nwarps * gridDim.y) {
    const size_t s = EpWindowLayout::slot(cfg, src, i);
    const size_t d = dst_base + i;
    if (copy_payload) {
      wave_copy_bytes_from_peer((uint8_t*)(out_x + d * cfg.hidden),
                                (const uint8_t*)(self.x_fp8() + s * cfg.hidden), cfg.hidden);
    }
    for (int j = lane; j < nsf; j += kWarpSize) out_sf[d * nsf + j] = self.sf()[s * nsf + j];
    for (int k = lane; k < cfg.num_topk; k += kWarpSize) {
      const int e = self.topk_idx()[s * cfg.num_topk + k];
      out_topk_idx[d * cfg.num_topk + k] = (e >= 0) ? (e - my_lo) : -1;
      out_topk_w[d * cfg.num_topk + k] = self.topk_w()[s * cfg.num_topk + k];
    }
    if (lane == 0) out_src_idx[d] = self.src_idx()[s];
  }
}

}  // namespace rccl_ep

#endif  // RCCL_EP_DISPATCH_H_
