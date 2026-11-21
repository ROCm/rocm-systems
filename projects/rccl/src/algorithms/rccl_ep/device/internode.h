/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Internode helpers (ep_create_nodes with node_size < num_ranks). A peer on another
 * node gets a WindowView into a local outbox instead of its LSA window, laid out so
 * the intranode kernels write that peer's rows densely into its own outbox region.
 * The host moves each region with RCCL send/recv; these kernels do the steps
 * around it.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_INTERNODE_H_
#define RCCL_EP_INTERNODE_H_

#include "device/ep_common.h"

namespace rccl_ep {

// Per-call peer counts, passed by value. remote[r] is 1 for a rank on another node.
struct InterCounts {
  int32_t sendc[kMaxWorldRanks];   // rows this rank sent rank r in the dispatch
  int32_t recvc[kMaxWorldRanks];   // rows this rank received from rank r
  int32_t prefix[kMaxWorldRanks];  // exclusive prefix of recvc: first receive index from r
  int32_t remote[kMaxWorldRanks];
};

// The LSA base of every rank on this node, [node_size].
__global__ void k_inter_lsa_bases(ncclWindow_t w, int node_size, uint8_t** out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int p = 0; p < node_size; ++p) out[p] = (uint8_t*)ncclGetLsaPointer(w, 0, p);
  }
}

// dispatch_impl's count release, for remote sources: their rows arrived by recv, whose
// sizes the host already knows.
__global__ void k_inter_set_counts(int num_ranks, InterCounts ic, int32_t* __restrict__ counts) {
  const int r = threadIdx.x;
  if (r < num_ranks && ic.remote[r]) counts[r] = ic.recvc[r];
}

// For a remote source r, replace each row's source index by its position among r's rows,
// so combine_impl writes them densely to outbox region r as rows [0, recvc[r]), in the
// order r sent them (its send_list).
__global__ void k_inter_remap_src(const int32_t* __restrict__ src_idx, int num_recv, int T, InterCounts ic,
                                  int32_t* __restrict__ out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_recv; i += gridDim.x * blockDim.x) {
    const int g = src_idx[i];
    const int r = g / T;
    out[i] = ic.remote[r] ? r * T + (i - ic.prefix[r]) : g;
  }
}

// The owner's side of a remote combine: row j of what rank r sent back (inbox region r)
// belongs to token send_list[r][j], the j-th token this rank dispatched to r. Scatter it,
// with its K weights, to slot(r, token) of this rank's own window, where the reduce epilogue
// reads it.
// y_row_elems is hidden (grouped) or num_topk * hidden (one row per slot and k).
__global__ void k_inter_unpack_combine(EpConfig cfg, InterCounts ic, const int32_t* __restrict__ send_list,
                                       int num_tokens, const bf16_t* __restrict__ in_y, const float* __restrict__ in_cw,
                                       int y_row_elems, WindowView self) {
  const int r = blockIdx.x;
  if (r >= cfg.num_ranks || !ic.remote[r]) return;
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank;
  const int n = ic.sendc[r];
  for (int j = blockIdx.y * nwarps + warp; j < n; j += nwarps * gridDim.y) {
    const int t = send_list[(size_t)r * num_tokens + j];
    const size_t src_row = (size_t)r * T + j;
    const size_t dst_slot = EpWindowLayout::slot(cfg, r, t);
    wave_copy(self.y() + dst_slot * y_row_elems, in_y + src_row * y_row_elems, y_row_elems);
    for (int k = lane; k < K; k += kWarpSize) self.cw()[dst_slot * K + k] = in_cw[src_row * K + k];
  }
}

}  // namespace rccl_ep

#endif  // RCCL_EP_INTERNODE_H_
