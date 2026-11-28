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

// Every source's count for the scan and the epilogue, run after barrier B: the bound receive count, so they copy
// exactly the rows the caller sized its outputs for. A remote source's rows arrived by a receive of that size. A
// same-node source published its own count, and `sendc` is this call's plan: either differs from the bound counts only
// for a plan dispatched without its own exchange and bind. The first such rank goes to *bad: r + 1 for this rank's
// send count to r, kMaxWorldRanks + r + 1 for the count same-node source r published, 0 for none.
__global__ void k_inter_set_counts(int num_ranks, InterCounts ic, const int32_t* __restrict__ sendc,
                                   int32_t* __restrict__ counts, int32_t* __restrict__ bad) {
  constexpr int kNone = 2 * kMaxWorldRanks;
  __shared__ int s_bad;
  const int r = threadIdx.x;
  if (r == 0) s_bad = kNone;
  __syncthreads();
  if (r < num_ranks) {
    int code = kNone;
    if (sendc[r] != ic.sendc[r]) {
      code = r;
    } else if (!ic.remote[r] && ld_acquire_sys<int32_t>(&counts[r]) != ic.recvc[r]) {
      code = kMaxWorldRanks + r;
    }
    if (code != kNone) atomicMin(&s_bad, code);
    counts[r] = ic.recvc[r];
  }
  __syncthreads();
  if (r == 0) *bad = s_bad == kNone ? 0 : s_bad + 1;
}

// For a remote source r, replace each row's source index by its position among r's rows,
// so combine_impl writes them densely to outbox region r as rows [0, recvc[r]), in the
// order r sent them (its send_list). Which source a row came from is read from the bound
// counts, rows [prefix[r], prefix[r] + recvc[r]) for r, not from src_idx, so a remote row
// reaches its own outbox row whatever src_idx holds. Rows from this node keep src_idx.
// Block 0 also writes the bound counts into this rank's window (`counts`), where combine_impl
// reads each source's rows from: they are then the combined dispatch's, not the latest one's.
__global__ void k_inter_remap_src(const int32_t* __restrict__ src_idx, int num_recv, int num_ranks, int T,
                                  InterCounts ic, int32_t* __restrict__ counts, int32_t* __restrict__ out) {
  if (blockIdx.x == 0) {
    for (int r = threadIdx.x; r < num_ranks; r += blockDim.x) counts[r] = ic.recvc[r];
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_recv; i += gridDim.x * blockDim.x) {
    // The last rank whose rows start at or before row i; any earlier rank with the same start has none.
    int r = 0;
    for (int step = kMaxWorldRanks / 2; step > 0; step /= 2) {
      if (r + step < num_ranks && ic.prefix[r + step] <= i) r += step;
    }
    out[i] = ic.remote[r] && i < ic.prefix[r] + ic.recvc[r] ? r * T + (i - ic.prefix[r]) : src_idx[i];
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
