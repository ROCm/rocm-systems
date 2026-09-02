/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * do_expand: grouped-by-expert output layout, so an expert GEMM consumes the
 * dispatched tensor without a permute. Row assignment must be deterministic,
 * not merely valid, because a cached replay indexes through the original run's
 * map; it comes from a segmented scan rather than per-expert atomics.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_EXPAND_H_
#define RCCL_EP_EXPAND_H_

#include <cstdint>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "device/ep_common.h"
#include "device/hip_prims.h"
#include "include/ep_layout.h"

namespace rccl_ep {

__host__ __device__ inline int align_count(int n, int a) {
  return a <= 1 ? n : ((n + a - 1) / a) * a;
}

// Count, per local expert, how many (token, slot) pairs select it.
__global__ void k_expand_count(EpConfig cfg,
                               const int32_t* __restrict__ recv_topk,  // [nrecv, num_topk] local ids
                               int num_recv,
                               int32_t* __restrict__ counts) {         // [experts_per_rank]
  const int e = blockIdx.x;
  const int epr = cfg.experts_per_rank();
  if (e >= epr) return;

  int n = 0;
  for (int i = threadIdx.x; i < num_recv; i += blockDim.x)
    for (int k = 0; k < cfg.num_topk; ++k)
      if (recv_topk[(size_t)i * cfg.num_topk + k] == e) ++n;

  // reduce across the whole CTA
  __shared__ int sh[64];
  const int w = get_warp_idx(), lane = get_lane_idx();
  n = reduce_add<int>(n);
  if (lane == 0) sh[w] = n;
  __syncthreads();
  if (threadIdx.x == 0) {
    int tot = 0;
    for (int i = 0; i < get_num_warps(); ++i) tot += sh[i];
    counts[e] = tot;
  }
}

// Exclusive prefix over the ALIGNED counts, giving each expert's base row, plus
// the inclusive prefix of REAL rows the caller reports as
// psum_num_recv_tokens_per_expert.
//
// Because every base is a multiple of the alignment, `offsets[e] + counts[e]` is
// exactly the end of expert e's real rows, and align(psum[e-1]) == offsets[e].
__global__ void k_expand_scan(EpConfig cfg, int expert_alignment, const int32_t* __restrict__ counts,
                              int32_t* __restrict__ offsets,   // [experts_per_rank]
                              int32_t* __restrict__ psum,      // [experts_per_rank]
                              int32_t* __restrict__ total_rows) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const int epr = cfg.experts_per_rank();
  int acc = 0;
  for (int e = 0; e < epr; ++e) {
    offsets[e] = acc;
    psum[e] = acc + counts[e];
    acc += align_count(counts[e], expert_alignment);
  }
  *total_rows = acc;
}

// Per received token, how many of its slots select each local expert.
// Sparse in principle (at most num_topk non-zeros per row) but small enough
// dense that the scan below stays a plain segmented prefix sum.
__global__ void k_expand_hist(EpConfig cfg, const int32_t* __restrict__ recv_topk, int num_recv,
                              int32_t* __restrict__ hist) {   // [nrecv, epr]
  const int epr = cfg.experts_per_rank();
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)num_recv; i += (size_t)gridDim.x * blockDim.x) {
    for (int e = 0; e < epr; ++e) hist[i * epr + e] = 0;
    for (int k = 0; k < cfg.num_topk; ++k) {
      const int e = recv_topk[i * cfg.num_topk + k];
      if (e >= 0) ++hist[i * epr + e];
    }
  }
}

// Exclusive scan of `hist` along the token axis, one expert per CTA.
__global__ void k_expand_base(EpConfig cfg, int num_recv,
                              int32_t* __restrict__ hist) {   // in place
  const int e = blockIdx.x;
  const int epr = cfg.experts_per_rank();
  if (e >= epr) return;
  if (threadIdx.x >= kWarpSize) return;

  const int lane = get_lane_idx();
  int running = 0;
  for (int base = 0; base < num_recv; base += kWarpSize) {
    const int i = base + lane;
    const int v = (i < num_recv) ? hist[(size_t)i * epr + e] : 0;
    const int excl = warp_exclusive_sum<int>(v);
    if (i < num_recv) hist[(size_t)i * epr + e] = running + excl;
    running += reduce_add<int>(v);
  }
}

// Final row index for every (received token, slot) pair.
__global__ void k_expand_map(EpConfig cfg, const int32_t* __restrict__ recv_topk, int num_recv,
                             const int32_t* __restrict__ offsets,
                             const int32_t* __restrict__ base,   // scanned hist
                             int32_t* __restrict__ out_map) {    // [nrecv, num_topk]
  const int epr = cfg.experts_per_rank(), K = cfg.num_topk;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)num_recv; i += (size_t)gridDim.x * blockDim.x) {
    for (int k = 0; k < K; ++k) {
      const int e = recv_topk[i * K + k];
      if (e < 0) {
        out_map[i * K + k] = -1;
        continue;
      }
      // rank of this slot among the same token's earlier slots on the same expert
      int within = 0;
      for (int k2 = 0; k2 < k; ++k2)
        if (recv_topk[i * K + k2] == e) ++within;
      out_map[i * K + k] = offsets[e] + base[i * epr + e] + within;
    }
  }
}

// Scatter payload and per-row weight into the mapped rows. bf16 path.
__global__ void k_expand_scatter(EpConfig cfg,
                                 const bf16_t* __restrict__ recv_x,     // [nrecv, hidden]
                                 const float* __restrict__ recv_w,      // [nrecv, num_topk]
                                 const int32_t* __restrict__ row_map,   // [nrecv, num_topk]
                                 int num_recv,
                                 bf16_t* __restrict__ out_x,            // [total_rows, hidden]
                                 float* __restrict__ out_w) {           // [total_rows]
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk;
  const size_t n = (size_t)num_recv * K;

  for (size_t p = (size_t)blockIdx.x * nwarps + warp; p < n; p += (size_t)gridDim.x * nwarps) {
    const int row = row_map[p];
    if (row < 0) continue;
    const size_t i = p / K;
    wave_copy_bf16(out_x + (size_t)row * cfg.hidden, recv_x + i * cfg.hidden, cfg.hidden);
    if (out_w && lane == 0) out_w[row] = recv_w[p];
  }
}

// FP8 payload plus per-128-element scales.
//
// `sf_row_stride` / `sf_col_stride` describe the destination scale layout. The
// caller asks for TMA-aligned column-major scales; there is no TMA on AMD, but
// the column-major *layout* is still what a downstream GEMM wants, so it is
// honoured as a strided write rather than silently ignored.
__global__ void k_expand_scatter_fp8(EpConfig cfg, const fp8_t* __restrict__ recv_x,
                                     const float* __restrict__ recv_sf,   // [nrecv, hidden_sf]
                                     const float* __restrict__ recv_w, const int32_t* __restrict__ row_map,
                                     int num_recv, fp8_t* __restrict__ out_x, float* __restrict__ out_sf,
                                     int sf_row_stride, int sf_col_stride, float* __restrict__ out_w) {
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, nsf = cfg.hidden_sf();
  const size_t n = (size_t)num_recv * K;

  for (size_t p = (size_t)blockIdx.x * nwarps + warp; p < n; p += (size_t)gridDim.x * nwarps) {
    const int row = row_map[p];
    if (row < 0) continue;
    const size_t i = p / K;
    wave_copy_bytes((uint8_t*)(out_x + (size_t)row * cfg.hidden), (const uint8_t*)(recv_x + i * cfg.hidden),
                    cfg.hidden);
    for (int j = lane; j < nsf; j += kWarpSize)
      out_sf[(size_t)row * sf_row_stride + (size_t)j * sf_col_stride] = recv_sf[i * nsf + j];
    if (out_w && lane == 0) out_w[row] = recv_w[p];
  }
}

// Zero the alignment padding rows. Separate kernel so do_zero_padding can be
// switched off without touching the scatter path.
__global__ void k_expand_zero_pad(EpConfig cfg, int expert_alignment, const int32_t* __restrict__ counts,
                                  const int32_t* __restrict__ offsets, int elem_bytes, void* __restrict__ out_x,
                                  float* __restrict__ out_sf, int sf_row_stride, int sf_col_stride,
                                  float* __restrict__ out_w) {
  const int e = blockIdx.x;
  if (e >= cfg.experts_per_rank()) return;

  const int n = counts[e];
  const int npad = align_count(n, expert_alignment) - n;
  const int base = offsets[e] + n;
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int nsf = cfg.hidden_sf();
  const size_t row_bytes = (size_t)cfg.hidden * elem_bytes;

  for (int p = warp; p < npad; p += nwarps) {
    const int row = base + p;
    uint8_t* dst = (uint8_t*)out_x + (size_t)row * row_bytes;
    for (size_t b = lane; b < row_bytes; b += kWarpSize) dst[b] = 0;
    if (out_sf) {
      for (int j = lane; j < nsf; j += kWarpSize) out_sf[(size_t)row * sf_row_stride + (size_t)j * sf_col_stride] = 0.f;
    }
    if (out_w && lane == 0) out_w[row] = 0.f;
  }
}

}  // namespace rccl_ep

#endif  // RCCL_EP_EXPAND_H_
