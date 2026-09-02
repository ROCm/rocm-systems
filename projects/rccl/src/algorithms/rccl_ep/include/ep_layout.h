/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Symmetric-window layout, a push model: sender s writes directly into region s
 * of receiver d's window, so the receiver concatenates regions in rank order
 * and gets tokens ordered by src_token_global_idx with no sort.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_LAYOUT_H_
#define RCCL_EP_LAYOUT_H_

#include <cstddef>
#include <cstdint>

namespace rccl_ep {

constexpr int kMaxRanks = 16;   // intranode scale-up domain
constexpr int kMaxTopk = 16;
// Fixed width of one gcnArchName slot in the window's arch region.
constexpr int kArchLen = 64;

// Alignment for every region start. 256 B keeps each region on a cache-line
// and dword4 boundary so peer stores never straddle.
constexpr size_t kAlign = 256;

__host__ __device__ inline size_t align_up(size_t v, size_t a = kAlign) {
  return (v + a - 1) / a * a;
}

// One EP configuration; pointer-free, cheap to pass by value into kernels.
struct EpConfig {
  int num_ranks;
  int rank;
  int num_experts;              // total, across all ranks
  int num_topk;
  int hidden;                   // elements per token
  int num_max_tokens_per_rank;  // capacity, per source rank

  __host__ __device__ int experts_per_rank() const {
    return num_experts / num_ranks;
  }
  // One scale per 128-element block: hidden_sf = ceil_div(hidden, 128).
  __host__ __device__ int hidden_sf() const {
    return (hidden + 127) / 128;
  }
  __host__ __device__ int expert_begin(int r) const {
    return r * experts_per_rank();
  }
  __host__ __device__ int expert_end(int r) const {
    return (r + 1) * experts_per_rank();
  }
};

// Byte offsets of each region within a rank's symmetric window.
//
//   [ counts    ] num_ranks int32          how many tokens source s sent me
//   [ flags     ] num_ranks uint32         per-source completion flag
//   [ x         ] R * T * hidden  (bf16)   token payload, staged by source
//   [ topk_idx  ] R * T * num_topk int32
//   [ topk_w    ] R * T * num_topk float
//   [ src_idx   ] R * T int32              src_token_global_idx
//   [ y         ] R * T * num_topk * hidden  combine payload, staged by owner
//   [ cw        ] R * T * num_topk float   combine top-k weights, by owner
//
// where R = num_ranks, T = num_max_tokens_per_rank.
//
// `cw` is deliberately not an alias of `topk_w`: aliasing works today but
// breaks the first time a dispatch result is read back after a combine.
// `y` is per (slot, topk), not per slot, because combine sums in strict topk
// order; pre-reducing per rank reorders it and fails the bit-exact bar.
struct EpWindowLayout {
  size_t off_counts, off_flags, off_x, off_sf, off_topk_idx, off_topk_w, off_src_idx, off_y, off_cw, off_arch;
  size_t total_bytes;

  __host__ __device__ static EpWindowLayout make(const EpConfig& c, size_t elem_bytes) {
    const size_t R = c.num_ranks;
    const size_t T = c.num_max_tokens_per_rank;
    const size_t slots = R * T;
    EpWindowLayout l{};
    size_t o = 0;
    l.off_counts = o;
    o = align_up(o + R * sizeof(int32_t));
    l.off_flags = o;
    o = align_up(o + R * sizeof(uint32_t));
    l.off_x = o;
    o = align_up(o + slots * c.hidden * elem_bytes);
    // FP8 scaling factors; always reserved so a layout can serve both dtypes.
    l.off_sf = o;
    o = align_up(o + slots * c.hidden_sf() * sizeof(float));
    l.off_topk_idx = o;
    o = align_up(o + slots * c.num_topk * sizeof(int32_t));
    l.off_topk_w = o;
    o = align_up(o + slots * c.num_topk * sizeof(float));
    l.off_src_idx = o;
    o = align_up(o + slots * sizeof(int32_t));
    // `y` is always bf16: combine payload is bf16 even when dispatch is fp8.
    l.off_y = o;
    o = align_up(o + slots * c.num_topk * c.hidden * sizeof(uint16_t));
    l.off_cw = o;
    o = align_up(o + slots * c.num_topk * sizeof(float));
    // Architecture strings, one slot per rank, used once at construction to
    // reject a mixed-architecture communicator (see include/ep_arch.h).
    l.off_arch = o;
    o = align_up(o + (size_t)R * kArchLen);
    l.total_bytes = o;
    return l;
  }

  // Slot index of the i-th token staged from source rank s.
  __host__ __device__ static size_t slot(const EpConfig& c, int s, int i) {
    return (size_t)s * c.num_max_tokens_per_rank + i;
  }
};

}  // namespace rccl_ep

#endif  // RCCL_EP_LAYOUT_H_
