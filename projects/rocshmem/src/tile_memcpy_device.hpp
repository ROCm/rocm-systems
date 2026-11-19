/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_TILE_MEMCPY_DEVICE_HPP_
#define LIBRARY_SRC_TILE_MEMCPY_DEVICE_HPP_

#include <cstdint>

#include "tile_layout.hpp"
#include "util.hpp"

namespace rocshmem {

template <TileScope Scope>
__device__ __forceinline__ void tile_scope_workers(int* worker_id,
                                                   int* worker_count) {
  if constexpr (Scope == TileScope::Lane) {
    *worker_id = 0;
    *worker_count = 1;
  } else if constexpr (Scope == TileScope::Wave) {
    *worker_id = get_flat_block_id() % WF_SIZE;
    *worker_count = WF_SIZE;
  } else {
    *worker_id = get_flat_block_id();
    *worker_count = get_flat_block_size();
  }
}

// The generic memcpy_wg path uses an out-of-line bulk-copy helper. Tile
// workgroup copies are frequently only a few KiB, so keeping the short vector
// loop in the caller avoids the call frame and distributes the vectors over
// the whole workgroup.
template <MemcpyKind Kind, int W>
__device__ __forceinline__ void tile_copy_contig_wg_vectors(
    char* dst, char* src, uint32_t bytes, int worker_id, int worker_count) {
  constexpr CachePolicy LP =
      is_put(Kind) ? CachePolicy::Standard : CachePolicy::SystemScope;
  constexpr CachePolicy SP =
      is_put(Kind) ? CachePolicy::SystemScope : CachePolicy::Standard;
  using Acc = AsmAccess<W, LP, SP>;
  using T = typename Acc::type;
  constexpr int Unroll = 4;

  const uint32_t total = bytes / W;
  const uint32_t stride = static_cast<uint32_t>(worker_count);
  uint32_t v = static_cast<uint32_t>(worker_id);

  for (; v + (Unroll - 1) * stride < total; v += Unroll * stride) {
    T regs[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      const uint32_t offset = (v + u * stride) * W;
      if constexpr (is_put(Kind)) {
        regs[u] = Acc::load(src + offset);
      } else {
        regs[u] = Acc::load_buffer(src, bytes, offset);
      }
    }
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      const uint32_t offset = (v + u * stride) * W;
      if constexpr (is_put(Kind)) {
        Acc::store_buffer(dst, bytes, offset, regs[u]);
      } else {
        Acc::store(dst + offset, regs[u]);
      }
    }
  }
  for (; v < total; v += stride) {
    const uint32_t offset = v * W;
    if constexpr (is_put(Kind)) {
      Acc::store_buffer(dst, bytes, offset, Acc::load(src + offset));
    } else {
      Acc::store(dst + offset, Acc::load_buffer(src, bytes, offset));
    }
  }
}

template <MemcpyKind Kind>
__device__ __forceinline__ void tile_memcpy_contig_wg(char* dst, char* src,
                                                       size_t bytes,
                                                       int worker_id,
                                                       int worker_count) {
  if (bytes == 0) return;
  if (bytes > UINT32_MAX) {
    memcpy_wg<Kind>(dst, src, bytes);
    return;
  }

  const uintptr_t bits = bytes | reinterpret_cast<uintptr_t>(dst) |
                         reinterpret_cast<uintptr_t>(src);
  const uint32_t size = static_cast<uint32_t>(bytes);
  if ((bits & 15) == 0) {
    tile_copy_contig_wg_vectors<Kind, 16>(dst, src, size, worker_id,
                                          worker_count);
  } else if ((bits & 7) == 0) {
    tile_copy_contig_wg_vectors<Kind, 8>(dst, src, size, worker_id,
                                         worker_count);
  } else if ((bits & 3) == 0) {
    tile_copy_contig_wg_vectors<Kind, 4>(dst, src, size, worker_id,
                                         worker_count);
  } else if ((bits & 1) == 0) {
    tile_copy_contig_wg_vectors<Kind, 2>(dst, src, size, worker_id,
                                         worker_count);
  } else {
    tile_copy_contig_wg_vectors<Kind, 1>(dst, src, size, worker_id,
                                         worker_count);
  }
}

template <MemcpyKind Kind, TileScope Scope>
__device__ __forceinline__ void tile_memcpy_contig(char* dst, char* src,
                                                   size_t bytes,
                                                   int worker_id) {
  if constexpr (Scope == TileScope::Wave) {
    memcpy_wave<Kind>(dst, src, bytes);
  } else if constexpr (Scope == TileScope::Wg) {
    tile_memcpy_contig_wg<Kind>(dst, src, bytes, worker_id,
                                get_flat_block_size());
  } else {
    memcpy_lane<Kind>(dst, src, bytes);
  }
}

// A strided tile expressed as equally sized contiguous byte runs. Each
// dimension packed into its inner neighbour on both sides is merged with it,
// so runs are as long as the layout allows. Every merged group of outer
// dimensions occupies the slot of its outermost dimension; all other slots
// have extent 1. Slots are only ever indexed by compile-time constants so the
// struct stays in registers.
struct TileRuns {
  uint32_t ext[TileView::MAX_NDIM];
  uint32_t src_sb[TileView::MAX_NDIM];  // byte strides
  uint32_t dst_sb[TileView::MAX_NDIM];
  uint32_t run_bytes;
  uint32_t num_runs;
  uint32_t src_span;  // bytes from the base to the end of the furthest run
  uint32_t dst_span;
};

// Returns false when the tile cannot be addressed with 32-bit byte offsets.
__device__ inline bool tile_make_runs(const TileView& view, TileRuns* r) {
  constexpr size_t limit = INT32_MAX;
  const size_t esize = view.element_size;

  size_t ext[TileView::MAX_NDIM];
  size_t ss[TileView::MAX_NDIM];
  size_t ds[TileView::MAX_NDIM];
#pragma unroll
  for (int d = 0; d < TileView::MAX_NDIM; d++) {
    ext[d] = 1;
    ss[d] = 0;
    ds[d] = 0;
  }

  // Current group of merged dimensions, innermost first.
  size_t g_ext = 1, g_ss = 0, g_ds = 0;
  bool have_group = false;
  bool innermost = true;
  size_t run_elems = 1;
  bool empty = false;

  // Only the innermost group can form the contiguous run.
  auto close_group = [&](int slot) {
    const bool is_run = innermost && g_ss == 1 && g_ds == 1;
    innermost = false;
    if (is_run) {
      run_elems = g_ext;
    } else {
      ext[slot] = g_ext;
      ss[slot] = g_ss;
      ds[slot] = g_ds;
    }
  };

#pragma unroll
  for (int d = TileView::MAX_NDIM - 1; d >= 0; d--) {
    if (d < view.ndim) {
      const size_t e = view.ext[d];
      empty |= (e == 0);
      if (e > 1) {
        const size_t s = view.src_s[d];
        const size_t t = view.dst_s[d];
        if (have_group && s == g_ext * g_ss && t == g_ext * g_ds) {
          g_ext *= e;
        } else {
          if (have_group && d + 1 < TileView::MAX_NDIM) close_group(d + 1);
          have_group = true;
          g_ext = e;
          g_ss = s;
          g_ds = t;
        }
      }
    }
  }
  if (have_group) close_group(0);

  if (empty) {
    r->num_runs = 0;
    return true;
  }

  const size_t run_bytes = run_elems * esize;
  size_t num_runs = 1;
  size_t src_span = run_bytes;
  size_t dst_span = run_bytes;
#pragma unroll
  for (int d = 0; d < TileView::MAX_NDIM; d++) {
    num_runs *= ext[d];
    src_span += (ext[d] - 1) * ss[d] * esize;
    dst_span += (ext[d] - 1) * ds[d] * esize;
  }
  if (run_bytes > limit || num_runs > limit || num_runs * run_bytes > limit ||
      src_span > limit || dst_span > limit) {
    return false;
  }

#pragma unroll
  for (int d = 0; d < TileView::MAX_NDIM; d++) {
    r->ext[d] = static_cast<uint32_t>(ext[d]);
    r->src_sb[d] = static_cast<uint32_t>(ss[d] * esize);
    r->dst_sb[d] = static_cast<uint32_t>(ds[d] * esize);
  }
  r->run_bytes = static_cast<uint32_t>(run_bytes);
  r->num_runs = static_cast<uint32_t>(num_runs);
  r->src_span = static_cast<uint32_t>(src_span);
  r->dst_span = static_cast<uint32_t>(dst_span);
  return true;
}

// Widest access size that keeps every vector of every run naturally aligned.
__device__ __forceinline__ int tile_runs_vec_width(const TileRuns& r,
                                                   const char* dst,
                                                   const char* src) {
  uintptr_t bits = r.run_bytes | reinterpret_cast<uintptr_t>(dst) |
                   reinterpret_cast<uintptr_t>(src);
#pragma unroll
  for (int i = 0; i < TileView::MAX_NDIM; i++) bits |= r.src_sb[i] | r.dst_sb[i];
  if ((bits & 15) == 0) return 16;
  if ((bits & 7) == 0) return 8;
  if ((bits & 3) == 0) return 4;
  if ((bits & 1) == 0) return 2;
  return 1;
}

// Vectors are striped across workers in run-major order, so adjacent lanes
// touch adjacent addresses within a run. The remote side goes through buffer
// instructions, which the compiler can keep in flight without a wait per
// access; the local side uses plain accesses so it may live in any segment.
template <MemcpyKind Kind, int W>
__device__ inline void tile_copy_runs(char* dst, char* src, const TileRuns& r,
                                      int worker_id, int worker_count) {
  constexpr CachePolicy LP =
      is_put(Kind) ? CachePolicy::Standard : CachePolicy::SystemScope;
  constexpr CachePolicy SP =
      is_put(Kind) ? CachePolicy::SystemScope : CachePolicy::Standard;
  using Acc = AsmAccess<W, LP, SP>;
  using T = typename Acc::type;
  constexpr int Unroll = 4;

  const uint32_t vecs_per_run = r.run_bytes / W;
  const uint32_t total = r.num_runs * vecs_per_run;
  const uint32_t stride = static_cast<uint32_t>(worker_count);

  auto offsets = [&](uint32_t v, uint32_t* src_off, uint32_t* dst_off) {
    uint32_t run = v / vecs_per_run;
    const uint32_t in_run = (v - run * vecs_per_run) * W;
    uint32_t s = in_run, d = in_run;
#pragma unroll
    for (int i = TileView::MAX_NDIM - 1; i >= 0; i--) {
      if (r.ext[i] > 1) {
        const uint32_t coord = run % r.ext[i];
        run /= r.ext[i];
        s += coord * r.src_sb[i];
        d += coord * r.dst_sb[i];
      }
    }
    *src_off = s;
    *dst_off = d;
  };
  auto load = [&](uint32_t off) -> T {
    if constexpr (is_put(Kind)) {
      return Acc::load(src + off);
    } else {
      return Acc::load_buffer(src, r.src_span, off);
    }
  };
  auto store = [&](uint32_t off, T val) {
    if constexpr (is_put(Kind)) {
      Acc::store_buffer(dst, r.dst_span, off, val);
    } else {
      Acc::store(dst + off, val);
    }
  };

  uint32_t v = static_cast<uint32_t>(worker_id);
  for (; v + (Unroll - 1) * stride < total; v += Unroll * stride) {
    uint32_t src_off[Unroll], dst_off[Unroll];
    T regs[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      offsets(v + u * stride, &src_off[u], &dst_off[u]);
    }
#pragma unroll
    for (int u = 0; u < Unroll; u++) regs[u] = load(src_off[u]);
#pragma unroll
    for (int u = 0; u < Unroll; u++) store(dst_off[u], regs[u]);
  }
  for (; v < total; v += stride) {
    uint32_t src_off, dst_off;
    offsets(v, &src_off, &dst_off);
    store(dst_off, load(src_off));
  }
}

// Returns false when the tile is too large for the run-based copy.
template <MemcpyKind Kind>
__device__ inline bool tile_memcpy_runs(const TileView& view, int worker_id,
                                        int worker_count) {
  TileRuns r;
  if (!tile_make_runs(view, &r)) return false;
  if (r.num_runs == 0) return true;

  char* dst = view.dst_base;
  char* src = view.src_base;
  switch (tile_runs_vec_width(r, dst, src)) {
    case 16: tile_copy_runs<Kind, 16>(dst, src, r, worker_id, worker_count); break;
    case 8:  tile_copy_runs<Kind, 8>(dst, src, r, worker_id, worker_count);  break;
    case 4:  tile_copy_runs<Kind, 4>(dst, src, r, worker_id, worker_count);  break;
    case 2:  tile_copy_runs<Kind, 2>(dst, src, r, worker_id, worker_count);  break;
    default: tile_copy_runs<Kind, 1>(dst, src, r, worker_id, worker_count);  break;
  }
  return true;
}

// On-node tile put/get via memcpy. Quiet remains the caller's responsibility.
// Put: dst_data is the remote symmetric pointer; src_data is local.
// Get: dst_data is local; src_data is the remote symmetric pointer.
template <MemcpyKind Kind, TileScope Scope>
__device__ inline void tile_memcpy_rma(
    void* dst_data, const void* src_data, const size_t* dst_strides,
    const size_t* src_strides, const size_t* start_coord,
    const size_t* boundary, int ndim, size_t element_size) {
  const TileView view =
      tile_make_view(dst_data, src_data, dst_strides, src_strides, start_coord,
                     boundary, ndim, element_size);

  int worker_id{0};
  int worker_count{1};
  tile_scope_workers<Scope>(&worker_id, &worker_count);

  if (view.ndim == 2) {
    const TileLayout layout = tile_classify(view);
    if (layout == TileLayout::Contiguous) {
      tile_memcpy_contig<Kind, Scope>(
          view.dst_base, view.src_base,
          view.ext[0] * view.ext[1] * view.element_size, worker_id);
    } else if (layout == TileLayout::RowContig) {
      if (!tile_memcpy_runs<Kind>(view, worker_id, worker_count)) {
        const size_t row_size = view.ext[1] * view.element_size;
        for (size_t i = worker_id; i < view.ext[0]; i += worker_count) {
          memcpy_lane<Kind>(
              view.dst_base + i * view.dst_s[0] * view.element_size,
              view.src_base + i * view.src_s[0] * view.element_size, row_size);
        }
      }
    } else if (layout == TileLayout::ColContig) {
      if (!tile_memcpy_runs<Kind>(view, worker_id, worker_count)) {
        const size_t col_size = view.ext[0] * view.element_size;
        for (size_t j = worker_id; j < view.ext[1]; j += worker_count) {
          memcpy_lane<Kind>(
              view.dst_base + j * view.dst_s[1] * view.element_size,
              view.src_base + j * view.src_s[1] * view.element_size, col_size);
        }
      }
    } else {
      if (!tile_memcpy_runs<Kind>(view, worker_id, worker_count)) {
        const int total_elements = view.ext[0] * view.ext[1];
        for (int idx = worker_id; idx < total_elements; idx += worker_count) {
          const int i = idx / view.ext[1];
          const int j = idx % view.ext[1];
          memcpy_lane<Kind>(
              view.dst_base +
                  (i * view.dst_s[0] + j * view.dst_s[1]) * view.element_size,
              view.src_base +
                  (i * view.src_s[0] + j * view.src_s[1]) * view.element_size,
              view.element_size);
        }
      }
    }
  } else if (view.ndim == 1) {
    if (tile_classify(view) == TileLayout::Contiguous) {
      tile_memcpy_contig<Kind, Scope>(view.dst_base, view.src_base,
                                      view.ext[0] * view.element_size,
                                      worker_id);
    } else if (!tile_memcpy_runs<Kind>(view, worker_id, worker_count)) {
      for (size_t i = worker_id; i < view.ext[0]; i += worker_count) {
        memcpy_lane<Kind>(
            view.dst_base + i * view.dst_s[0] * view.element_size,
            view.src_base + i * view.src_s[0] * view.element_size,
            view.element_size);
      }
    }
  } else if (view.ndim > 2 && view.ndim <= TileView::MAX_NDIM) {
    // N-dimensional (3D/4D): compute total element count then dispatch.
    size_t total = 1;
    for (int d = 0; d < view.ndim; d++) total *= view.ext[d];

    if (tile_classify(view) == TileLayout::Contiguous) {
      tile_memcpy_contig<Kind, Scope>(view.dst_base, view.src_base,
                                     total * view.element_size, worker_id);
    } else if (!tile_memcpy_runs<Kind>(view, worker_id, worker_count)) {
      // General strided: recover per-dimension coordinates from a flat index.
      for (size_t idx = worker_id; idx < total; idx += worker_count) {
        size_t src_off = 0, dst_off = 0, rem = idx;
        for (int d = view.ndim - 1; d >= 0; d--) {
          const size_t coord = rem % view.ext[d];
          rem /= view.ext[d];
          src_off += coord * view.src_s[d];
          dst_off += coord * view.dst_s[d];
        }
        memcpy_lane<Kind>(
            view.dst_base + dst_off * view.element_size,
            view.src_base + src_off * view.element_size,
            view.element_size);
      }
    }
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TILE_MEMCPY_DEVICE_HPP_
