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

template <MemcpyKind Kind, TileScope Scope>
__device__ __forceinline__ void tile_memcpy_contig(char* dst, char* src,
                                                   size_t bytes,
                                                   int worker_id) {
  if constexpr (Scope == TileScope::Wave) {
    memcpy_wave<Kind>(dst, src, bytes);
  } else if constexpr (Scope == TileScope::Wg) {
    if (worker_id == 0) {
      memcpy_lane<Kind>(dst, src, bytes);
    }
  } else {
    memcpy_lane<Kind>(dst, src, bytes);
  }
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
                     boundary, ndim, element_size, is_put(Kind));

  int worker_id{0};
  int worker_count{1};
  tile_scope_workers<Scope>(&worker_id, &worker_count);

  if (view.ndim == 2) {
    const TileLayout layout = tile_classify(view);
    if (layout == TileLayout::Contiguous) {
      tile_memcpy_contig<Kind, Scope>(
          view.dst_base, view.src_base,
          view.ext0 * view.ext1 * view.element_size, worker_id);
    } else if (layout == TileLayout::RowContig) {
      const size_t row_size = view.ext1 * view.element_size;
      for (size_t i = worker_id; i < view.ext0; i += worker_count) {
        memcpy_lane<Kind>(view.dst_base + i * view.dst_s0 * view.element_size,
                          view.src_base + i * view.src_s0 * view.element_size,
                          row_size);
      }
    } else if (layout == TileLayout::ColContig) {
      const size_t col_size = view.ext0 * view.element_size;
      for (size_t j = worker_id; j < view.ext1; j += worker_count) {
        memcpy_lane<Kind>(view.dst_base + j * view.dst_s1 * view.element_size,
                          view.src_base + j * view.src_s1 * view.element_size,
                          col_size);
      }
    } else {
      const int total_elements = view.ext0 * view.ext1;
      for (int idx = worker_id; idx < total_elements; idx += worker_count) {
        const int i = idx / view.ext1;
        const int j = idx % view.ext1;
        memcpy_lane<Kind>(
            view.dst_base +
                (i * view.dst_s0 + j * view.dst_s1) * view.element_size,
            view.src_base +
                (i * view.src_s0 + j * view.src_s1) * view.element_size,
            view.element_size);
      }
    }
  } else if (view.ndim == 1) {
    if (tile_classify(view) == TileLayout::Contiguous) {
      tile_memcpy_contig<Kind, Scope>(view.dst_base, view.src_base,
                                      view.ext0 * view.element_size,
                                      worker_id);
    } else {
      for (size_t i = worker_id; i < view.ext0; i += worker_count) {
        memcpy_lane<Kind>(
            view.dst_base + i * view.dst_s0 * view.element_size,
            view.src_base + i * view.src_s0 * view.element_size,
            view.element_size);
      }
    }
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TILE_MEMCPY_DEVICE_HPP_
