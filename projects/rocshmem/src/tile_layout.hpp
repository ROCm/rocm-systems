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

#ifndef LIBRARY_SRC_TILE_LAYOUT_HPP_
#define LIBRARY_SRC_TILE_LAYOUT_HPP_

#include <cstddef>
#include <hip/hip_runtime.h>

namespace rocshmem {

enum class TileLayout {
  Contiguous,  // fully packed in both dimensions
  RowContig,   // each row contiguous (stride1 == 1), rows may be gapped
  ColContig,   // each column contiguous (stride0 == 1), cols may be gapped
  Strided      // general strided / element-wise
};

enum class TileScope { Lane, Wave, Wg };

struct TileView {
  char* src_base;
  char* dst_base;
  size_t ext0;
  size_t ext1;
  size_t src_s0;
  size_t src_s1;
  size_t dst_s0;
  size_t dst_s1;
  size_t element_size;
  int ndim;
};

// Layout classes for 2D tiles (element strides).
__device__ __forceinline__ TileLayout tile_classify_2d(
    size_t src_stride0, size_t src_stride1, size_t dst_stride0,
    size_t dst_stride1, [[maybe_unused]] size_t extent0, size_t extent1) {
  const bool row_contig = (src_stride1 == 1 && dst_stride1 == 1);
  const bool col_contig = (src_stride0 == 1 && dst_stride0 == 1);
  if (row_contig && src_stride0 == extent1 && dst_stride0 == extent1) {
    return TileLayout::Contiguous;
  }
  if (row_contig) {
    return TileLayout::RowContig;
  }
  if (col_contig) {
    return TileLayout::ColContig;
  }
  return TileLayout::Strided;
}

__device__ __forceinline__ TileLayout tile_classify(const TileView& view) {
  if (view.ndim == 2) {
    return tile_classify_2d(view.src_s0, view.src_s1, view.dst_s0, view.dst_s1,
                            view.ext0, view.ext1);
  }
  if (view.ndim == 1 && view.src_s0 == 1 && view.dst_s0 == 1) {
    return TileLayout::Contiguous;
  }
  return TileLayout::Strided;
}

// Put applies start_coord to dest; get applies it to source.
__device__ __forceinline__ TileView tile_make_view(
    void* dst_data, const void* src_data, const size_t* dst_strides,
    const size_t* src_strides, const size_t* start_coord,
    const size_t* boundary, int ndim, size_t element_size,
    bool apply_start_to_dst) {
  TileView view{};
  view.src_base = static_cast<char*>(const_cast<void*>(src_data));
  view.dst_base = static_cast<char*>(dst_data);
  view.element_size = element_size;
  view.ndim = ndim;

  if (ndim == 2) {
    view.src_s0 = src_strides[0];
    view.src_s1 = src_strides[1];
    view.dst_s0 = dst_strides[0];
    view.dst_s1 = dst_strides[1];
    view.ext0 = boundary[0] - start_coord[0];
    view.ext1 = boundary[1] - start_coord[1];
    const size_t src_off =
        (start_coord[0] * view.src_s0 + start_coord[1] * view.src_s1) *
        element_size;
    const size_t dst_off =
        (start_coord[0] * view.dst_s0 + start_coord[1] * view.dst_s1) *
        element_size;
    if (apply_start_to_dst) {
      view.dst_base += dst_off;
    } else {
      view.src_base += src_off;
    }
  } else if (ndim == 1) {
    view.src_s0 = src_strides[0];
    view.dst_s0 = dst_strides[0];
    view.src_s1 = 0;
    view.dst_s1 = 0;
    view.ext0 = boundary[0] - start_coord[0];
    view.ext1 = 0;
    if (apply_start_to_dst) {
      view.dst_base += start_coord[0] * view.dst_s0 * element_size;
    } else {
      view.src_base += start_coord[0] * view.src_s0 * element_size;
    }
  }

  return view;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TILE_LAYOUT_HPP_
