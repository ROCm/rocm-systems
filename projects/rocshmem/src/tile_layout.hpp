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
  // Maximum tensor rank supported by the tile API.
  static constexpr int MAX_NDIM = 4;

  char* src_base;
  char* dst_base;
  size_t ext[MAX_NDIM];    // extent (boundary - start_coord) per dimension
  size_t src_s[MAX_NDIM];  // source strides in elements per dimension
  size_t dst_s[MAX_NDIM];  // destination strides in elements per dimension
  size_t element_size;
  int ndim;
};

// Layout classification for 2D tiles (element strides).
__device__ __forceinline__ TileLayout tile_classify_2d(
    size_t src_stride0, size_t src_stride1, size_t dst_stride0,
    size_t dst_stride1, size_t extent0, size_t extent1) {
  const bool row_contig = (src_stride1 == 1 && dst_stride1 == 1);
  const bool col_contig = (src_stride0 == 1 && dst_stride0 == 1);
  if (row_contig && src_stride0 == extent1 && dst_stride0 == extent1) {
    return TileLayout::Contiguous;
  }
  if (col_contig && src_stride1 == extent0 && dst_stride1 == extent0) {
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
    return tile_classify_2d(view.src_s[0], view.src_s[1],
                            view.dst_s[0], view.dst_s[1],
                            view.ext[0],   view.ext[1]);
  }
  if (view.ndim == 1 && view.src_s[0] == 1 && view.dst_s[0] == 1) {
    return TileLayout::Contiguous;
  }
  if (view.ndim > 2) {
    // For ndim > 2: contiguous when the innermost dimension is packed and
    // each outer dimension's stride equals the product of inner extents.
    bool contiguous = (view.src_s[view.ndim - 1] == 1 &&
                       view.dst_s[view.ndim - 1] == 1);
    if (contiguous) {
      size_t inner = view.ext[view.ndim - 1];
      for (int d = view.ndim - 2; d >= 0; d--) {
        if (view.src_s[d] != inner || view.dst_s[d] != inner) {
          contiguous = false;
          break;
        }
        inner *= view.ext[d];
      }
    }
    return contiguous ? TileLayout::Contiguous : TileLayout::Strided;
  }
  return TileLayout::Strided;
}

__device__ __forceinline__ TileView tile_make_view(
    void* dst_data, const void* src_data, const size_t* dst_strides,
    const size_t* src_strides, const size_t* start_coord,
    const size_t* boundary, int ndim, size_t element_size) {
  TileView view{};
  view.src_base = static_cast<char*>(const_cast<void*>(src_data));
  view.dst_base = static_cast<char*>(dst_data);
  view.element_size = element_size;
  view.ndim = ndim;

  size_t src_offset = 0;
  size_t dst_offset = 0;
  for (int d = 0; d < ndim; d++) {
    view.src_s[d] = src_strides[d];
    view.dst_s[d] = dst_strides[d];
    view.ext[d]   = boundary[d] - start_coord[d];
    src_offset   += start_coord[d] * src_strides[d];
    dst_offset   += start_coord[d] * dst_strides[d];
  }
  view.src_base += src_offset * element_size;
  view.dst_base += dst_offset * element_size;

  return view;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TILE_LAYOUT_HPP_
