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

#ifndef LIBRARY_SRC_TDM_POLICY_HPP_
#define LIBRARY_SRC_TDM_POLICY_HPP_

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "constmem.hpp"
#include "util.hpp"
#include "tdm.hpp"

namespace rocshmem {

// Thread 0 drives TDM into the registered LDS buffer; the rest of the block
// covers what TDM doesn't. Falls back to memcpy_wg if TDM isn't available.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wg_tdm(void* dst, void* src,
                                                                size_t size) {
#if ROCSHMEM_TDM_ENABLED
  // Non-temporal + system scope: correct either direction (peer may be remote).
  constexpr int TdmLoadPolicy = tdm::make_cache_policy(
    tdm::LoadTemporalHint::NonTemporal, tdm::Scope::System);
  constexpr int TdmStorePolicy = tdm::make_cache_policy(
    tdm::StoreTemporalHint::NonTemporal, tdm::Scope::System);

  // Below threshold, TDM doesn't pay off (like the SDMA threshold); size
  // is small enough here that Unroll=4 is plenty.
  if (size < constmem.tdm_threshold) {
    memcpy_wg<Kind, 4>(dst, src, size);
    return;
  }
  const tdm::LdsRegistration reg = tdm::get_lds();
  // Below 16 bytes there's no room for a double-buffered tile.
  if (reg.ptr != nullptr && reg.bytes >= 16) {
    constexpr uint32_t element_bytes = 1u << tdm::FlatCopyElementLog2;
    uint32_t tile_bytes = static_cast<uint32_t>(
        min(static_cast<size_t>(constmem.tdm_tile_bytes), reg.bytes / 2));
    // Round down to a whole element -- an unaligned stride would leave
    // gaps between tiles.
    tile_bytes -= tile_bytes % element_bytes;
    if (tile_bytes > 0) {
      if (is_thread_zero_in_block()) {
        tdm::copy_region<TdmLoadPolicy, TdmStorePolicy>(dst, src, size, reg.ptr, tile_bytes,
                                            /*double_buffered=*/true);
      }
      const size_t covered = tdm::covered_bytes(size, tile_bytes);
      if (covered < size) {
        // Runs concurrently with thread 0's TDM copy; Unroll=4 trims
        // register pressure here for free (too small to need 8 anyway).
        memcpy_wg<Kind, 4>(static_cast<char*>(dst) + covered,
                           static_cast<char*>(src) + covered, size - covered);
      }
      return;
    }
  }
#endif
  memcpy_wg<Kind>(dst, src, size);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TDM_POLICY_HPP_
