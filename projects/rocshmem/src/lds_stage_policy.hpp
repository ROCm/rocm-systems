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

#ifndef LIBRARY_SRC_LDS_STAGE_POLICY_HPP_
#define LIBRARY_SRC_LDS_STAGE_POLICY_HPP_

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "constmem.hpp"
#include "util.hpp"
#include "lds_stage.hpp"

namespace rocshmem {

// TDM backend: single-lane descriptor issue, so the cache policy is a real
// TDM cachepolicy immediate. Async backend: issue_load_tile/issue_store_tile
// ignore these template parameters entirely (see lds_stage.hpp), so any
// value compiles away -- 0 is a placeholder, not a meaningful policy.
#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)
template <MemcpyKind Kind>
constexpr int lds_stage_load_policy_v =
    is_put(Kind) ? lds_stage::tdm::make_cache_policy(lds_stage::tdm::LoadTemporalHint::Default,
                                                     lds_stage::tdm::Scope::Wgp)
                 : lds_stage::tdm::make_cache_policy(lds_stage::tdm::LoadTemporalHint::NonTemporal,
                                                     lds_stage::tdm::Scope::System);
template <MemcpyKind Kind>
constexpr int lds_stage_store_policy_v =
    is_put(Kind)
        ? lds_stage::tdm::make_cache_policy(lds_stage::tdm::StoreTemporalHint::NonTemporal,
                                            lds_stage::tdm::Scope::System)
        : lds_stage::tdm::make_cache_policy(lds_stage::tdm::StoreTemporalHint::Default,
                                            lds_stage::tdm::Scope::Wgp);
#elif defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)
template <MemcpyKind Kind>
constexpr int lds_stage_load_policy_v = 0;
template <MemcpyKind Kind>
constexpr int lds_stage_store_policy_v = 0;
#endif

// Work-group counterpart: on the TDM backend, thread 0 of the block drives
// TDM into the registered LDS buffer while the rest of the block covers what
// TDM doesn't -- concurrently, since thread 0 doesn't reach the remainder
// copy below until its (blocking) TDM call returns. On the async backend,
// every thread in the block participates in the tile loop itself (each lane
// issues its own slice of every tile), so there is no spare lane to run the
// remainder concurrently -- it runs right after, same call site, same
// structure, just not overlapped.
// Falls back to memcpy_wg if no backend is available.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wg_lds_stage(void* dst, void* src,
                                                                      size_t size) {
#if ROCSHMEM_LDS_STAGE_ENABLED
  constexpr int LoadPolicy = lds_stage_load_policy_v<Kind>;
  constexpr int StorePolicy = lds_stage_store_policy_v<Kind>;

  // Below threshold, this path doesn't pay off (like the SDMA threshold);
  // size is small enough here that Unroll=4 is plenty.
  if (size < (is_put(Kind) ? constmem.lds_stage_put_threshold : constmem.lds_stage_get_threshold)) {
    memcpy_wg<Kind, 4>(dst, src, size);
    return;
  }
  const lds_stage::LdsRegistration reg = lds_stage::get_lds();
  // Below 16 bytes there's no room for a double-buffered tile.
  if (reg.ptr != nullptr && reg.bytes >= 16) {
    constexpr uint32_t element_bytes = 1u << lds_stage::FlatCopyElementLog2;
    uint32_t tile_bytes = static_cast<uint32_t>(
        min(static_cast<size_t>(constmem.lds_stage_tile_bytes), reg.bytes / 2));
    // Round down to a whole element -- an unaligned stride would leave
    // gaps between tiles.
    tile_bytes -= tile_bytes % element_bytes;
    if (tile_bytes > 0) {
#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)
      if (is_thread_zero_in_block()) {
        lds_stage::copy_pipelined_deep<LoadPolicy, StorePolicy>(dst, src, size, reg.ptr,
                                                                 tile_bytes);
      }
#elif defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)
      lds_stage::copy_pipelined_deep<LoadPolicy, StorePolicy>(
          dst, src, size, reg.ptr, tile_bytes, static_cast<uint32_t>(get_flat_block_id()),
          static_cast<uint32_t>(get_flat_block_size()));
#endif
      const size_t covered = lds_stage::covered_bytes(size, tile_bytes);
      if (covered < size) {
        // On the TDM backend this runs concurrently with thread 0's copy
        // (thread 0 joins in once its own call returns); on the async
        // backend every thread reaches this at roughly the same time.
        // Unroll=4 trims register pressure here for free (too small to
        // need 8 anyway).
        memcpy_wg<Kind, 4>(static_cast<char*>(dst) + covered,
                           static_cast<char*>(src) + covered, size - covered);
      }
      return;
    }
  }
#endif
  memcpy_wg<Kind>(dst, src, size);
}

// Wave counterpart of memcpy_wg_lds_stage: on the TDM backend, lane 0 of the
// calling wave drives TDM (Put uses the shallower, single-buffer-optional
// pipeline; Get uses the deeper one -- a deliberate, already-tuned
// per-direction asymmetry, not something to flatten away), the rest covers
// what TDM doesn't. On the async backend, every lane of the wave
// participates in both the Put and Get tile loop (there's no per-direction
// asymmetry to preserve there -- that tuning was specific to TDM's
// descriptor-issue cost, which doesn't apply to a per-lane VMEM instruction).
// The registered buffer is split into one fixed slot per wave the block
// could launch (not just the ones that call this), keeping slots disjoint
// without runtime coordination -- more waves per block means a smaller slot
// (and tile) per wave.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wave_lds_stage(void* dst, void* src,
                                                                        size_t size) {
#if ROCSHMEM_LDS_STAGE_ENABLED
  constexpr int LoadPolicy = lds_stage_load_policy_v<Kind>;
  constexpr int StorePolicy = lds_stage_store_policy_v<Kind>;

  if (size < (is_put(Kind) ? constmem.lds_stage_put_threshold : constmem.lds_stage_get_threshold)) {
    memcpy_wave<Kind>(dst, src, size);
    return;
  }
  const lds_stage::LdsRegistration reg = lds_stage::get_lds();
  // Round up: a partial trailing wavefront still needs its own slot --
  // undercounting would let its wave_id index past its slice.
  const int num_waves = max((get_flat_block_size() + WF_SIZE - 1) / WF_SIZE, 1);
  constexpr uint32_t element_bytes = 1u << lds_stage::FlatCopyElementLog2;
  // Round down to a whole element -- reg.ptr is only guaranteed 8-byte
  // aligned, so an unaligned stride could start a wave's slot (and its
  // tile) on an unaligned address.
  size_t slot_bytes = reg.bytes / static_cast<size_t>(num_waves);
  slot_bytes -= slot_bytes % element_bytes;
  // Below 16 bytes there's no room for a double-buffered tile in this slot.
  if (reg.ptr != nullptr && slot_bytes >= 16) {
    uint32_t tile_bytes = static_cast<uint32_t>(
        min(static_cast<size_t>(constmem.lds_stage_tile_bytes), slot_bytes / 2));
    // Round down to a whole element -- an unaligned stride would leave
    // gaps between tiles.
    tile_bytes -= tile_bytes % element_bytes;
    if (tile_bytes > 0) {
      const int wave_id = get_flat_block_id() / WF_SIZE;
      void* wave_lds = static_cast<uint8_t*>(reg.ptr) +
                        static_cast<size_t>(wave_id) * slot_bytes;
#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)
      if (is_thread_zero_in_wave()) {
        if constexpr (is_put(Kind)) {
          lds_stage::copy_pipelined_shallow<LoadPolicy, StorePolicy>(
              dst, src, size, wave_lds, tile_bytes, /*double_buffered=*/true);
        } else {
          lds_stage::copy_pipelined_deep<LoadPolicy, StorePolicy>(dst, src, size, wave_lds,
                                                                   tile_bytes);
        }
      }
#elif defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)
      {
        const uint32_t wave_tid = static_cast<uint32_t>(get_flat_block_id() % WF_SIZE);
        const uint32_t wave_size = static_cast<uint32_t>(wave_SZ());
        lds_stage::copy_pipelined_deep<LoadPolicy, StorePolicy>(dst, src, size, wave_lds,
                                                                 tile_bytes, wave_tid, wave_size);
      }
#endif
      const size_t covered = lds_stage::covered_bytes(size, tile_bytes);
      if (covered < size) {
        memcpy_wave<Kind>(static_cast<char*>(dst) + covered,
                          static_cast<char*>(src) + covered, size - covered);
      }
      return;
    }
  }
#endif
  memcpy_wave<Kind>(dst, src, size);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_LDS_STAGE_POLICY_HPP_
