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

#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)

// Thread 0 drives TDM; the rest covers what TDM doesn't.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wg_lds_stage(void* dst, void* src,
                                                                      size_t size) {
  constexpr int LoadPolicy = is_put(Kind)
      ? lds_stage::tdm::make_cache_policy(lds_stage::tdm::LoadTemporalHint::Default,
                                          lds_stage::tdm::Scope::Wgp)
      : lds_stage::tdm::make_cache_policy(lds_stage::tdm::LoadTemporalHint::NonTemporal,
                                          lds_stage::tdm::Scope::System);
  constexpr int StorePolicy = is_put(Kind)
      ? lds_stage::tdm::make_cache_policy(lds_stage::tdm::StoreTemporalHint::NonTemporal,
                                          lds_stage::tdm::Scope::System)
      : lds_stage::tdm::make_cache_policy(lds_stage::tdm::StoreTemporalHint::Default,
                                          lds_stage::tdm::Scope::Wgp);

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
    // Round down to a whole element -- avoids gaps between tiles.
    tile_bytes -= tile_bytes % element_bytes;
    if (tile_bytes > 0) {
      if (is_thread_zero_in_block()) {
        lds_stage::tdm::copy_pipelined_deep<LoadPolicy, StorePolicy>(dst, src, size, reg.ptr,
                                                                      tile_bytes);
      }
      const size_t covered = lds_stage::covered_bytes(size, tile_bytes);
      if (covered < size) {
        // Concurrent with thread 0's TDM copy; Unroll=4 trims reg pressure.
        memcpy_wg<Kind, 4>(static_cast<char*>(dst) + covered,
                           static_cast<char*>(src) + covered, size - covered);
      }
      return;
    }
  }
  memcpy_wg<Kind>(dst, src, size);
}

// Wave counterpart: lane 0 drives TDM (Put shallow, Get deep -- tuned
// asymmetry). Buffer splits one fixed slot per possible wave, so slots
// stay disjoint without runtime coordination.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wave_lds_stage(void* dst, void* src,
                                                                        size_t size) {
  constexpr int LoadPolicy = is_put(Kind)
      ? lds_stage::tdm::make_cache_policy(lds_stage::tdm::LoadTemporalHint::Default,
                                          lds_stage::tdm::Scope::Wgp)
      : lds_stage::tdm::make_cache_policy(lds_stage::tdm::LoadTemporalHint::NonTemporal,
                                          lds_stage::tdm::Scope::System);
  constexpr int StorePolicy = is_put(Kind)
      ? lds_stage::tdm::make_cache_policy(lds_stage::tdm::StoreTemporalHint::NonTemporal,
                                          lds_stage::tdm::Scope::System)
      : lds_stage::tdm::make_cache_policy(lds_stage::tdm::StoreTemporalHint::Default,
                                          lds_stage::tdm::Scope::Wgp);

  if (size < (is_put(Kind) ? constmem.lds_stage_put_threshold : constmem.lds_stage_get_threshold)) {
    memcpy_wave<Kind>(dst, src, size);
    return;
  }
  const lds_stage::LdsRegistration reg = lds_stage::get_lds();
  // Round up: a partial trailing wavefront still needs its own slot.
  const int num_waves = max((get_flat_block_size() + WF_SIZE - 1) / WF_SIZE, 1);
  constexpr uint32_t element_bytes = 1u << lds_stage::FlatCopyElementLog2;
  // Round down to a whole element -- reg.ptr is only 8-byte aligned.
  size_t slot_bytes = reg.bytes / static_cast<size_t>(num_waves);
  slot_bytes -= slot_bytes % element_bytes;
  // Below 16 bytes there's no room for a double-buffered tile in this slot.
  if (reg.ptr != nullptr && slot_bytes >= 16) {
    uint32_t tile_bytes = static_cast<uint32_t>(
        min(static_cast<size_t>(constmem.lds_stage_tile_bytes), slot_bytes / 2));
    tile_bytes -= tile_bytes % element_bytes;
    if (tile_bytes > 0) {
      const int wave_id = get_flat_block_id() / WF_SIZE;
      void* wave_lds = static_cast<uint8_t*>(reg.ptr) +
                        static_cast<size_t>(wave_id) * slot_bytes;
      if (is_thread_zero_in_wave()) {
        if constexpr (is_put(Kind)) {
          lds_stage::tdm::copy_pipelined_shallow<LoadPolicy, StorePolicy>(
              dst, src, size, wave_lds, tile_bytes, /*double_buffered=*/true);
        } else {
          lds_stage::tdm::copy_pipelined_deep<LoadPolicy, StorePolicy>(dst, src, size, wave_lds,
                                                                        tile_bytes);
        }
      }
      const size_t covered = lds_stage::covered_bytes(size, tile_bytes);
      if (covered < size) {
        memcpy_wave<Kind>(static_cast<char*>(dst) + covered,
                          static_cast<char*>(src) + covered, size - covered);
      }
      return;
    }
  }
  memcpy_wave<Kind>(dst, src, size);
}

#elif defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)

// Every thread tiles together (striped across lanes), no single-lane
// driver; remainder runs right after, not overlapped.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wg_lds_stage(void* dst, void* src,
                                                                      size_t size) {
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
    tile_bytes -= tile_bytes % element_bytes;
    if (tile_bytes > 0) {
      // Swap to copy_pipelined_deep_asynccnt to benchmark the s_wait_asynccnt path.
      constexpr int LoadPolicy = static_cast<int>(cache_policy_aux(
           is_put(Kind) ? CachePolicy::Standard : CachePolicy::SystemScope));
       constexpr int StorePolicy = static_cast<int>(cache_policy_aux(
           is_put(Kind) ? CachePolicy::SystemScope : CachePolicy::Standard));
       lds_stage::async_lds::copy_pipelined_deep_mark<LoadPolicy, StorePolicy>(
          dst, src, size, reg.ptr, tile_bytes, static_cast<uint32_t>(get_flat_block_id()),
          static_cast<uint32_t>(get_flat_block_size()));
      const size_t covered = lds_stage::covered_bytes(size, tile_bytes);
      if (covered < size) {
        memcpy_wg<Kind, 4>(static_cast<char*>(dst) + covered,
                           static_cast<char*>(src) + covered, size - covered);
      }
      return;
    }
  }
  memcpy_wg<Kind>(dst, src, size);
}

// Wave counterpart: every lane tiles together, no single-lane driver.
// Buffer splits one fixed slot per possible wave.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wave_lds_stage(void* dst, void* src,
                                                                        size_t size) {
  if (size < (is_put(Kind) ? constmem.lds_stage_put_threshold : constmem.lds_stage_get_threshold)) {
    memcpy_wave<Kind>(dst, src, size);
    return;
  }
  const lds_stage::LdsRegistration reg = lds_stage::get_lds();
  // Round up: a partial trailing wavefront still needs its own slot.
  const int num_waves = max((get_flat_block_size() + WF_SIZE - 1) / WF_SIZE, 1);
  constexpr uint32_t element_bytes = 1u << lds_stage::FlatCopyElementLog2;
  size_t slot_bytes = reg.bytes / static_cast<size_t>(num_waves);
  slot_bytes -= slot_bytes % element_bytes;
  // Below 16 bytes there's no room for a double-buffered tile in this slot.
  if (reg.ptr != nullptr && slot_bytes >= 16) {
    uint32_t tile_bytes = static_cast<uint32_t>(
        min(static_cast<size_t>(constmem.lds_stage_tile_bytes), slot_bytes / 2));
    tile_bytes -= tile_bytes % element_bytes;
    if (tile_bytes > 0) {
      const int wave_id = get_flat_block_id() / WF_SIZE;
      void* wave_lds = static_cast<uint8_t*>(reg.ptr) +
                        static_cast<size_t>(wave_id) * slot_bytes;
      const uint32_t wave_tid = static_cast<uint32_t>(get_flat_block_id() % WF_SIZE);
      const uint32_t wave_size = static_cast<uint32_t>(wave_SZ());
      // Swap to copy_pipelined_deep_asynccnt to benchmark the s_wait_asynccnt path.
      lds_stage::async_lds::copy_pipelined_deep_mark(dst, src, size, wave_lds, tile_bytes, wave_tid,
                                                     wave_size);
      const size_t covered = lds_stage::covered_bytes(size, tile_bytes);
      if (covered < size) {
        memcpy_wave<Kind>(static_cast<char*>(dst) + covered,
                          static_cast<char*>(src) + covered, size - covered);
      }
      return;
    }
  }
  memcpy_wave<Kind>(dst, src, size);
}

#else

// Neither backend is built/usable here: plain copy.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wg_lds_stage(void* dst, void* src,
                                                                      size_t size) {
  memcpy_wg<Kind>(dst, src, size);
}

template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wave_lds_stage(void* dst, void* src,
                                                                        size_t size) {
  memcpy_wave<Kind>(dst, src, size);
}

#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_LDS_STAGE_POLICY_HPP_
