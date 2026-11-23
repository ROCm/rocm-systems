// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vmem_direct.h
/// @brief Plain local vector loads and stores straight against host backing.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm_access.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_transaction.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::amdgpu {

class L1VectorCache;
class L2Cache;
class VmDirectBacking;

/// @brief Whether direct vector access is enabled for this process.
/// @details RJ_VMEM_DIRECT=0 keeps every request on the cache model so the
/// two paths can be compared inside one binary.
[[nodiscard]] bool vmem_direct_enabled();

/// One plain vector request whose bounding box local ingress already admitted.
struct DirectVectorRequest {
  const uint64_t *addrs = nullptr;
  uint64_t lanes = 0;
  uint32_t wf_size = 0;
  uint32_t elem_size = 0;
  uint32_t num_elems = 0;
  Mtype instruction_mtype = Mtype::RW;
  MemoryTransactionPolicy transactions;
  /// Lane-major data, lane * elem_size * num_elems: the load destination or store source.
  uint8_t *data = nullptr;
  bool is_load = true;
};

/// @brief Complete a request against local host backing without the cache
/// model, or decline before any effect.
/// @details Every coalesced transaction must lie in one cacheable local
/// extent. Pages other than driver-sealed RAM are copied under the host access
/// guard; a fault declines after a partial transfer, which the routed path then
/// repeats from the start, as plain loads and stores are idempotent. Loads need no unpublished L2
/// bytes: functional caches write through, so backing is never older than a resident copy. A store
/// also needs this CU's L1 and the L2 to hold no copy of its bytes; a fill that races the write is
/// discarded afterwards. Each transaction keeps its backing stripe locks, so other threads observe
/// the same transaction atomicity.
[[nodiscard]] std::optional<VmAccessOutcome>
try_direct_vector_access(const DirectVectorRequest &request, const VmDirectBacking &backing,
                         const L1VectorCache &l1, L2Cache &l2);

/// @brief Complete the swizzled (dword- or 16-byte-interleaved private) lanes
/// of @p state against host backing, or decline before any effect.
/// @details Spans are visited exactly as the cache model visits them. Spans
/// that are consecutive in memory with equal size and a constant payload
/// stride, such as one element of every lane of an interleaved scratch
/// access, move as one run. The same extent, MTYPE, residency, permit,
/// admission and fault rules as try_direct_vector_access() apply.
[[nodiscard]] std::optional<VmAccessOutcome>
try_direct_swizzled_access(const VectorMemState &state, uint64_t lanes, uint8_t *data,
                           const VmDirectBacking &backing, const L1VectorCache &l1, L2Cache &l2);

} // namespace rocjitsu::amdgpu
