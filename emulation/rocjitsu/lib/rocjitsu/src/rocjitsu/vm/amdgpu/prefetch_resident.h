// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <array>

namespace rocjitsu::amdgpu {

/// Copy an admitted L2 line into a private cache, preserving resident bytes.
/// The caller holds the L1 maintenance locks. A null access identifies physical memory.
template <class Store>
void prefetch_resident_line(Store &cache, L2Cache *l2, uint64_t address, const GpuVmAccess *access,
                            bool warm_l2) {
  if (!l2)
    return;
  const uint64_t line = Store::line_address(address);
  uint32_t vmid = 0;
  uint64_t identity = 0;
  uint64_t generation = 0;
  if (access) {
    if (!access->info().legacy_cache_compatible)
      return;
    auto admission = access->admit_access(line, Store::LINE_SIZE, VmAccessKind::Read, false);
    if (!admission || !admission.cacheable ||
        access->query_access(line, Store::LINE_SIZE, VmAccessKind::Read) !=
            VmAccessOutcome::Complete)
      return;
    auto policy = access->query_mtype(line);
    if (!policy || (*policy != Mtype::RW && *policy != Mtype::WB))
      return;
    vmid = access->info().vmid;
    identity = access->cache_identity();
    generation = admission.generation;
  }
  const auto *resident = cache.peek(line, vmid);
  if (resident &&
      (resident->dirty || resident->admission_generation != generation ||
       resident->cache_identity != identity || Store::bytes_valid(*resident, 0, Store::LINE_SIZE)))
    return;
  if (warm_l2)
    l2->prefetch_line(line, access);
  std::array<uint8_t, Store::LINE_SIZE> bytes;
  if (!l2->try_read_peer_cached(line, bytes, vmid, generation, identity))
    return;
  if (resident) {
    cache.fill_missing_bytes(line, bytes.data(), vmid);
    return;
  }
  auto allocated = cache.allocate_clean_victim_with_data(line, vmid);
  if (!allocated.tag)
    return;
  allocated.tag->admission_generation = generation;
  allocated.tag->cache_identity = identity;
  cache.fill_line(line, bytes.data(), vmid);
}
} // namespace rocjitsu::amdgpu
