// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/request_mtype_resolver.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <format>

namespace rocjitsu {
namespace amdgpu {
namespace {

template <typename F>
uint32_t for_each_coalesced_lane_run(const uint64_t *addrs, uint64_t lane_mask, uint32_t wf_size,
                                     uint32_t stride, F &&fn) {
  uint32_t run_count = 0;
  uint64_t remaining = lane_mask;
  while (remaining) {
    const uint32_t first_lane = std::countr_zero(remaining);
    remaining &= ~(uint64_t{1} << first_lane);

    uint32_t last_lane = first_lane;
    while (last_lane + 1 < wf_size) {
      const uint32_t next_lane = last_lane + 1;
      const uint64_t next_bit = uint64_t{1} << next_lane;
      if (!(remaining & next_bit) || addrs[next_lane] != addrs[last_lane] + stride)
        break;
      remaining &= ~next_bit;
      last_lane = next_lane;
    }

    fn(first_lane, last_lane - first_lane + 1);
    ++run_count;
  }
  return run_count;
}

bool all_elements_use_lane_mask(std::span<const uint64_t> element_lane_masks, uint64_t lane_mask,
                                uint32_t num_elems) {
  if (element_lane_masks.empty())
    return true;
  assert(element_lane_masks.size() == num_elems);
  (void)num_elems;
  return std::ranges::all_of(element_lane_masks,
                             [lane_mask](uint64_t mask) { return mask == lane_mask; });
}

uint64_t fully_valid_lane_mask(std::span<const uint64_t> element_lane_masks, uint64_t lane_mask) {
  uint64_t full_lane_mask = lane_mask;
  for (uint64_t element_mask : element_lane_masks)
    full_lane_mask &= element_mask;
  return full_lane_mask;
}

} // namespace

L1VectorCache::L1VectorCache(L2Cache *l2)
    : l2_(l2), coherence_epoch_(l2_ ? l2_->coherence_domain()->current_epoch() : 0) {}

L1VectorCache::~L1VectorCache() = default;

void L1VectorCache::set_l2(L2Cache *l2) {
  invalidate_all_lines();
  l2_ = l2;
  coherence_epoch_ = l2_ ? l2_->coherence_domain()->current_epoch() : 0;
}

void L1VectorCache::set_gpu_vm(GpuVm *gpu_vm) {
  invalidate_all_lines();
  gpu_vm_ = gpu_vm;
}

VmAccessOutcome L1VectorCache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr, nullptr, vmid))
    return VmAccessOutcome::Complete;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  cache_.allocate(addr, vmid, &evicted, evicted_data);

  assert(!evicted.dirty && "L1 V$ is write-through; lines should never be dirty");

  uint8_t line_buf[LINE_SIZE];
  const VmAccessOutcome outcome = l2_->fetch_line(line_addr, line_buf, vmid);
  if (outcome != VmAccessOutcome::Complete) {
    cache_.invalidate(addr, vmid);
    return outcome;
  }
  cache_.fill_line(addr, line_buf, vmid);
  return VmAccessOutcome::Complete;
}

// Per-line CC invalidation is sufficient: the CP serializes dispatch N's cache
// management before dispatch N+1 begins execution, so no blanket invalidation
// at dispatch boundaries is needed.
VmAccessOutcome L1VectorCache::read_bytes(uint64_t addr, uint8_t *dst, uint32_t size,
                                          bool non_temporal, bool request_l1_bypass, uint32_t vmid,
                                          RequestMtypeResolver &mtypes) {
  const Mtype effective = mtypes.at(addr);

  util::Logger::cp([&](auto &os) {
    static thread_local uint64_t mtype_counts[5] = {};
    static thread_local uint64_t total = 0;
    ++mtype_counts[static_cast<int>(effective)];
    ++total;
    if ((total & (total - 1)) == 0 && total >= 1024) {
      os << std::format("L1V_READ_MTYPE_STATS total={} UC={} CC={} RW={} WB={} NT={} "
                        "last: addr={:#x} inst={} eff={} vmid={}",
                        total, mtype_counts[0], mtype_counts[1], mtype_counts[2], mtype_counts[3],
                        mtype_counts[4], addr, static_cast<int>(mtypes.fallback()),
                        static_cast<int>(effective), vmid);
    }
  });

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const Mtype chunk_mtype = mtypes.at(ea);

    if (chunk_mtype == Mtype::UC || non_temporal || request_l1_bypass) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }

    if (chunk_mtype == Mtype::CC) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }

    const VmAccessOutcome outcome = ensure_line(ea, vmid);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L1VectorCache::write_bytes(uint64_t addr, const uint8_t *src, uint32_t size,
                                           bool non_temporal, uint32_t vmid,
                                           RequestMtypeResolver &mtypes) {
  const Mtype effective = mtypes.at(addr);

  util::Logger::vm([&](auto &os) {
    if (addr >= 0x4d00c00000ULL && addr < 0x4d00c00100ULL) {
      uint32_t val = 0;
      if (size >= 4)
        std::memcpy(&val, src, 4);
      else if (size >= 2)
        std::memcpy(&val, src, size);
      else
        val = src[0];
      static thread_local uint32_t tw = 0;
      if (++tw <= 20)
        os << std::format("L1_WRITE @{:#x} size={} val={:#x} mtype={}", addr, size, val,
                          static_cast<int>(effective));
    }
  });

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const Mtype chunk_mtype = mtypes.at(ea);

    if (chunk_mtype == Mtype::UC || non_temporal) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->write(ea, src + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }

    const VmAccessOutcome fill_outcome = ensure_line(ea, vmid);
    if (fill_outcome != VmAccessOutcome::Complete)
      return fill_outcome;

    // Write through to L2 for all cacheable stores. This ensures partial writes
    // from different CUs sharing the same L2 are properly merged at byte
    // granularity via L2::write(), rather than full-line replacement via
    // writeback_line() during L1 eviction/flush.
    const VmAccessOutcome write_outcome = l2_->write(ea, src + copied, chunk, chunk_mtype, vmid);
    if (write_outcome != VmAccessOutcome::Complete)
      return write_outcome;

    cache_.write_line(ea, src + copied, line_offset, chunk, vmid);

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag, vmid);
    assert(tag != nullptr && "ensure_line must guarantee hit");

    // L1 line stays clean since L2 has the authoritative copy.
    tag->coherence = (chunk_mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED
                                                : simdojo::CoherenceState::EXCLUSIVE;
    tag->dirty = false;
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L1VectorCache::load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                                    uint32_t num_elems, uint8_t *dst, Mtype mtype,
                                    bool non_temporal, bool request_l1_bypass, uint32_t wf_size,
                                    uint32_t vmid, uint32_t addr_stride, uint32_t addr_base_offset,
                                    std::span<const uint64_t> element_lane_masks) {
  synchronize_epoch();
  RequestMtypeResolver mtypes(gpu_vm_, vmid, mtype);
  uint32_t stride = num_elems * elem_size;
  // Scratch swizzle: consecutive dwords of one lane's private space sit
  // addr_stride bytes apart, the hardware dword-interleaved layout rocm-dbgapi
  // reads. Addresses are materialized per element, so this also honours
  // per-element lane validity: an element a lane is not valid for is skipped
  // rather than strided over. With no element masks this walks exactly the
  // lanes and bytes the uniform path would. addr_base_offset identifies the
  // low bits added after swizzling so they do not move the logical dword boundary.
  if (addr_stride != 0) {
    assert(addr_base_offset < sizeof(uint32_t));
    const uint32_t astride = addr_stride;
    for (uint32_t elem = 0; elem < num_elems; ++elem) {
      uint64_t mask =
          element_lane_masks.empty() ? lane_mask : (element_lane_masks[elem] & lane_mask);
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= mask - 1;
        const uint64_t base = addrs[lane];
        const uint32_t first_byte_in_dword = static_cast<uint32_t>((base - addr_base_offset) & 3);
        uint32_t copied = elem * elem_size;
        const uint32_t elem_end = copied + elem_size;
        while (copied < elem_end) {
          const uint32_t logical_byte = first_byte_in_dword + copied;
          const uint32_t byte_in_dword = logical_byte & 3;
          const uint32_t chunk = std::min(elem_end - copied, 4 - byte_in_dword);
          const uint64_t ea =
              base - first_byte_in_dword + logical_byte / 4 * astride + byte_in_dword;
          const VmAccessOutcome outcome = read_bytes(ea, dst + lane * stride + copied, chunk,
                                                     non_temporal, request_l1_bypass, vmid, mtypes);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
          copied += chunk;
        }
      }
    }
    return VmAccessOutcome::Complete;
  }
  if (!all_elements_use_lane_mask(element_lane_masks, lane_mask, num_elems)) {
    uint64_t full_lane_mask = fully_valid_lane_mask(element_lane_masks, lane_mask);
    VmAccessOutcome outcome = VmAccessOutcome::Complete;
    for_each_coalesced_lane_run(
        addrs, full_lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
          if (outcome == VmAccessOutcome::Complete)
            outcome = read_bytes(addrs[first_lane], dst + first_lane * stride, run_lanes * stride,
                                 non_temporal, request_l1_bypass, vmid, mtypes);
        });
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    for (uint32_t elem = 0; elem < num_elems; ++elem) {
      uint64_t mask = element_lane_masks[elem] & lane_mask & ~full_lane_mask;
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= ~(uint64_t{1} << lane);
        outcome = read_bytes(addrs[lane] + static_cast<uint64_t>(elem) * elem_size,
                             dst + lane * stride + elem * elem_size, elem_size, non_temporal,
                             request_l1_bypass, vmid, mtypes);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
    }
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  for_each_coalesced_lane_run(
      addrs, lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
        if (outcome == VmAccessOutcome::Complete)
          outcome = read_bytes(addrs[first_lane], dst + first_lane * stride, run_lanes * stride,
                               non_temporal, request_l1_bypass, vmid, mtypes);
      });
  return outcome;
}

VmAccessOutcome L1VectorCache::store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                                     uint32_t num_elems, const uint8_t *src, Mtype mtype,
                                     bool non_temporal, uint32_t wf_size, uint32_t vmid,
                                     uint32_t addr_stride, uint32_t addr_base_offset,
                                     std::span<const uint64_t> element_lane_masks) {
  synchronize_epoch();
  RequestMtypeResolver mtypes(gpu_vm_, vmid, mtype);
  uint32_t stride = num_elems * elem_size;
  const uint32_t active_lanes = std::popcount(lane_mask);
  ++store_count_;
  if (active_lanes > 0)
    ++store_active_count_;
  // Scratch swizzle: consecutive dwords of one lane's private space sit
  // addr_stride bytes apart, the hardware dword-interleaved layout rocm-dbgapi
  // reads. Addresses are materialized per element, so this also honours
  // per-element lane validity: an element a lane is not valid for is skipped
  // rather than strided over. With no element masks this walks exactly the
  // lanes and bytes the uniform path would. addr_base_offset identifies the
  // low bits added after swizzling so they do not move the logical dword boundary.
  if (addr_stride != 0) {
    assert(addr_base_offset < sizeof(uint32_t));
    const uint32_t astride = addr_stride;
    for (uint32_t elem = 0; elem < num_elems; ++elem) {
      uint64_t mask =
          element_lane_masks.empty() ? lane_mask : (element_lane_masks[elem] & lane_mask);
      store_l2_writes_ += std::popcount(mask);
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= mask - 1;
        const uint64_t base = addrs[lane];
        const uint32_t first_byte_in_dword = static_cast<uint32_t>((base - addr_base_offset) & 3);
        uint32_t copied = elem * elem_size;
        const uint32_t elem_end = copied + elem_size;
        while (copied < elem_end) {
          const uint32_t logical_byte = first_byte_in_dword + copied;
          const uint32_t byte_in_dword = logical_byte & 3;
          const uint32_t chunk = std::min(elem_end - copied, 4 - byte_in_dword);
          const uint64_t ea =
              base - first_byte_in_dword + logical_byte / 4 * astride + byte_in_dword;
          const VmAccessOutcome outcome =
              write_bytes(ea, src + lane * stride + copied, chunk, non_temporal, vmid, mtypes);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
          copied += chunk;
        }
      }
    }
    return VmAccessOutcome::Complete;
  }
  if (!all_elements_use_lane_mask(element_lane_masks, lane_mask, num_elems)) {
    uint64_t full_lane_mask = fully_valid_lane_mask(element_lane_masks, lane_mask);
    VmAccessOutcome outcome = VmAccessOutcome::Complete;
    store_l2_writes_ += for_each_coalesced_lane_run(
        addrs, full_lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
          if (outcome == VmAccessOutcome::Complete)
            outcome = write_bytes(addrs[first_lane], src + first_lane * stride, run_lanes * stride,
                                  non_temporal, vmid, mtypes);
        });
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    for (uint32_t elem = 0; elem < num_elems; ++elem) {
      uint64_t mask = element_lane_masks[elem] & lane_mask & ~full_lane_mask;
      store_l2_writes_ += std::popcount(mask);
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= ~(uint64_t{1} << lane);
        outcome = write_bytes(addrs[lane] + static_cast<uint64_t>(elem) * elem_size,
                              src + lane * stride + elem * elem_size, elem_size, non_temporal, vmid,
                              mtypes);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
    }
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  store_l2_writes_ += for_each_coalesced_lane_run(
      addrs, lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
        if (outcome == VmAccessOutcome::Complete)
          outcome = write_bytes(addrs[first_lane], src + first_lane * stride, run_lanes * stride,
                                non_temporal, vmid, mtypes);
      });
  return outcome;
}

void L1VectorCache::invalidate(uint64_t addr, uint32_t vmid) {
  synchronize_epoch();
  cache_.invalidate(addr, vmid);
}

void L1VectorCache::invalidate_all() {
  synchronize_epoch();
  invalidate_all_lines();
}

void L1VectorCache::flush_all() {
  synchronize_epoch();
  invalidate_all_lines();
}

void L1VectorCache::invalidate_all_lines() { cache_.invalidate_all(); }

void L1VectorCache::synchronize_epoch() {
  assert(l2_ != nullptr && "L1 vector cache requires an L2 cache");
  const uint64_t current_epoch = l2_->coherence_domain()->current_epoch();
  if (coherence_epoch_ == current_epoch)
    return;
  invalidate_all_lines();
  coherence_epoch_ = current_epoch;
}

} // namespace amdgpu
} // namespace rocjitsu
