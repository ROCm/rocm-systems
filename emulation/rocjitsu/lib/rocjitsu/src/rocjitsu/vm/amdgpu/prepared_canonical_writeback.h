// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file prepared_canonical_writeback.h
/// @brief Prepared, bounded canonical backing effects for cache maintenance.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/fabric_backing.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {
class SimulatedKfd;
}

namespace rocjitsu::amdgpu {

class DeviceCacheCoherence;

/// @brief Retain one admitted alias group for direct cache writeback.
/// @details KFD prepares every backing and VM identity before cache locks are
/// taken. The group coordinator resolves each dirty line against its resident
/// tag and retains every counted ingress permit until all cache locks leave.
/// KFD entries also carry direct byte views held alive by their backing lease;
/// a batch may copy multiple runs only through those views after acquiring all
/// VM gates. write() remains the bounded, single-effect fallback for a generic
/// FabricBacking owner whose virtual write can fail.
class PreparedCanonicalWriteback {
public:
  PreparedCanonicalWriteback(const PreparedCanonicalWriteback &) = delete;
  PreparedCanonicalWriteback &operator=(const PreparedCanonicalWriteback &) = delete;
  PreparedCanonicalWriteback(PreparedCanonicalWriteback &&other) noexcept
      : outcome_(std::exchange(other.outcome_, VmAccessOutcome::Unavailable)),
        entries_(std::move(other.entries_)), permit_(std::move(other.permit_)),
        effect_gate_(std::move(other.effect_gate_)) {}
  PreparedCanonicalWriteback &operator=(PreparedCanonicalWriteback &&other) noexcept {
    if (this != &other) {
      entries_ = std::move(other.entries_);
      permit_ = std::move(other.permit_);
      effect_gate_ = std::move(other.effect_gate_);
      outcome_ = std::exchange(other.outcome_, VmAccessOutcome::Unavailable);
    }
    return *this;
  }
  ~PreparedCanonicalWriteback() = default;

  [[nodiscard]] VmAccessOutcome outcome() const noexcept {
    return outcome_ == VmAccessOutcome::Complete && !*this ? VmAccessOutcome::Unavailable
                                                           : outcome_;
  }
  [[nodiscard]] explicit operator bool() const noexcept {
    return outcome_ == VmAccessOutcome::Complete && !entries_.empty() && permit_ && effect_gate_;
  }

  /// @brief Check that one resident tag has a complete, unambiguous backing.
  /// @details This is only a precheck. write() still enters the retained VM
  /// gate; callers must preserve dirty state if that entry is later refused.
  [[nodiscard]] bool covers(uint64_t address, size_t size, uint32_t vmid, uint64_t cache_identity,
                            uint64_t admission_generation) const noexcept {
    return find_entry(address, size, vmid, cache_identity, admission_generation) != nullptr;
  }

  /// @brief Commit one complete dirty run directly to its retained backing.
  /// @details The only operation inside the VM gate is the bounded backing
  /// write. Refused gate entry returns Revoked; missing identity, coverage, or
  /// a failed backing write returns Unavailable without changing dirty state.
  [[nodiscard]] VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes,
                                      uint32_t vmid, uint64_t cache_identity,
                                      uint64_t admission_generation) const noexcept {
    const Entry *entry =
        find_entry(address, bytes.size(), vmid, cache_identity, admission_generation);
    if (entry == nullptr || !effect_gate_)
      return VmAccessOutcome::Unavailable;

    const uint64_t backing_offset = entry->backing_offset_ + (address - entry->address_);
    const FabricBacking view = entry->backing_.subview(backing_offset, bytes.size());
    if (!view)
      return VmAccessOutcome::Unavailable;

    bool written = false;
    const bool entered = effect_gate_->commit([&] { written = view.write(bytes); });
    if (!entered)
      return VmAccessOutcome::Revoked;
    return written ? VmAccessOutcome::Complete : VmAccessOutcome::Unavailable;
  }

private:
  friend class ::rocjitsu::SimulatedKfd;
  friend class DeviceCacheCoherence;
  friend class PreparedCanonicalWritebackTestAccess;

  struct ResolvedLine {
    uint64_t backing_id;
    uint64_t backing_offset;
    FabricBacking backing;
    std::span<std::byte> direct_bytes;
  };

  class Entry {
  public:
    Entry(uint64_t address, size_t extent, uint32_t vmid, VmCacheNamespace cache_namespace,
          uint64_t cache_identity, uint64_t admission_generation, uint64_t backing_id,
          uint64_t backing_offset, FabricBacking backing, std::span<std::byte> direct_bytes = {})
        : address_(address), extent_(extent), vmid_(vmid), cache_namespace_(cache_namespace),
          cache_identity_(cache_identity), admission_generation_(admission_generation),
          backing_id_(backing_id), backing_offset_(backing_offset), backing_(std::move(backing)),
          direct_bytes_(direct_bytes) {}

  private:
    friend class PreparedCanonicalWriteback;

    [[nodiscard]] bool valid() const noexcept {
      return extent_ != 0 && extent_ - 1 <= std::numeric_limits<uint64_t>::max() - address_ &&
             vmid_ != 0 && static_cast<bool>(cache_namespace_) && cache_identity_ != 0 &&
             admission_generation_ != 0 && backing_id_ != 0 && backing_ &&
             backing_.backing_id() == backing_id_ && backing_.offset() == backing_offset_ &&
             backing_.extent() == extent_ &&
             (direct_bytes_.empty() || direct_bytes_.size() == extent_);
    }

    [[nodiscard]] bool covers(uint64_t address, size_t size, uint32_t vmid, uint64_t cache_identity,
                              uint64_t admission_generation) const noexcept {
      if (size == 0 || vmid_ != vmid || cache_identity_ != cache_identity ||
          admission_generation_ != admission_generation || address < address_)
        return false;
      const uint64_t relative = address - address_;
      return relative <= extent_ && size <= extent_ - relative;
    }

    uint64_t address_;
    size_t extent_;
    uint32_t vmid_;
    VmCacheNamespace cache_namespace_;
    uint64_t cache_identity_;
    uint64_t admission_generation_;
    uint64_t backing_id_;
    uint64_t backing_offset_;
    FabricBacking backing_;
    // KFD's retained GpuBackingLease owns these bytes through backing_. A
    // generic FabricBacking owner leaves this empty and remains fallible.
    std::span<std::byte> direct_bytes_;
  };

  explicit PreparedCanonicalWriteback(VmAccessOutcome outcome) : outcome_(outcome) {}

  PreparedCanonicalWriteback(std::vector<Entry> entries, MemoryIngressPermit permit,
                             std::shared_ptr<VmTransferEffectGate> effect_gate)
      : entries_(std::move(entries)), permit_(std::move(permit)),
        effect_gate_(std::move(effect_gate)) {
    if (!permit_ || !effect_gate_ || entries_.empty())
      return;
    for (const Entry &entry : entries_)
      if (!entry.valid())
        return;
    outcome_ = VmAccessOutcome::Complete;
  }

  [[nodiscard]] const Entry *find_entry(uint64_t address, size_t size, uint32_t vmid,
                                        uint64_t cache_identity,
                                        uint64_t admission_generation) const noexcept {
    if (!*this)
      return nullptr;
    const Entry *match = nullptr;
    for (const Entry &entry : entries_) {
      if (!entry.covers(address, size, vmid, cache_identity, admission_generation))
        continue;
      if (match != nullptr)
        return nullptr;
      match = &entry;
    }
    return match;
  }

  [[nodiscard]] std::optional<ResolvedLine>
  resolve_line(uint64_t address, size_t size, uint32_t vmid, uint64_t cache_identity,
               uint64_t admission_generation) const noexcept {
    const Entry *entry = find_entry(address, size, vmid, cache_identity, admission_generation);
    if (entry == nullptr)
      return std::nullopt;
    const size_t relative = static_cast<size_t>(address - entry->address_);
    const uint64_t offset = entry->backing_offset_ + relative;
    FabricBacking backing = entry->backing_.subview(offset, size);
    if (!backing)
      return std::nullopt;
    const std::span<std::byte> direct_bytes = entry->direct_bytes_.empty()
                                                  ? std::span<std::byte>{}
                                                  : entry->direct_bytes_.subspan(relative, size);
    return ResolvedLine{entry->backing_id_, offset, std::move(backing), direct_bytes};
  }

  VmAccessOutcome outcome_ = VmAccessOutcome::Unavailable;
  std::vector<Entry> entries_;
  MemoryIngressPermit permit_;
  std::shared_ptr<VmTransferEffectGate> effect_gate_;
};

} // namespace rocjitsu::amdgpu
