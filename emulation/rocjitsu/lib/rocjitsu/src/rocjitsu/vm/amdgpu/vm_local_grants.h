// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm_access.h"

namespace rocjitsu::amdgpu {

/// Uniform local allocation permissions published by one frontend binding.
class VmLocalGrants {
public:
  VmLocalGrants(uint64_t identity, std::shared_ptr<const std::atomic<uint64_t>> publication,
                std::function<VmLocalGrant(uint64_t, size_t)> lookup)
      : identity_(identity), publication_(std::move(publication)), lookup_(std::move(lookup)) {}

  const std::atomic<uint64_t> *publication() const { return publication_.get(); }

  std::optional<uint64_t> generation(uint64_t address, size_t size) const {
    if (const auto *grant = find(address, size))
      return grant->generation;
    return std::nullopt;
  }

  const VmLocalGrant *find(uint64_t address, size_t size) const {
    if (!publication_ || !size)
      return nullptr;
    const uint64_t epoch = publication_->load(std::memory_order_acquire);
    if (epoch == UINT64_MAX)
      return nullptr;
    struct Entry {
      uint64_t identity = 0, epoch = 0;
      VmLocalGrant grant;
    };
    static thread_local Entry last{};
    const auto covers = [&](const Entry &entry) {
      return entry.identity == identity_ && entry.epoch == epoch && address >= entry.grant.begin &&
             address < entry.grant.end && size <= entry.grant.end - address;
    };
    // Adjacent cache levels usually check subspans of the same allocation.
    if (covers(last))
      return &last.grant;
    static thread_local std::array<Entry, 8> entries{};
    for (const auto &entry : entries) {
      if (covers(entry)) {
        last = entry;
        return &last.grant;
      }
    }
    if (!lookup_)
      return nullptr;
    const auto grant = lookup_(address, size);
    if (!grant.end || address < grant.begin || address >= grant.end || size > grant.end - address)
      return nullptr;
    static thread_local unsigned next = 0;
    last = entries[next++ % entries.size()] = {identity_, epoch, grant};
    return &last.grant;
  }

private:
  const uint64_t identity_;
  const std::shared_ptr<const std::atomic<uint64_t>> publication_;
  const std::function<VmLocalGrant(uint64_t, size_t)> lookup_;
};

} // namespace rocjitsu::amdgpu
