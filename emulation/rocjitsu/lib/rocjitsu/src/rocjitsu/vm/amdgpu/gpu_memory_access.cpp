// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"

#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#include <cstring>
#include <limits>

namespace rocjitsu::amdgpu {
namespace {

bool valid_range(uint64_t address, std::size_t size) {
  return size == 0 || size - 1 <= std::numeric_limits<uint64_t>::max() - address;
}

bool valid_atomic(VmMemoryDomain domain, uint64_t address, uint32_t width) {
  return domain == VmMemoryDomain::Local &&
         (width == sizeof(uint32_t) || width == sizeof(uint64_t)) && address % width == 0 &&
         valid_range(address, width);
}

} // namespace

VmAccessOutcome GpuMemoryPhysicalAccess::read(VmMemoryDomain domain, uint64_t address,
                                              std::span<std::byte> bytes) {
  if (domain != VmMemoryDomain::Local || !valid_range(address, bytes.size()))
    return VmAccessOutcome::Malformed;
  memory_->read_block(
      address, {reinterpret_cast<uint8_t *>(bytes.data()), static_cast<std::size_t>(bytes.size())});
  return VmAccessOutcome::Complete;
}

VmAccessOutcome GpuMemoryPhysicalAccess::write(VmMemoryDomain domain, uint64_t address,
                                               std::span<const std::byte> bytes) {
  if (domain != VmMemoryDomain::Local || !valid_range(address, bytes.size()))
    return VmAccessOutcome::Malformed;
  memory_->write_block(address, {reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()});
  return VmAccessOutcome::Complete;
}

AtomicLoadResult GpuMemoryPhysicalAccess::atomic_load(VmMemoryDomain domain, uint64_t address,
                                                      uint32_t width) {
  if (!valid_atomic(domain, address, width))
    return {.outcome = VmAccessOutcome::Malformed};

  uint64_t value = 0;
  const bool loaded = memory_->atomic_modify(
      address, width, [&](const uint8_t *bytes) { std::memcpy(&value, bytes, width); });
  return {.outcome = loaded ? VmAccessOutcome::Complete : VmAccessOutcome::Malformed,
          .value = value};
}

VmAccessOutcome GpuMemoryPhysicalAccess::atomic_store(VmMemoryDomain domain, uint64_t address,
                                                      uint32_t width, uint64_t value) {
  if (!valid_atomic(domain, address, width))
    return VmAccessOutcome::Malformed;
  const bool stored = memory_->atomic_modify(
      address, width, [&](uint8_t *bytes) { std::memcpy(bytes, &value, width); });
  return stored ? VmAccessOutcome::Complete : VmAccessOutcome::Malformed;
}

AtomicCompareExchangeResult
GpuMemoryPhysicalAccess::compare_exchange(VmMemoryDomain domain, uint64_t address, uint32_t width,
                                          uint64_t expected, uint64_t desired) {
  if (!valid_atomic(domain, address, width))
    return {.outcome = VmAccessOutcome::Malformed};

  uint64_t observed = 0;
  bool exchanged = false;
  const bool compared = memory_->atomic_modify(address, width, [&](uint8_t *bytes) {
    std::memcpy(&observed, bytes, width);
    const uint64_t mask = width == sizeof(uint32_t) ? std::numeric_limits<uint32_t>::max()
                                                    : std::numeric_limits<uint64_t>::max();
    if (observed == (expected & mask)) {
      std::memcpy(bytes, &desired, width);
      exchanged = true;
    }
  });
  return {.outcome = compared ? VmAccessOutcome::Complete : VmAccessOutcome::Malformed,
          .observed = observed,
          .exchanged = exchanged};
}

VmAccessOutcome
GpuMemoryPhysicalAccess::atomic_modify(VmMemoryDomain domain, uint64_t address, uint32_t width,
                                       const PhysicalMemoryAccess::AtomicMutation &mutation) {
  if (!valid_atomic(domain, address, width) || !mutation)
    return VmAccessOutcome::Malformed;
  const bool modified = memory_->atomic_modify(address, width, [&](uint8_t *bytes) {
    mutation({reinterpret_cast<std::byte *>(bytes), width});
  });
  return modified ? VmAccessOutcome::Complete : VmAccessOutcome::Malformed;
}

} // namespace rocjitsu::amdgpu
