// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vm_direct_backing.h
/// @brief Host access to whole local extents for requests that bypass the caches.

#pragma once

#include "rocjitsu/vm/amdgpu/legacy_page_table.h"
#include "rocjitsu/vm/amdgpu/raw_backing_write_gate.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// @brief Translate and copy local memory of one binding without a cache model.
/// @details Resolve every extent first; extent() may take page-table locks and
/// must never run inside run_admitted(). run_admitted() holds one copied-policy
/// admission and a host-mapping lease across @p fn, so the resolved host
/// pointers stay mapped and current; copy() is valid only inside it.
class VmDirectBacking {
public:
  [[nodiscard]] virtual uint64_t generation() const = 0;
  [[nodiscard]] virtual std::optional<LegacyLocalExtent> extent(uint64_t address,
                                                                uint64_t generation) const = 0;
  /// Returns false, without calling @p fn, when @p generation is no longer current.
  [[nodiscard]] virtual bool run_admitted(uint64_t generation, void (*fn)(void *),
                                          void *context) const = 0;
  /// Excludes controlled writeback of [address, address + size) while held.
  [[nodiscard]] virtual std::optional<VmRawBackingWriteGate::Permit>
  begin_write(uint64_t address, std::size_t size) const = 0;
  /// @brief Copy one span within a resolved extent under the backing transaction locks.
  /// @details Application-owned pages may be unmapped or protected by the
  /// application at any time, so @p guarded copies turn a host fault into a
  /// false return; bytes before the fault may already have moved.
  [[nodiscard]] virtual bool copy(uint64_t address, uint8_t *host, void *local, std::size_t size,
                                  bool into_memory, bool guarded) const = 0;

protected:
  ~VmDirectBacking() = default;
};

} // namespace rocjitsu::amdgpu
