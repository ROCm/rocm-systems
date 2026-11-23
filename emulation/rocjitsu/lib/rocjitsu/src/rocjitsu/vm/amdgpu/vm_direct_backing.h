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
#include <span>

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
  /// One span within a resolved extent and its value bytes.
  struct Copy {
    uint64_t address = 0;
    uint8_t *host = nullptr;
    void *local = nullptr;
    std::size_t size = 0;
  };
  /// @brief Copy spans within resolved extents, in order, under the backing
  /// transaction locks.
  /// @details Pages other than driver-sealed RAM may be unmapped or protected
  /// by the application at any time, so @p guarded copies turn a host fault
  /// into a false return; bytes before the fault may already have moved.
  [[nodiscard]] virtual bool copy(std::span<const Copy> copies, bool into_memory,
                                  bool guarded) const = 0;

protected:
  ~VmDirectBacking() = default;
};

} // namespace rocjitsu::amdgpu
