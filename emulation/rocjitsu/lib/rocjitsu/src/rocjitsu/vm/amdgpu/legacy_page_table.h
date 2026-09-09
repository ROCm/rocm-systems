// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legacy_page_table.h
/// @brief Frontend-neutral page-table data used by legacy host mappings.

#pragma once

#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rocjitsu::amdgpu {

inline constexpr uint64_t kLegacyPageShift = 12;
inline constexpr uint64_t kLegacyPageSize = uint64_t{1} << kLegacyPageShift;

/// @brief Who owns host storage named by a compatibility mapping.
enum class LegacyHostExtentOwner : uint8_t {
  Driver,
  Application,
};

/// @brief One host-backed interval within a GPU page.
class LegacyHostExtent {
public:
  uint8_t *host_ptr = nullptr;
  std::size_t host_backed_bytes = 0;
  std::size_t gpu_page_offset = 0;
  LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application;

  bool operator==(const LegacyHostExtent &) const = default;
};

/// @brief Per-page compatibility translation entry.
class LegacyPageTableEntry {
public:
  LegacyPageTableEntry() = default;
  LegacyPageTableEntry(uint8_t *host_ptr, Mtype page_mtype,
                       LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application)
      : mtype(page_mtype), host_extents{{host_ptr, kLegacyPageSize, 0, owner}} {}
  LegacyPageTableEntry(uint8_t *host_ptr, Mtype page_mtype, std::size_t host_backed_bytes,
                       std::size_t gpu_page_offset,
                       LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application)
      : mtype(page_mtype), host_extents{{host_ptr, host_backed_bytes, gpu_page_offset, owner}} {}

  Mtype mtype = Mtype::RW;
  std::vector<LegacyHostExtent> host_extents;

  bool operator==(const LegacyPageTableEntry &) const = default;
};

using LegacyPageTable = std::unordered_map<uint64_t, LegacyPageTableEntry>;

} // namespace rocjitsu::amdgpu
