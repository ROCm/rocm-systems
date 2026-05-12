// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file spill_manager.h
/// @brief Per-kernel scratch reservation for DBI spill/fill slots.

#ifndef ROCJITSU_CODE_PATCH_SPILL_MANAGER_H_
#define ROCJITSU_CODE_PATCH_SPILL_MANAGER_H_

#include "rocjitsu/isa/register_set.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility> // for std::pair

namespace rocjitsu {

/// @brief Per-kernel scratch reservation for DBI spill/fill slots.
///
/// Appends a "DBI spill zone" to the kernel's existing per-lane scratch
/// segment and hands out stable byte offsets within per-lane scratch for
/// each spilled register. Allocation is idempotent: spilling the same
/// register twice returns the same offset.
class SpillManager final {
public:
  /// @brief Slot unit in bytes. Every spilled register occupies one slot per
  ///        32-bit lane (a 64-bit pair gets two consecutive slots).
  static constexpr uint32_t kSlotBytes = 4;

  /// @brief DBI spill zone is appended at the next multiple of this alignment
  ///        above the kernel's existing private_segment_fixed_size.
  static constexpr uint32_t kDbiZoneAlignment = 16;

  /// @param original_private_bytes  Existing private_segment_fixed_size from
  ///                                the kernel descriptor.
  /// @param per_lane_scratch_limit  Hard cap (bytes) for the bumped per-lane
  ///                                scratch. Allocations that would push
  ///                                total_private_bytes() past this cap fail.
  /// @note If @p original_private_bytes (rounded up to @c kDbiZoneAlignment)
  ///       already exceeds @p per_lane_scratch_limit, every allocation will
  ///       fail; the manager is constructible but unusable.
  SpillManager(uint32_t original_private_bytes, uint32_t per_lane_scratch_limit);

  /// @brief Allocate a single 32-bit slot for @p reg.
  /// @returns Byte offset within per-lane scratch of the slot for @p reg, or
  ///          nullopt if allocation would exceed @c per_lane_scratch_limit, or
  ///          if @p reg.cls is not tracked by RegisterSet, or if @p reg.index
  ///          is past the per-class hardware bound.
  ///          Idempotent for the same @p reg (cache hit returns the existing
  ///          offset even when the limit is otherwise exhausted).
  /// @note The @c width field of @p reg is IGNORED — only @c (cls, index)
  ///       determines the slot key. Use @c allocate_slots for multi-lane refs.
  [[nodiscard]] std::optional<uint32_t> allocate_slot(RegisterRef reg);

  /// @brief Allocate @p width consecutive 32-bit slots starting at @p reg.
  /// @param width Number of consecutive 32-bit slots; must be >= 1. Width 0
  ///              returns nullopt.
  /// @returns Byte offset of the first slot, or nullopt on overflow or if
  ///          @p reg.index + @p width would exceed UINT16_MAX.
  /// @note width=2 is the common case (SGPR pair, 64-bit VGPR pair).
  ///       On failure the manager is unchanged: the underlying @c reserve
  ///       call performs an upfront capacity check before mutating, so a
  ///       partially-completed allocation never becomes visible.
  [[nodiscard]] std::optional<uint32_t> allocate_slots(RegisterRef reg, unsigned width);

  /// @brief Allocate slots for every register in @p set.
  /// @returns true on success. On failure the manager is unchanged — the
  ///          capacity check runs upfront across all new registers, so no
  ///          partial commit is possible.
  /// @note An empty set or a set whose registers are all already cached
  ///       returns true even on an over-limit manager — no new bytes are
  ///       requested, so no capacity check fires.
  [[nodiscard]] bool reserve(const RegisterSet &set);

  /// @returns The bumped total per-lane scratch bytes.
  [[nodiscard]] uint32_t total_private_bytes() const { return total_bytes_; }

  /// @returns Slot offset previously allocated for @p reg, or nullopt.
  [[nodiscard]] std::optional<uint32_t> offset_for(RegisterRef reg) const;

private:
  /// Hash for (RegClass, register-index). RegClass fits in 8 bits and the
  /// index in 16, so the combined key is collision-free in 32 bits.
  struct RegKeyHash {
    size_t operator()(const std::pair<RegClass, uint16_t> &k) const noexcept {
      return (static_cast<size_t>(k.first) << 16) | k.second;
    }
  };

  uint32_t base_offset_; ///< First DBI slot. align_up(orig, 16).
  uint32_t total_bytes_; ///< Bumped private_segment_fixed_size.
  uint32_t limit_;       ///< Hard per-lane scratch cap (inclusive: offset+kSlotBytes <= limit OK).
  uint32_t next_offset_; ///< Next free byte within DBI zone.
  std::unordered_map<std::pair<RegClass, uint16_t>, uint32_t, RegKeyHash> reg_to_offset_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_PATCH_SPILL_MANAGER_H_
