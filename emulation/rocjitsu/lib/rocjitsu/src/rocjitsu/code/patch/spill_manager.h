// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file spill_manager.h
/// @brief Per-kernel scratch reservation for DBI spill/fill slots.

#ifndef ROCJITSU_CODE_PATCH_SPILL_MANAGER_H_
#define ROCJITSU_CODE_PATCH_SPILL_MANAGER_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"
#include "util/bit.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility> // for std::pair
#include <vector>

namespace rocjitsu {

/// @brief Monotonic byte-range allocator for one kernel's private segment.
///
/// @details This deliberately knows nothing about register identity or spill
/// lifetime. DBI layers stable register-to-slot mapping on top; semantic DBT
/// creates short-lived frames on top. Keeping the common range arithmetic next
/// to SpillManager avoids a separate one-struct allocator header while both
/// users retain their independent allocation policies.
class PrivateSegmentCursor final {
public:
  /// @brief Begin allocation at the first byte not owned by an earlier policy.
  explicit PrivateSegmentCursor(uint32_t first_byte) : cursor_(first_byte) {}

  /// @brief Compute the next aligned allocation without advancing the cursor.
  /// @param byte_count Number of contiguous bytes requested.
  /// @param alignment Required power-of-two byte alignment for the returned base.
  /// @param limit Exclusive upper bound for the allocation.
  /// @returns The aligned base, or std::nullopt for an invalid or overflowing range.
  [[nodiscard]] std::optional<uint32_t>
  preview(uint32_t byte_count, uint32_t alignment,
          uint32_t limit = std::numeric_limits<uint32_t>::max()) const {
    if (byte_count == 0 || !std::has_single_bit(alignment))
      return std::nullopt;
    const uint64_t base =
        util::align_up(static_cast<uint64_t>(cursor_), static_cast<uint64_t>(alignment));
    if (base + byte_count > limit)
      return std::nullopt;
    return static_cast<uint32_t>(base);
  }

  /// @brief Allocate the next aligned range and advance the high-water mark.
  [[nodiscard]] std::optional<uint32_t>
  allocate(uint32_t byte_count, uint32_t alignment,
           uint32_t limit = std::numeric_limits<uint32_t>::max()) {
    auto base = preview(byte_count, alignment, limit);
    if (!base)
      return std::nullopt;
    cursor_ = *base + byte_count;
    return base;
  }

  /// @brief One-past-end byte of all successfully allocated ranges.
  [[nodiscard]] uint32_t high_water_mark() const { return cursor_; }

private:
  uint32_t cursor_ = 0;
};

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
  [[nodiscard]] uint32_t total_private_bytes() const { return slots_.high_water_mark(); }

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

  uint32_t limit_;             ///< Hard per-lane scratch cap (end <= limit is valid).
  PrivateSegmentCursor slots_; ///< Shared range arithmetic; DBI owns stable slot identity.
  std::unordered_map<std::pair<RegClass, uint16_t>, uint32_t, RegKeyHash> reg_to_offset_;
};

/// @brief Standalone save/restore program for one ordinary VGPR window.
///
/// @details Each register receives one stable B32 per-lane slot. Save and
/// restore end in zero-threshold waits so callers can safely clobber the
/// window after save and resume guest code after restore. The instructions
/// run under the caller's current EXEC mask and do not modify EXEC. These
/// conservative waits also drain older wave scratch stores/loads; relaxing
/// that ordering requires a hardware-backed same-address ordering proof.
struct VgprSpillSequence {
  uint16_t vgpr_base = 0;
  uint16_t vgpr_count = 0;
  std::vector<uint32_t> slot_offsets;
  std::vector<uint32_t> save_words;
  std::vector<uint32_t> restore_words;
  uint32_t total_private_bytes = 0;
  bool uses_dynamic_stack_frame = false;
  uint16_t dynamic_frame_base_sgpr = 0;
  uint32_t dynamic_frame_bytes = 0;

  [[nodiscard]] bool has_complete_slot_metadata() const {
    return slot_offsets.size() == vgpr_count;
  }
};

/// Dynamic-stack ABI state established by AMDGPU callable functions.
///
/// These are architectural ABI assignments, not instrumentation-reserved
/// scratch registers. A site-local frame must preserve their incoming values.
inline constexpr uint16_t kDynamicStackTopSgpr = 32u;
inline constexpr uint16_t kDynamicFrameBaseSgpr = 33u;

/// @brief Dynamic-stack spill program bootstrapped without a dead SGPR.
///
/// @details The complete VGPR window is first saved relative to the incoming
/// runtime stack top. Four VGPRs from that now-preserved window then hold the
/// incoming borrowed SGPR pair, frame base, and SCC while the temporary frame
/// is active. This breaks the full-register-pressure dependency cycle without
/// assigning an instrumentation-specific preserved SGPR range. The borrowed
/// window is intentionally one pair because the current consumer uses it to
/// preserve the complete 64-bit VCC value; a wider scalar spill is a distinct
/// resource plan rather than an implicit extension of this contract.
struct DynamicStackBorrowedSgprSpillSequence {
  static constexpr uint16_t kScalarReservoirCount = 4u;

  uint16_t vgpr_base = 0;
  uint16_t vgpr_count = 0;
  uint16_t borrowed_sgpr_base = 0;
  uint16_t scalar_reservoir_vgpr_base = 0;
  std::vector<uint32_t> slot_offsets;
  std::vector<uint32_t> save_words;
  std::vector<uint32_t> restore_words;
  uint32_t total_private_bytes = 0;
  uint32_t dynamic_frame_bytes = 0;

  [[nodiscard]] VgprSpillSequence as_vgpr_spill_sequence() const {
    return {
        .vgpr_base = vgpr_base,
        .vgpr_count = vgpr_count,
        .slot_offsets = slot_offsets,
        .save_words = save_words,
        .restore_words = restore_words,
        .total_private_bytes = total_private_bytes,
        .uses_dynamic_stack_frame = true,
        .dynamic_frame_base_sgpr = kDynamicFrameBaseSgpr,
        .dynamic_frame_bytes = dynamic_frame_bytes,
    };
  }
};

/// @brief Standalone save/restore program for one ordinary SGPR window.
///
/// @details A caller-provided VGPR transfers each wave-uniform scalar value
/// through private memory. The transfer VGPR must already have been preserved
/// when the sequence executes.
struct SgprSpillSequence {
  uint16_t sgpr_base = 0;
  uint16_t sgpr_count = 0;
  std::vector<uint32_t> save_words;
  std::vector<uint32_t> restore_words;
  uint32_t total_private_bytes = 0;
};

/// @brief Reserve slots and encode a CDNA3, CDNA4, RDNA4, or gfx1250 VGPR
/// spill/fill sequence.
///
/// @returns A complete sequence, or nullopt for an unsupported architecture,
/// invalid VGPR range, unencodable slot, or capacity failure. On failure
/// @p manager is unchanged.
[[nodiscard]] std::optional<VgprSpillSequence> build_vgpr_spill_sequence(SpillManager &manager,
                                                                         uint16_t vgpr_base,
                                                                         uint16_t vgpr_count,
                                                                         rj_code_arch_t arch);

/// @brief Encode a site-local dynamic-stack frame around one VGPR spill window.
///
/// @details Follows the AMDGPU callable-function convention used by dynamic
/// stack kernels: save the current frame base, set it to the stack top,
/// advance the top by the temporary frame size, then reverse those operations
/// after filling the VGPRs. The SCC value is restored before guest code runs.
/// The encoding is available for CDNA3, CDNA4, RDNA3, RDNA4, and gfx1250. Callers
/// remain responsible for requiring enough descriptor/AQL backing for the
/// compiler maximum stack depth plus this temporary frame.
[[nodiscard]] std::optional<VgprSpillSequence>
build_dynamic_stack_vgpr_spill_sequence(uint16_t vgpr_base, uint16_t vgpr_count,
                                        uint16_t stack_top_sgpr, uint16_t frame_base_sgpr,
                                        uint16_t saved_frame_base_sgpr, uint16_t saved_scc_sgpr,
                                        rj_code_arch_t arch, uint32_t additional_frame_bytes = 0);

/// @brief Encode an RDNA4-family dynamic-stack frame when both register files
/// are live.
///
/// @param borrowed_sgpr_base Live ordinary SGPR pair borrowed by the caller.
/// @param scalar_reservoir_vgpr_base First of four consecutive VGPRs inside
///        the spilled window used to preserve the pair, frame base, and SCC.
/// @param original_private_bytes Compiler-declared maximum private backing
///        before the instrumentation frame is added.
/// @param additional_frame_bytes Extra frame storage reserved after the VGPR
///        slots for a caller-owned scalar spill layout.
///
/// @details Save leaves the borrowed pair available to the caller and restore
/// expects any special state saved in that pair (for example VCC) to have
/// already been restored. total_private_bytes is the compiler maximum plus the
/// frame for descriptor validation; dynamic_frame_bytes is the addend applied
/// above the runtime-selected AQL private depth.
[[nodiscard]] std::optional<DynamicStackBorrowedSgprSpillSequence>
build_dynamic_stack_borrowed_sgpr_spill_sequence(uint16_t vgpr_base, uint16_t vgpr_count,
                                                 uint16_t borrowed_sgpr_base,
                                                 uint16_t scalar_reservoir_vgpr_base,
                                                 uint32_t original_private_bytes,
                                                 rj_code_arch_t arch,
                                                 uint32_t additional_frame_bytes = 0);

/// @brief Save an SGPR window in an already-established dynamic-stack frame.
///
/// @details Slots begin at @p frame_byte_offset relative to the frame base
/// recorded in @p vgpr_frame. The VGPR save must establish a dynamic frame
/// containing the scalar slots and preserve @p transfer_vgpr; this sequence
/// validates that contract before using the transfer register.
[[nodiscard]] std::optional<SgprSpillSequence>
build_dynamic_stack_sgpr_spill_sequence(uint16_t sgpr_base, uint16_t sgpr_count,
                                        uint16_t transfer_vgpr, const VgprSpillSequence &vgpr_frame,
                                        uint32_t frame_byte_offset, rj_code_arch_t arch);

enum class SpillDescriptorUpdate : uint8_t {
  Updated,
  Unchanged,
  InvalidDescriptor,
  InvalidPrivateSize,
};

/// @brief Grow one kernel descriptor's fixed private segment for spill slots.
///
/// @details Sets ENABLE_PRIVATE_SEGMENT when needed and preserves all other
/// descriptor fields. Dynamic-stack dispatch backing is a separate runtime
/// contract because the launch's stack depth is not fixed in the descriptor.
[[nodiscard]] SpillDescriptorUpdate
update_kernel_descriptor_for_spills(std::span<uint8_t> image, uint64_t descriptor_file_offset,
                                    uint32_t required_private_bytes);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_PATCH_SPILL_MANAGER_H_
