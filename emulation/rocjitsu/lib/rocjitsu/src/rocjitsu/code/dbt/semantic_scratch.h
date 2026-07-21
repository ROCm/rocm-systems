// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_scratch.h
/// @brief Architecture-neutral scratch allocation for semantic lowerings.

#pragma once

#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu {

class Instruction;
class LivenessAnalysis;
struct TranslationContext;

/// @brief Why a semantic scratch allocation could not be satisfied.
enum class SemanticScratchFailure : uint8_t {
  None,
  InvalidRequest,
  NoRegisterWindow,
  SpillOffsetUnencodable,
};

/// @brief Contiguous per-lane private-memory slots used by one spill payload.
struct SemanticSpillRange {
  uint32_t byte_offset = 0;
  uint16_t dword_count = 0;

  /// @brief Byte offset encoded by the final dword access in this range.
  [[nodiscard]] uint32_t last_dword_offset() const {
    return dword_count == 0
               ? byte_offset
               : byte_offset + (static_cast<uint32_t>(dword_count) - 1u) * sizeof(uint32_t);
  }
};

/// @brief Register window selected for one semantic replacement sequence.
///
/// @details A dead or descriptor-grown window needs no preservation. A spilled
/// window borrows guest registers and names the private-memory range where the
/// target-specific emitter must save and restore their values.
struct SemanticScratchLease {
  RegClass reg_class = RegClass::VGPR;
  uint16_t base = 0;
  uint16_t count = 0;
  bool spilled = false;
  uint32_t spill_offset = 0;

  [[nodiscard]] RegisterRef registers() const {
    return RegisterRef{reg_class, base, static_cast<uint8_t>(count)};
  }
};

/// @brief Target register-file and spill-address constraints used by allocation.
///
/// @details These are data constraints, not instruction encodings. A target
/// scratch emitter publishes the policy appropriate for its spill/fill form.
struct SemanticScratchPolicy {
  uint16_t max_vgprs = static_cast<uint16_t>(REGISTER_SET_MAX_VGPRS);
  uint32_t max_spill_dword_offset = std::numeric_limits<uint32_t>::max();
};

/// @brief Request for one contiguous temporary VGPR window.
struct SemanticScratchRequest {
  uint16_t count = 0;
  uint16_t alignment = 1;
  RegisterSet forbidden;
  bool allow_spill = true;
  std::optional<uint16_t> preferred_victim_base;
};

/// @brief Typed result of a semantic scratch allocation.
struct SemanticScratchResult {
  std::optional<SemanticScratchLease> lease;
  SemanticScratchFailure failure = SemanticScratchFailure::None;

  [[nodiscard]] explicit operator bool() const { return lease.has_value(); }
};

/// @brief One reusable private-memory spill frame for a replacement sequence.
///
/// @details Each semantic instruction begins at the same transient frame base,
/// after persistent kernel-wide semantic storage. Allocations within the frame
/// advance a cursor so simultaneously-live spill payloads cannot overlap. The
/// kernel descriptor records only the largest frame high-water mark.
class SemanticSpillFrame final {
public:
  explicit SemanticSpillFrame(TranslationContext &context);

  /// @brief Allocate contiguous dword slots with byte alignment.
  /// @param max_dword_offset Largest encodable byte offset for an individual
  ///        dword spill/fill instruction on the target architecture.
  [[nodiscard]] std::optional<SemanticSpillRange>
  allocate_dwords(uint16_t dword_count, uint32_t byte_alignment,
                  uint32_t max_dword_offset = std::numeric_limits<uint32_t>::max());

private:
  TranslationContext &context_;
  PrivateSegmentCursor cursor_;
};

/// @brief Architecture-neutral register and spill-slot allocator for one rule.
class SemanticScratchAllocator final {
public:
  SemanticScratchAllocator(const Instruction &inst, const LivenessAnalysis &liveness,
                           TranslationContext &context, SemanticScratchPolicy policy);

  /// @brief Prefer dead VGPRs, then borrow and spill an allowed guest window.
  [[nodiscard]] SemanticScratchResult acquire_vgprs(const SemanticScratchRequest &request);

  /// @brief Reserve additional non-overlapping spill state in this rule's frame.
  [[nodiscard]] std::optional<SemanticSpillRange>
  allocate_spill_dwords(uint16_t dword_count, uint32_t byte_alignment = sizeof(uint32_t));

private:
  [[nodiscard]] bool window_is_allowed(uint16_t base, const SemanticScratchRequest &request,
                                       uint16_t available_count) const;

  const Instruction &inst_;
  const LivenessAnalysis &liveness_;
  TranslationContext &context_;
  SemanticScratchPolicy policy_;
  SemanticSpillFrame spill_frame_;
};

} // namespace rocjitsu
