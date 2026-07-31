// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_rewrite_hazards.h
/// @brief Short instruction-timing windows visible at gfx1250 rewrite sites.

#pragma once

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/isa/register_set.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {

class BasicBlock;
class Instruction;

/// @brief Whether one source instruction can open a supported gfx1250 rewrite
/// timing window.
[[nodiscard]] bool gfx1250_is_rewrite_hazard_producer(const Instruction &inst);

/// @brief Whether a scope contains a source instruction that can open a
/// gfx1250 rewrite timing window.
///
/// @details The full analysis is unnecessary without a TRANS, WMMA, or SWMMAC
/// producer: copied instructions cannot violate one of the supported rules,
/// and a generated ordinary VALU has no active source window to overlap.
[[nodiscard]] bool gfx1250_scope_has_rewrite_hazard_producer(KernelBlockScope blocks);

/// @brief Timing constraints for an ordinary VALU inserted before one source instruction.
///
/// @details Each set at index N contains registers whose use as a generated
/// destination would require N leading V_NOPs. The union is a soft scratch
/// allocation exclusion: selecting another dead range avoids padding, while a
/// high-pressure fallback can still select an overlapping range and emit the
/// exact maximum separation returned by required_nops().
struct Gfx1250GeneratedValuHazard {
  static constexpr uint8_t kMaxNops = 8;

  std::array<RegisterSet, kMaxNops + 1> defs_requiring_nops;
  std::array<RegisterSet, kMaxNops + 1> uses_requiring_nops;

  /// @brief Logical VGPRs to avoid for destinations generated in one physical bank.
  ///
  /// @details Timing analysis tracks the complete 1024-entry physical register
  /// file. SemanticScratchAllocator uses logical indices within one 256-VGPR
  /// bank, so this query selects physical accesses from @p bank and projects
  /// them into the allocator's index space. It considers destination conflicts
  /// only; current callers emit a write before any read of the selected lease.
  [[nodiscard]] RegisterSet avoid_destination_vgprs_in_bank(uint8_t bank) const;

  /// @brief Leading V_NOPs needed before generated VALU accesses.
  [[nodiscard]] uint8_t required_nops(const RegisterSet &defs, const RegisterSet &uses = {}) const;
};

enum class Gfx1250InstructionHazardKind : uint8_t {
  TransCoexecution,
  WmmaCoexecution,
};

/// @brief One insufficient instruction-level separation in decoded gfx1250 code.
struct Gfx1250InstructionHazard {
  Gfx1250InstructionHazardKind kind = Gfx1250InstructionHazardKind::TransCoexecution;
  uint64_t producer_offset = 0;
  uint64_t consumer_offset = 0;
  uint8_t required_slots = 0;
  uint8_t existing_valu_slots = 0;

  [[nodiscard]] uint8_t missing_nops() const {
    return required_slots > existing_valu_slots ? required_slots - existing_valu_slots : 0;
  }
};

/// @brief Bounded CFG-aware recognizer for supported hazards at semantic rewrites.
///
/// @details The generated-VALU query models the gfx1250 portions of LLVM's
/// GCNHazardRecognizer that can be newly triggered when a lowering inserts an
/// ordinary vector VALU using allocator-selected VGPRs:
///
/// - WMMA/SWMMAC co-execution windows, including format-dependent 4/8 and 2/4
///   WMMA-to-VALU distances.
/// - A non-TRANS VALU immediately following a TRANS instruction.
///
/// Only eight preceding VALU slots can matter to that query, so it walks a
/// finite, memoized predecessor window rather than materializing whole-kernel
/// history. The forward decoded-stream verifier also supports WMMA-to-WMMA,
/// whose IU8 form can require nine slots.
///
/// This is deliberately not full LLVM hazard parity. The supported decoded
/// stream directions are TRANS-to-ordinary-vector-VALU and
/// WMMA/SWMMAC-to-ordinary-vector-VALU-or-WMMA/SWMMAC. Mixed TRANS/WMMA
/// directions, pseudo-scalar or SGPR TRANS overlaps, and unrelated gfx1250
/// instruction hazards remain outside this analysis.
class Gfx1250RewriteHazardAnalysis {
public:
  Gfx1250RewriteHazardAnalysis(KernelBlockScope blocks, BasicBlock *entry,
                               std::span<const ScopedCfgEdge> extra_edges = {},
                               std::span<const uint8_t> text = {});
  ~Gfx1250RewriteHazardAnalysis();

  Gfx1250RewriteHazardAnalysis(const Gfx1250RewriteHazardAnalysis &) = delete;
  Gfx1250RewriteHazardAnalysis &operator=(const Gfx1250RewriteHazardAnalysis &) = delete;
  Gfx1250RewriteHazardAnalysis(Gfx1250RewriteHazardAnalysis &&) noexcept;
  Gfx1250RewriteHazardAnalysis &operator=(Gfx1250RewriteHazardAnalysis &&) noexcept;

  /// @brief Return constraints for an ordinary VALU inserted immediately before @p inst.
  [[nodiscard]] Gfx1250GeneratedValuHazard before_generated_valu(const Instruction &inst) const;

  /// @brief Verify the supported TRANS/WMMA timing subset in a decoded stream.
  [[nodiscard]] std::vector<Gfx1250InstructionHazard> find_instruction_hazards() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu
