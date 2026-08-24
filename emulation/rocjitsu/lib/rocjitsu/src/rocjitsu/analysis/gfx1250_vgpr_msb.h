// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_vgpr_msb.h
/// @brief Forward CFG analysis for gfx1250 WAVE_MODE.VGPR_MSB state.

#pragma once

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace rocjitsu {

class Instruction;

/// @brief Resolve gfx1250 encoded VGPR operands to their 256-register bank.
///
/// @details gfx1250 stores only the low eight bits of a VGPR index in vector
/// instructions. S_SET_VGPR_MSB and MODE register writes provide two high bits
/// independently for DST, SRC0, SRC1, and SRC2. This analysis propagates those
/// four fields through a kernel-local CFG. A field is known at a join only when
/// every reachable predecessor agrees; otherwise bank_before() returns
/// std::nullopt so clients can behave conservatively.
class Gfx1250VgprMsbAnalysis {
public:
  /// @param text Raw .text image, used to read S_SETREG_IMM32_B32 literals safely
  ///        at src_loc()+4. Empty is tolerated: such writes then mark the affected
  ///        banks ambiguous rather than reading a literal.
  /// @param additional_entries Further blocks that are entered with the same
  ///        architectural state as @p entry rather than through a CFG edge. A
  ///        device function whose address is only ever taken has no decoded edge
  ///        into it, so without this its blocks stay unreachable and every bank
  ///        query over them answers nullopt. See the note on entry_state().
  Gfx1250VgprMsbAnalysis(KernelBlockScope blocks, BasicBlock *entry,
                         std::span<const ScopedCfgEdge> extra_edges = {},
                         std::span<const uint8_t> text = {},
                         std::span<BasicBlock *const> additional_entries = {},
                         bool setreg_vgpr_msb_fixup = true);
  ~Gfx1250VgprMsbAnalysis();

  Gfx1250VgprMsbAnalysis(const Gfx1250VgprMsbAnalysis &) = delete;
  Gfx1250VgprMsbAnalysis &operator=(const Gfx1250VgprMsbAnalysis &) = delete;
  Gfx1250VgprMsbAnalysis(Gfx1250VgprMsbAnalysis &&) noexcept;
  Gfx1250VgprMsbAnalysis &operator=(Gfx1250VgprMsbAnalysis &&) noexcept;

  /// @brief Bank selected for an operand role immediately before @p inst.
  /// @returns 0..3 when proven, or nullopt for unreachable/ambiguous state.
  [[nodiscard]] std::optional<uint8_t> bank_before(const Instruction &inst,
                                                   amdgpu::VgprMsbRole role) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu
