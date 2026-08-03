// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0.h
/// @brief gfx1250 B0-to-A0 legalization classification.

#ifndef ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_
#define ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class Gfx1250VgprMsbAnalysis;
class Instruction;
struct InstructionLegalization;

/// @brief Classify instructions handled by the gfx1250 B0-to-A0 profile.
///
/// @details B0 and A0 use the same architectural instruction encodings, so
/// instructions outside the translation profile need no legalization entry and can be copied
/// verbatim. A non-null result deliberately reports `Action::Expand` for a
/// translation candidate. The semantic translator then selects the matching
/// handwritten expansion rule; a classified instruction without a matching
/// rule fails closed.
///
/// Some translation rules are conditional on operands or whole-kernel context. This
/// classifier may therefore recognize a complete mnemonic family while the
/// corresponding semantic rule inspects the precise operand predicate before
/// changing code.
[[nodiscard]] const InstructionLegalization *gfx1250_b0_to_a0_legalization(const Instruction &inst);

/// @brief True when a low-precision B0 WMMA needs an A0 completion wait.
///
/// @details This conservative classification covers every dense FP8/BF8,
/// F8F6F4, and FP4 WMMA form that executes through the affected low-precision
/// path after translation. It is intentionally separate from legalization
/// because some of these instructions otherwise retain their original encoding.
[[nodiscard]] bool gfx1250_b0_to_a0_requires_wmma_completion_wait(const Instruction &inst);

/// @brief Append an A0 completion wait when the source stream does not already
///        provide one for this low-precision WMMA.
///
/// @details A trailing VA_VDST=0 wait may cover consecutive independent WMMA
/// producers, including the two halves of an M=32 expansion. It cannot be
/// credited across a later WMMA that reads an earlier pending destination.
/// Only these read-after-write overlaps block credit; overlapping destination
/// ranges remain creditable because they do not consume a pending result.
/// Generated VGPR-bank transition instructions are transparent, and physical
/// VGPR dependencies are resolved with @p vgpr_msb.
///
/// @pre The caller has selected the gfx1250 B0-to-A0 translation profile.
void gfx1250_b0_to_a0_append_wmma_completion_wait_if_needed(
    const Instruction &inst,
    const std::unordered_map<uint64_t, const Instruction *> &source_instruction_by_offset,
    const Gfx1250VgprMsbAnalysis &vgpr_msb, std::vector<uint32_t> &words);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_
