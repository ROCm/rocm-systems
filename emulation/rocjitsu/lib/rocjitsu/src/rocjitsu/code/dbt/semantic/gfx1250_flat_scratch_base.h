// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_flat_scratch_base.h
/// @brief gfx1250 B0-to-A0 lowering for FLAT_SCRATCH_BASE source operands.

#ifndef ROCJITSU_CODE_DBT_SEMANTIC_GFX1250_FLAT_SCRATCH_BASE_H_
#define ROCJITSU_CODE_DBT_SEMANTIC_GFX1250_FLAT_SCRATCH_BASE_H_

#include "rocjitsu/code/dbt/translation_rule.h"

#include <cstdint>
#include <span>

namespace rocjitsu {

class Instruction;
class LivenessAnalysis;

/// @brief True when @p inst reads FLAT_SCRATCH_BASE through a 64-bit source
/// position that the A0 profile encodes differently.
///
/// @details Selector 230 (`SRC_FLAT_SCRATCH_BASE_LO`) and selector 231
/// (`SRC_FLAT_SCRATCH_BASE_HI`) both deliver the whole 64-bit base value when
/// they appear in a source position of that width. A0 accepts only the 230
/// spelling for scalar reads, and takes the value through an ordinary SGPR pair
/// for vector reads, so both cases need an operand rewrite.
///
/// Detection is operand-driven rather than opcode-driven: any instruction with
/// a 64-bit source position can name these selectors.
[[nodiscard]] bool gfx1250_reads_flat_scratch_base_64bit(const Instruction &inst);

/// @brief Rewrite a 64-bit FLAT_SCRATCH_BASE source for the A0 profile.
///
/// @details Scalar reads keep their instruction and change the selector to 230.
/// Vector reads cannot name the selector at all, so the base is first moved
/// into a dead SGPR pair with a 64-bit scalar read and the source position is
/// repointed at that pair; one pair serves every affected position in the
/// instruction.
///
/// @param inst        The decoded instruction.
/// @param offset      Byte offset of @p inst in @p source_text.
/// @param source_text Full source .text bytes, authoritative for literal and
///                    modifier words that follow the base encoding.
/// @param liveness    Kernel-scoped live-before data used to find a dead pair.
/// @param context     Kernel translation context, accepted to match the rule
///                    signature. A borrowed pair needs no descriptor change on
///                    this target: its scalar file is fixed rather than sized by
///                    the descriptor, so only the addressable range constrains
///                    the choice.
/// @returns Success with replacement words, NotHandled when @p inst reads no
///          such operand, or Failed when the rewrite cannot be encoded.
[[nodiscard]] ExpandResult gfx1250_lower_flat_scratch_base_source(
    const Instruction &inst, uint64_t offset, std::span<const uint8_t> source_text,
    const LivenessAnalysis &liveness, TranslationContext &context);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_DBT_SEMANTIC_GFX1250_FLAT_SCRATCH_BASE_H_
