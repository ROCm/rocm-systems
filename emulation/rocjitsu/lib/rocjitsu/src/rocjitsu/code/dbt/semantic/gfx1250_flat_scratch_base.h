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
/// a 64-bit source position can name these selectors. The selector is matched
/// against the encoding field, not the decoded operand value: a literal source
/// reports its own value there, so the constants 230 and 231 would otherwise be
/// mistaken for the selectors they collide with.
[[nodiscard]] bool gfx1250_reads_flat_scratch_base_64bit(const Instruction &inst);

/// @brief True when a final A0 instruction still needs the 64-bit
/// FLAT_SCRATCH_BASE operand rewrite.
///
/// @details Scalar A0 reads are canonical with selector 230, while selector
/// 231 remains actionable. Vector ALU source fields cannot retain either
/// selector and must name the ordinary SGPR pair produced by the lowering.
[[nodiscard]] bool gfx1250_flat_scratch_base_residual(const Instruction &inst);

/// @brief Rewrite a 64-bit FLAT_SCRATCH_BASE source for the A0 profile.
///
/// @details Scalar reads keep their instruction and change the selector to 230.
/// A 64-bit vector source cannot name the selector directly. The preferred
/// replacement moves the base into a dead SGPR pair. When none is available,
/// an explicitly opted-in EXEC-masked VALU encoding may instead use two 32-bit
/// vector moves to stage the halves in the instruction's overwritten two-VGPR
/// destination. Destination staging requires no explicit or implicit read of
/// that pair and known, matching source/destination VGPR-MSB banks. One chosen
/// temporary serves every affected source position. The replacement can grow
/// by one scalar move, or by two vector moves plus dependency padding.
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
