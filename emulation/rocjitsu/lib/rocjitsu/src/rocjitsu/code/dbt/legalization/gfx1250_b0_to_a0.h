// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0.h
/// @brief gfx1250 B0-to-A0 errata legalization classification.

#ifndef ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_
#define ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_

namespace rocjitsu {

class Instruction;
struct InstructionLegalization;

/// @brief Classify instructions that require a gfx1250 B0-to-A0 workaround.
///
/// @details B0 and A0 use the same architectural instruction encodings, so
/// unaffected instructions need no legalization entry and can be copied
/// verbatim. A non-null result deliberately reports `Action::Expand` for a
/// known errata candidate. The semantic translator then selects the matching
/// handwritten expansion rule; a classified instruction without a matching
/// rule fails closed instead of silently retaining B0 behavior on A0.
///
/// Some workarounds are conditional on operands or whole-kernel context. This
/// classifier may therefore recognize a complete mnemonic family while the
/// corresponding semantic rule inspects the precise operand predicate before
/// changing code.
[[nodiscard]] const InstructionLegalization *gfx1250_b0_to_a0_legalization(const Instruction &inst);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_
