// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0.h
/// @brief gfx1250 B0-to-A0 legalization classification.

#ifndef ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_
#define ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_

namespace rocjitsu {

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

} // namespace rocjitsu

#endif // ROCJITSU_CODE_DBT_LEGALIZATION_GFX1250_B0_TO_A0_H_
