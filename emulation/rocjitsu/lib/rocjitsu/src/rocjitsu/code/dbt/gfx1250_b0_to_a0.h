// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0.h
/// @brief Initial gfx1250 B0-to-A0 errata legalization classification.

#ifndef ROCJITSU_CODE_DBT_GFX1250_B0_TO_A0_H_
#define ROCJITSU_CODE_DBT_GFX1250_B0_TO_A0_H_

namespace rocjitsu {

class Instruction;
struct InstructionLegalization;

/// @brief Classify instructions that require a gfx1250 B0-to-A0 workaround.
///
/// @details B0 and A0 use the same architectural instruction encodings, so
/// unaffected instructions need no legalization entry and can be copied
/// verbatim. A non-null result deliberately reports `Action::Expand` for a
/// known errata candidate. The first offline-translation milestone has no
/// expansion implementations; callers therefore fail closed with an
/// `ExpandMissing` diagnostic instead of silently emitting B0 behavior on A0.
///
/// Some workarounds are conditional on operands or whole-kernel context. This
/// initial classifier is intentionally conservative and marks the complete
/// mnemonic family. The eventual semantic rule must inspect the precise
/// predicate before changing code.
[[nodiscard]] const InstructionLegalization *gfx1250_b0_to_a0_legalization(const Instruction &inst);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_DBT_GFX1250_B0_TO_A0_H_
