// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file classify.h
/// @brief Deciding what an instruction is, at the granularity timing charges.
///
/// @details The decoder knows an encoding, an opcode and a mnemonic; it does
/// not know which unit an instruction occupies or what it costs. Turning one
/// into the other is a timing decision, so it lives here rather than in the ISA
/// layer, where it would have to be regenerated per target and would bind the
/// decoder to one performance model.
///
/// The ladder is ordered by how much evidence each step carries, strongest
/// first: the `s_` and `v_` prefixes name the unit outright, the family
/// prefixes name a memory path, and the decoder's own flags are the last
/// resort for an opcode whose name nobody has seen. Only when all three say
/// nothing does an instruction become InstClass::Unknown, which is the
/// expensive class rather than a free one, so a coverage gap makes a run read
/// slow and look suspicious instead of reading fast and looking like an answer.

#pragma once

#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <cstdint>
#include <string_view>

namespace rocjitsu {
class Instruction;
} // namespace rocjitsu

namespace rocjitsu::timing {

/// @brief The class of @p inst, or InstClass::Unknown when nothing identifies
///        it.
InstClass classify(const rocjitsu::Instruction &inst);

/// @brief Re-decide a memory class now that the addresses are known.
///
/// @details A FLAT access is named identically whether it reaches memory or the
/// local data share, and only the aperture its addresses landed in decides
/// which. Everything else is already on the only path it can take.
InstClass refine_with_memory_space(InstClass initial, bool is_local, bool is_load);

/// @brief Multiply-accumulates one matrix instruction performs, from its name.
/// @returns Zero when the mnemonic carries no shape.
///
/// @details A matrix opcode carries its shape: `v_mfma_f32_32x32x8_f16` is a
/// 32x32x8 product. Costing every matrix instruction the same number is wrong
/// by the ratio of the largest shape to the smallest, which is eightfold inside
/// one ISA and measurable end to end.
std::uint64_t matrix_macs(std::string_view mnemonic);

/// @brief Which matrix-core rate an opcode's inputs select.
///
/// @details The input type is the *tail* of the mnemonic:
/// `v_mfma_f32_16x16x16_f16` accumulates in single precision from
/// half-precision inputs, and it is the inputs that set the rate. Reading the
/// leading type instead gives every shape the accumulator's rate, which is the
/// same mistake as having one rate: it made a single-workgroup FP32 GEMM read
/// four times too fast.
MatrixType matrix_type_of(std::string_view mnemonic);

} // namespace rocjitsu::timing
