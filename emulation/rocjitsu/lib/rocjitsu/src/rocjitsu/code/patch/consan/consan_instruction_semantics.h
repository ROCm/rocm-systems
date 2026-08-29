// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_PATCH_CONSAN_INSTRUCTION_SEMANTICS_H
#define ROCJITSU_CODE_PATCH_CONSAN_INSTRUCTION_SEMANTICS_H

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <memory>
#include <span>

namespace rocjitsu {

class Decoder;
class Instruction;

// Target encoding sentinels used while normalizing FLAT/VGLOBAL operands.
inline constexpr uint32_t kCdnaGlobalNoSaddrEncoding = 0x7fu;
inline constexpr uint32_t kRdna3FlatNoSaddrEncoding = 0x7cu;
inline constexpr uint32_t kRdna3GlobalNoSaddrEncoding = 0x7cu;

/// Interpret the low 13 bits as a signed VGLOBAL/SCRATCH displacement.
[[nodiscard]] constexpr int32_t sign_extend_13_bit_offset(uint32_t value) {
  return static_cast<int32_t>(value << 19u) >> 19u;
}

[[nodiscard]] std::unique_ptr<Instruction>
decode_bounded_instruction(Decoder &decoder, std::span<const uint32_t> words,
                           uint64_t source_offset);

[[nodiscard]] int32_t sign_extend_24(uint32_t value);

[[nodiscard]] bool is_barrier_instruction(const Instruction &instruction);

[[nodiscard]] ConSanBarrierSite::Scope barrier_scope_for_id(int32_t barrier_id);

void decode_barrier_operand(const Instruction &instruction,
                            std::span<const uint8_t> instruction_bytes, ConSanBarrierSite &site);

[[nodiscard]] bool is_s_clause(const Instruction &instruction);

[[nodiscard]] uint32_t s_clause_following_instruction_count(const Instruction &instruction);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_PATCH_CONSAN_INSTRUCTION_SEMANTICS_H
