// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instruction_builder.h"

#include "rocjitsu/isa/instruction.h"

#include <limits>
#include <optional>

namespace rocjitsu {

namespace {

enum class ScalarSop2Op {
  AddU32,
  SubU32,
  AddcU32,
  SubbU32,
};

/// @brief Return the SOP2 opcode number for scalar arithmetic on @p arch.
///
/// @details These builders are used for patch-time code emission, so the
/// architecture-specific opcode mapping lives with the instruction packers
/// instead of the static recovery matcher.
[[nodiscard]] std::optional<uint8_t> scalar_sop2_opcode(rj_code_arch_t arch, ScalarSop2Op op) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    switch (op) {
    case ScalarSop2Op::AddU32:
      return 0;
    case ScalarSop2Op::SubU32:
      return 1;
    case ScalarSop2Op::AddcU32:
      return 4;
    case ScalarSop2Op::SubbU32:
      return 5;
    }
    return std::nullopt;
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Build an encoded scalar `s_add_u32` instruction.
///
/// @details Used only for final static-PC recovery fixup emission, after the
/// recovery pass has proven that the old address-recovery region can be
/// overwritten. The operands are already scalar operand encoding values.
[[nodiscard]] std::optional<uint32_t> build_s_add_u32(rj_code_arch_t arch, uint16_t sdst,
                                                      uint16_t ssrc0, uint16_t ssrc1) {
  auto opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddU32);
  if (!opcode)
    return std::nullopt;
  return pack_sop2(*opcode, sdst, ssrc0, ssrc1);
}

/// @brief Build an encoded scalar `s_sub_u32` instruction.
/// @copydetails build_s_add_u32
[[nodiscard]] std::optional<uint32_t> build_s_sub_u32(rj_code_arch_t arch, uint16_t sdst,
                                                      uint16_t ssrc0, uint16_t ssrc1) {
  auto opcode = scalar_sop2_opcode(arch, ScalarSop2Op::SubU32);
  if (!opcode)
    return std::nullopt;
  return pack_sop2(*opcode, sdst, ssrc0, ssrc1);
}

/// @brief Build an encoded scalar `s_addc_u32` instruction.
/// @copydetails build_s_add_u32
[[nodiscard]] std::optional<uint32_t> build_s_addc_u32(rj_code_arch_t arch, uint16_t sdst,
                                                       uint16_t ssrc0, uint16_t ssrc1) {
  auto opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddcU32);
  if (!opcode)
    return std::nullopt;
  return pack_sop2(*opcode, sdst, ssrc0, ssrc1);
}

/// @brief Build an encoded scalar `s_subb_u32` instruction.
/// @copydetails build_s_add_u32
[[nodiscard]] std::optional<uint32_t> build_s_subb_u32(rj_code_arch_t arch, uint16_t sdst,
                                                       uint16_t ssrc0, uint16_t ssrc1) {
  auto opcode = scalar_sop2_opcode(arch, ScalarSop2Op::SubbU32);
  if (!opcode)
    return std::nullopt;
  return pack_sop2(*opcode, sdst, ssrc0, ssrc1);
}

} // namespace

bool patch_pcrel_branch_offset(const Instruction &inst, std::span<uint32_t> words,
                               int64_t delta_bytes, [[maybe_unused]] rj_code_arch_t arch) {
  if ((inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_CALL)) == 0)
    return false;
  if (!inst.branch_offset_bytes())
    return false;
  if (words.empty())
    return false;
  if (delta_bytes % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return false;

  const int64_t delta_dwords = delta_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (delta_dwords < std::numeric_limits<int16_t>::min() ||
      delta_dwords > std::numeric_limits<int16_t>::max())
    return false;

  // AMDGPU SOPP direct branches and SOPK s_call_b64 encode a signed 16-bit
  // dword offset from the next instruction. Preserve the already-translated
  // opcode and operand bits, replacing only that simm16 field.
  words[0] = (words[0] & 0xFFFF0000u) | static_cast<uint16_t>(static_cast<int16_t>(delta_dwords));
  return true;
}

bool append_pc_delta_builder(std::vector<uint32_t> &words, rj_code_arch_t arch, uint16_t pc_sreg,
                             int64_t delta) {
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;
  const bool negative = delta < 0;
  const uint64_t magnitude =
      negative ? (~static_cast<uint64_t>(delta) + 1u) : static_cast<uint64_t>(delta);
  const uint32_t lo = static_cast<uint32_t>(magnitude);
  const uint32_t hi = static_cast<uint32_t>(magnitude >> 32);
  const uint16_t pc_hi = static_cast<uint16_t>(pc_sreg + 1);

  // The recovered region is proven to contain only address-recovery code, so
  // final fixup can replace it with a canonical builder. The low half always
  // uses a literal; the high half uses inline zero when possible to keep the
  // replacement smaller than the original builder sequence.
  if (negative) {
    auto sub = build_s_sub_u32(arch, pc_sreg, pc_sreg, kLiteralOperand);
    auto subb = build_s_subb_u32(arch, pc_hi, pc_hi, hi == 0 ? kInlineInt0 : kLiteralOperand);
    if (!sub || !subb)
      return false;
    words.push_back(*sub);
    words.push_back(lo);
    words.push_back(*subb);
  } else {
    auto add = build_s_add_u32(arch, pc_sreg, pc_sreg, kLiteralOperand);
    auto addc = build_s_addc_u32(arch, pc_hi, pc_hi, hi == 0 ? kInlineInt0 : kLiteralOperand);
    if (!add || !addc)
      return false;
    words.push_back(*add);
    words.push_back(lo);
    words.push_back(*addc);
  }
  if (hi != 0)
    words.push_back(hi);
  return true;
}

} // namespace rocjitsu
