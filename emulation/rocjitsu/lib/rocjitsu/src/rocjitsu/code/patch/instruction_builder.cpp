// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instruction_builder.h"

#include "rocjitsu/isa/instruction.h"

#include <limits>

namespace rocjitsu {

bool patch_pcrel_branch_offset(const Instruction &inst, std::span<uint32_t> words,
                               int64_t delta_bytes, [[maybe_unused]] rj_code_arch_t arch) {
  if ((inst.flags() & (BRANCH | COND_BRANCH)) == 0)
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

  // Generated AMDGPU direct branches expose only SOPP branch immediates through
  // branch_offset_bytes() today. Preserve the already-translated opcode bits and
  // replace just the signed simm16 field so this utility covers s_branch and the
  // s_cbranch_* family without depending on per-ISA opcode numbers.
  words[0] = (words[0] & 0xFFFF0000u) | static_cast<uint16_t>(static_cast<int16_t>(delta_dwords));
  return true;
}

} // namespace rocjitsu
