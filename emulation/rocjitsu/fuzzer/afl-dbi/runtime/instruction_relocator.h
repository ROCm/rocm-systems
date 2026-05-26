#pragma once

#include "rocjitsu/code/rj_code.h"

#include <stdint.h>

#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {
class Instruction;
} // namespace rocjitsu

namespace rocjitsu::fuzzer::afl_dbi {

struct RelocatedInstruction {
  std::vector<uint32_t> words;
  std::optional<uint64_t> direct_branch_target_text_offset;
};

std::optional<RelocatedInstruction>
relocate_overwritten_instruction(std::span<const uint8_t> text, uint64_t patch_text_offset,
                                 uint32_t instruction_size, rj_code_arch_t arch,
                                 const char **failure_reason);

bool s_mov_b32_requires_state_preservation(const rocjitsu::Instruction &inst);
bool s_or_b64_requires_state_preservation(const rocjitsu::Instruction &inst);
bool scc_compare_requires_state_preservation(const rocjitsu::Instruction &inst);

} // namespace rocjitsu::fuzzer::afl_dbi
