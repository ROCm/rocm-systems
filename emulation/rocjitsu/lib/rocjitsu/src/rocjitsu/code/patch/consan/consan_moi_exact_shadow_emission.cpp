// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan_moi_impl {

bool append_add_shifted_vgpr_field(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                   uint16_t field_vgpr, uint16_t shift, uint32_t mask,
                                   uint16_t temporary_vgpr, rj_code_arch_t arch) {
  const auto mask_words =
      instrumentation::build_v_and_b32_literal(temporary_vgpr, mask, field_vgpr, arch);
  const auto shift_word = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(shift), temporary_vgpr, arch);
  const auto add_word = instrumentation::build_v_add_u32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), temporary_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mask_words, shift_word, add_word);
}

bool append_add_literal_field(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                              uint32_t value, uint16_t temporary_vgpr, rj_code_arch_t arch) {
  if (value == 0)
    return true;
  const auto mov_value = instrumentation::build_v_mov_b32_literal(temporary_vgpr, value, arch);
  const auto add_word = instrumentation::build_v_add_u32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), temporary_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_value, add_word);
}

bool append_extract_exact_shadow_field(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                       uint16_t packed_vgpr, uint16_t shift, uint32_t mask,
                                       rj_code_arch_t arch) {
  const auto shifted = instrumentation::build_v_lshrrev_b32(
      destination_vgpr, scalar_positive_inline_u32(shift), packed_vgpr, arch);
  const auto masked =
      instrumentation::build_v_and_b32_literal(destination_vgpr, mask, destination_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(shifted, masked);
}

} // namespace rocjitsu::consan_moi_impl
