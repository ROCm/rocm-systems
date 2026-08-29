// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan_moi_detail {

bool append_store_u32_literal(std::vector<uint32_t> &words, uint64_t address, uint32_t value,
                              uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_value = instrumentation::build_v_mov_b32_literal(value_vgpr, value, arch);
  const auto store = instrumentation::build_flat_store_b32(address_lo_vgpr, value_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, mov_value, store);
}

bool append_store_u32_vgpr(std::vector<uint32_t> &words, uint64_t address, uint16_t value_vgpr,
                           uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto store = instrumentation::build_flat_store_b32(address_lo_vgpr, value_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, store);
}

bool append_store_u32_sgpr(std::vector<uint32_t> &words, uint64_t address, uint16_t value_sgpr,
                           uint16_t value_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  words.push_back(build_v_mov_b32_e32(value_vgpr, value_sgpr, arch));
  return append_store_u32_vgpr(words, address, value_vgpr, scratch_vgpr, arch);
}

bool append_store_u32_vgpr_at_offset(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                     uint32_t byte_offset, uint16_t value_vgpr,
                                     rj_code_arch_t arch) {
  const auto store =
      instrumentation::build_flat_store_b32(address_vgpr, value_vgpr, arch, byte_offset);
  InstructionSequence sequence(words);
  return sequence.emit(store);
}

bool append_load_u32_vgpr_at_offset(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                    uint32_t byte_offset, uint16_t destination_vgpr,
                                    rj_code_arch_t arch) {
  const auto load =
      instrumentation::build_flat_load_b32(address_vgpr, destination_vgpr, arch, byte_offset);
  const auto wait = instrumentation::build_s_wait_global_load0(arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(load, wait);
}

} // namespace rocjitsu::consan_moi_detail
