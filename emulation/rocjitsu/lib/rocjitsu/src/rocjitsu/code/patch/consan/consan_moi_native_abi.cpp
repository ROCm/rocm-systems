// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"

#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan_moi_impl {

std::optional<uint16_t> gfx1250_vgpr_msb_mode_at(std::span<const uint8_t> bytes,
                                                 uint64_t text_file_offset,
                                                 uint64_t container_entry_text_offset,
                                                 uint64_t site_file_offset) {
  return consan_gfx1250_vgpr_msb_mode_at(bytes, text_file_offset, container_entry_text_offset,
                                         site_file_offset);
}

bool append_moi_flat_load_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}

bool append_moi_global_atomic_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr && consan_detail::append_moi_global_atomic_completion(words, *target);
}

bool append_moi_lds_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_lds0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}

} // namespace rocjitsu::consan_moi_impl
