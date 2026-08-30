// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx1100_target_ops.cpp
/// @brief gfx1100 raw program-analysis decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

template <typename Raw> void fill_gfx1100_flat_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  site.raw_vaddr = static_cast<uint32_t>(raw.addr);
  site.raw_vsrc = static_cast<uint32_t>(raw.data);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset));
  site.raw_scope = 2u;
  site.raw_th = static_cast<uint32_t>(raw.glc) | (static_cast<uint32_t>(raw.slc) << 1u);
  site.returns_old_value = raw.glc != 0u;
}

bool decode_gfx1100_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                std::span<const uint8_t> instruction) {
  if (mnemonic.starts_with("flat_atomic") && instruction.size() >= sizeof(rdna3::FlatMachineInst)) {
    rdna3::FlatMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx1100_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("global_atomic") &&
      instruction.size() >= sizeof(rdna3::FlatGlobalMachineInst)) {
    rdna3::FlatGlobalMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx1100_flat_atomic_site(site, raw);
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_program_analysis_target_detail
