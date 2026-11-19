// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_descriptor_rsrc3.h"

#include "rocjitsu/code/kernel_descriptor_abi.h"

#include <cassert>

namespace rocjitsu {

Rsrc3Layout rsrc3_layout(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return Rsrc3Layout::Reserved;
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return Rsrc3Layout::Gfx9Accum;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return Rsrc3Layout::Gfx10;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return Rsrc3Layout::Gfx11;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return Rsrc3Layout::Gfx120;
  case ROCJITSU_CODE_ARCH_CDNA5:
    return Rsrc3Layout::Gfx125;
  default:
    return Rsrc3Layout::Incompatible;
  }
}

bool rsrc3_layout_has_shared_vgpr_count(Rsrc3Layout layout) {
  return layout == Rsrc3Layout::Gfx10 || layout == Rsrc3Layout::Gfx11;
}

std::optional<uint32_t> inst_pref_size(uint32_t rsrc3, rj_code_arch_t arch) {
  switch (rsrc3_layout(arch)) {
  // The ABI header names GFX11's 6-bit field GFX10_PLUS; GFX10 has no INST_PREF_SIZE.
  case Rsrc3Layout::Gfx11:
    return RJ_AMDHSA_BITS_GET(rsrc3, amdhsa::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE);
  case Rsrc3Layout::Gfx120:
  case Rsrc3Layout::Gfx125:
    return RJ_AMDHSA_BITS_GET(rsrc3, amdhsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE);
  default:
    return std::nullopt;
  }
}

void set_inst_pref_size(uint32_t &rsrc3, rj_code_arch_t arch, uint32_t lines) {
  switch (rsrc3_layout(arch)) {
  case Rsrc3Layout::Gfx11:
    assert(lines <= (amdhsa::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE >>
                     amdhsa::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE_SHIFT) &&
           "INST_PREF_SIZE does not fit GFX11's 6-bit field");
    RJ_AMDHSA_BITS_SET(rsrc3, amdhsa::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE, lines);
    return;
  case Rsrc3Layout::Gfx120:
  case Rsrc3Layout::Gfx125:
    assert(lines <= (amdhsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE >>
                     amdhsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE_SHIFT) &&
           "INST_PREF_SIZE does not fit GFX12's 8-bit field");
    RJ_AMDHSA_BITS_SET(rsrc3, amdhsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE, lines);
    return;
  default:
    assert(false && "the target has no INST_PREF_SIZE");
    return;
  }
}

} // namespace rocjitsu
