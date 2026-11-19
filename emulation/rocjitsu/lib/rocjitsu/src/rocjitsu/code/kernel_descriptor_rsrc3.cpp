// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_descriptor_rsrc3.h"

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

} // namespace rocjitsu
