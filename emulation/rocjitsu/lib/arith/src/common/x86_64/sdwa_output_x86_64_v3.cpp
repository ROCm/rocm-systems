// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 SDWA finalization, isolated from other compilation domains.
#include "common/sdwa_dispatch.h"
#include "common/x86_64/sdwa_output_x86_64.h"

#include <stdint.h>

namespace rocjitsu::arith {
// Finalize selected halves using this backend's explicitly compiled lane width.
void finalize_sdwa_half_x86_64_v3(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode,
                                  uint32_t *data) {
  RJ_ARITH_IMPL_NAMESPACE::finalize_sdwa_half_simd(flags, exec_mask, lanes, mode, data);
}
// Select the output format outside the vector loop; policies share one raw-bit model.
void scale_sdwa_x86_64_v3(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode,
                          uint32_t *data, SdwaOutput output) {
  if (output == SdwaOutput::F16) {
    return RJ_ARITH_IMPL_NAMESPACE::scale_sdwa_simd<SdwaOutput::F16>(flags, exec_mask, lanes, mode,
                                                                     data);
  }
  if (output == SdwaOutput::F32Nearest) {
    return RJ_ARITH_IMPL_NAMESPACE::scale_sdwa_simd<SdwaOutput::F32Nearest>(flags, exec_mask, lanes,
                                                                            mode, data);
  }
  return RJ_ARITH_IMPL_NAMESPACE::scale_sdwa_simd<SdwaOutput::F32Mode>(flags, exec_mask, lanes,
                                                                       mode, data);
}
} // namespace rocjitsu::arith
