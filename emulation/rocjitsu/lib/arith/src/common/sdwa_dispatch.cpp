// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline dispatch for inherited SDWA floating output scaling and finalization.
#include "common/sdwa_dispatch.h"
#include "common/alu.h"
#include "common/sdwa_output.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace {
using rocjitsu::arith::SdwaOutput;
// Scale rounded scalar results while preserving inactive lanes and unselected halves.
template <SdwaOutput Output>
void scale_sdwa_scalar(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode,
                       uint32_t *data) {
  const auto policy = rocjitsu::arith::sdwa_output_policy(flags, mode, Output);
  for (int lane = 0; lane < lanes; ++lane) {
    if ((exec_mask >> lane) & 1U) {
      data[lane] = rocjitsu::arith::sdwa_scale_word<Output>(data[lane], mode, policy);
    }
  }
}
} // namespace

namespace rocjitsu::arith {
// Select a caller-enabled SIMD pass; arithmetic and FP-status reporting are already complete.
void finalize_sdwa_half_wave(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode,
                             uint32_t *data) {
  mode = uint32_t(rdna3_output_modifiers(flags, mode));
  if (!(mode & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP))) {
    return;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    return finalize_sdwa_half_x86_64_v4(flags, exec_mask, lanes, mode, data);
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    return finalize_sdwa_half_x86_64_v3(flags, exec_mask, lanes, mode, data);
  }
#endif
  const auto policy = sdwa_output_policy(flags, mode);
  for (int lane = 0; lane < lanes; ++lane) {
    if ((exec_mask >> lane) & 1U) {
      data[lane] = sdwa_finalize_half_word(data[lane], mode, policy);
    }
  }
}
// Dispatch the inherited rounded-output policy using only caller-enabled CPU features.
void scale_sdwa_wave(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode, uint32_t *data,
                     SdwaOutput output) {
  // Native nearest output ignores the guest denormal gate, but still observes IEEE.
  const uint64_t gate_flags =
      output == SdwaOutput::F32Nearest ? flags | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : flags;
  mode = uint32_t(rdna3_output_modifiers(gate_flags, mode));
  if (!(mode & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP))) {
    return;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    return scale_sdwa_x86_64_v4(flags, exec_mask, lanes, mode, data, output);
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    return scale_sdwa_x86_64_v3(flags, exec_mask, lanes, mode, data, output);
  }
#endif
  if (output == SdwaOutput::F16) {
    return scale_sdwa_scalar<SdwaOutput::F16>(flags, exec_mask, lanes, mode, data);
  }
  if (output == SdwaOutput::F32Nearest) {
    return scale_sdwa_scalar<SdwaOutput::F32Nearest>(flags, exec_mask, lanes, mode, data);
  }
  return scale_sdwa_scalar<SdwaOutput::F32Mode>(flags, exec_mask, lanes, mode, data);
}
} // namespace rocjitsu::arith
