// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Internal SDWA output passes for legacy floating-point entry points.
// Only validated calls reach these routines; status classification precedes this pass.
#pragma once

#include "common/isa.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith {
// Shared ABI selector; model definitions stay inside their compilation domain.
enum class SdwaOutput : uint8_t { F16, F32Mode };

// Scale a rounded F16/F32 result with guest controls, then clamp, for active lanes.
// Half results preserve the unselected destination half.
void scale_sdwa_wave(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode, uint32_t *data,
                     SdwaOutput output);
void scale_sdwa_x86_64_v3(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode,
                          uint32_t *data, SdwaOutput output);
void scale_sdwa_x86_64_v4(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode,
                          uint32_t *data, SdwaOutput output);
} // namespace rocjitsu::arith

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Defer legacy SDWA output controls until a validated arithmetic call succeeds.
// The core still owns qualification, operand modifiers, guest arithmetic and FP status.
template <SdwaOutput Output, typename Mask, typename Call>
inline int with_sdwa_output(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                            Call evaluate) {
  if (!(mode & RJ_ARITH_ALU_SDWA)) {
    return evaluate(mode);
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  const uint64_t core_mode =
      mode & ~uint64_t(RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP);
  const int error = evaluate(core_mode);
  if (!error && exec_mask) {
    scale_sdwa_wave(flags, exec_mask, int(sizeof(Mask) * 8), uint32_t(mode), d[0], Output);
  }
  return error;
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
