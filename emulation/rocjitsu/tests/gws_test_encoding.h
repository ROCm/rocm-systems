// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gws_test_encoding.h
/// @brief Shared GWS (Global Wave Sync) opcode/operand encoding helpers for tests.
///
/// @details Keeps the DS GWS opcode map and the count-carrying ADDR-operand choice
/// in one place so every GWS test (single-CU decode/semantics in ds_gws_test.cpp,
/// cross-XCD rendezvous in xcd_distribution_test.cpp) encodes the ops the same way.
/// Only the architecture-neutral encoding knowledge lives here; each test keeps its
/// own fixture-specific build_ds()/execute setup.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>

namespace rocjitsu::test {

/// @brief The six DS GWS operations, in opcode order from SEMA_RELEASE_ALL.
enum class GwsOp { kReleaseAll, kInit, kSemaV, kSemaBr, kSemaP, kBarrier };

/// @brief Whether @p op reads its resource count from the ADDR VGPR.
/// @details INIT/SEMA_BR/BARRIER carry a count; the rest are address-less.
constexpr bool gws_has_addr(GwsOp op) {
  return op == GwsOp::kInit || op == GwsOp::kSemaBr || op == GwsOp::kBarrier;
}

/// @brief Whether @p arch is a GFX9 (CDNA1-3) target.
constexpr bool gws_is_gfx9(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3;
}

/// @brief DS opcode for GWS @p op on @p arch.
/// @details The six GWS ops are contiguous from SEMA_RELEASE_ALL: GFX9 (CDNA1-3)
/// starts that block at 152, GFX10/11 (RDNA1-3.5) at 24. This is the single source
/// of truth for the map (so, e.g., CDNA3 SEMA_V/SEMA_P/BARRIER are 154/156/157).
constexpr uint16_t gws_opcode(rj_code_arch_t arch, GwsOp op) {
  const uint16_t base = gws_is_gfx9(arch) ? 152 : 24;
  switch (op) {
  case GwsOp::kReleaseAll:
    return base + 0;
  case GwsOp::kInit:
    return base + 1;
  case GwsOp::kSemaV:
    return base + 2;
  case GwsOp::kSemaBr:
    return base + 3;
  case GwsOp::kSemaP:
    return base + 4;
  case GwsOp::kBarrier:
    return base + 5;
  }
  return base;
}

} // namespace rocjitsu::test
