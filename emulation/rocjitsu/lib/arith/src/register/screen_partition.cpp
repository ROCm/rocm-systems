// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA screen-partition coverage using rocjitsu's inherited CDNA2 table.
// gfx942/gfx950 retain its extrapolated model; strict empirical modes are unsupported.

#include "register/screen_partition.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Eight four-bit coverage masks per word keep the shared lookup table at 128 bytes.
const uint32_t screen_partition_table[32] = {
    0xfff5f731U, 0xfffffff7U, 0xffafe62fU, 0xffffffbfU, 0xf5ffc4fdU, 0xfffffdffU, 0xafff8fb9U,
    0xffffefffU, 0xfdc4ffffU, 0xfffefff6U, 0xb98fffffU, 0xffdff99fU, 0x31f7ffffU, 0xfbfff9ffU,
    0x2fe6ffffU, 0x7fff6ff6U, 0xfffffffbU, 0xfffafb32U, 0xffffff7fU, 0xff5fd91fU, 0xfffffeffU,
    0xfaffc8feU, 0xffffdfffU, 0x5fff4f76U, 0xfffdfff9U, 0xfec8ffffU, 0xffeff66fU, 0x764fffffU,
    0xf7fff6ffU, 0x32fbffffU, 0xbfff9ff9U, 0x1fd9ffffU,
};

} // namespace rocjitsu::arith

// Map each low-byte rectangle selector to its four-bit shader-engine coverage.
int rj_arith_v_screen_partition_4se_b32_w64(uint64_t flags, uint64_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  auto semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    screen_partition_x86_64_v4(exec_mask, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    screen_partition_x86_64_v3(exec_mask, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[64];
  for (int lane = 0; lane < 64; ++lane) {
    uint32_t selector = a[0][lane] & 255;
    result[lane] = (screen_partition_table[selector >> 3] >> ((selector & 7) * 4)) & 15;
  }
  for (int lane = 0; lane < 64; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}
