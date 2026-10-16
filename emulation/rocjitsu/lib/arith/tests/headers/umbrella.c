// SPDX-License-Identifier: MIT

// Standalone C/C++ compilation of the public umbrella header.

// Deliberately test that the umbrella exposes both public components.
#include "@rj_arith_source_dir@/include/rj_arith/rj_arith.h"

// Repeated inclusion must be harmless.
#include "@rj_arith_source_dir@/include/rj_arith/rj_arith.h"

#include <stddef.h>
#include <stdint.h>

// Provide a complete translation unit for standalone header compilation.
int main(void) {
  uint32_t words[32] = {0};
  uint32_t *vgpr = words;
  const uint32_t *input = words;
  return rj_arith_v_fma_f32_w32(rj_arith_init_cpu_flags(), 1, 0, &vgpr, &input, &input, &input,
                                NULL) != RJ_ARITH_SUCCESS;
}
