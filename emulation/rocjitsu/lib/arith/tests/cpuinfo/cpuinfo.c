// SPDX-License-Identifier: MIT

// Standalone CPU-feature diagnostic printing detected flags and their name.

#include "rj_arith/rj_arith.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

// Print detected CPU flags as hexadecimal and a readable CPU-level name.
int main(void) {
  const uint64_t flags = rj_arith_init_cpu_flags();
  printf("0x%016" PRIx64 " (%s)\n", flags, rj_arith_cpu_name(flags));
  return 0;
}
