// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone C/C++ compilation of the common public declarations.

// Intentionally include only the header under test, twice: test self-containment
// and include guards in both C23 and C++17.
#include "@rj_arith_source_dir@/include/rj_arith/detail/rj_arith_common.h"

// Repeated inclusion must be harmless.
#include "@rj_arith_source_dir@/include/rj_arith/detail/rj_arith_common.h"

// Constants must work in C23 as well as C++ constant-expression contexts.
static uint64_t combined_flags = RJ_ARITH_CPU_X86_64_V3 | RJ_ARITH_SEMANTICS_LOOSE;
int main(void) {
  switch (RJ_ARITH_SUCCESS) {
  case RJ_ARITH_SUCCESS:
    return combined_flags == 0;
  case RJ_ARITH_ERROR_INVALID_FLAGS:
    return 2;
  default:
    return 3;
  }
}
