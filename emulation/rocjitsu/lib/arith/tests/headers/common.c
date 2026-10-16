// SPDX-License-Identifier: MIT

// Standalone C/C++ compilation of the common public declarations.

// Intentionally include only the header under test, twice: test self-containment
// and include guards in both C99 and C++.
#include "@rj_arith_source_dir@/include/rj_arith/detail/rj_arith_common.h"

// Repeated inclusion must be harmless.
#include "@rj_arith_source_dir@/include/rj_arith/detail/rj_arith_common.h"

// Provide a complete translation unit for standalone header compilation.
int main(void) { return (rj_arith_init_cpu_flags() & ~RJ_ARITH_CPU_MASK) != 0; }
