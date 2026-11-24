// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// V4 entry points for gfx1201 sparse integer WMMA.
#include "wmma/swmmac_integer.h"
#include "wmma/x86_64/swmmac_integer_x86_64.h"

namespace rocjitsu::arith {
template <int Bits, int K, int Wave>
void swmmac_integer_x86_64_v4(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *index) {
  sparse_x86::compute_integer<Bits, K, Wave>(mode, d, a, b, index);
}

template SwmmacIntegerBackend swmmac_integer_x86_64_v4<8, 32, 32>;
template SwmmacIntegerBackend swmmac_integer_x86_64_v4<8, 32, 64>;
template SwmmacIntegerBackend swmmac_integer_x86_64_v4<4, 32, 32>;
template SwmmacIntegerBackend swmmac_integer_x86_64_v4<4, 32, 64>;
template SwmmacIntegerBackend swmmac_integer_x86_64_v4<4, 64, 32>;
template SwmmacIntegerBackend swmmac_integer_x86_64_v4<4, 64, 64>;

} // namespace rocjitsu::arith
