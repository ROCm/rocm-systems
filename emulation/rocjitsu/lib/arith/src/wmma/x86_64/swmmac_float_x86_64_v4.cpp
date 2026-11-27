// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// V4 entry points for gfx1201 sparse floating-point WMMA.
#include "wmma/swmmac_float.h"
#include "wmma/x86_64/swmmac_loose_x86_64.h"

namespace rocjitsu::arith {
template <bool Bf16, int Wave, bool Packed>
void swmmac16_x86_64_v4(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *index, bool saturate) {
  sparse_x86::compute_float<16, Bf16, Bf16, Wave, Packed>(mode, d, a, b, index, saturate);
}

template Swmmac16Backend swmmac16_x86_64_v4<false, 32, false>;
template Swmmac16Backend swmmac16_x86_64_v4<false, 32, true>;
template Swmmac16Backend swmmac16_x86_64_v4<false, 64, false>;
template Swmmac16Backend swmmac16_x86_64_v4<false, 64, true>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 32, false>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 32, true>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 64, false>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 64, true>;

} // namespace rocjitsu::arith
