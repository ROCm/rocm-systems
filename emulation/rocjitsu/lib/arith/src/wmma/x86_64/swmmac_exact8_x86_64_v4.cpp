// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "wmma/swmmac_exact8.h"
#include "wmma/x86_64/swmmac_exact8_x86_64.h"

namespace rocjitsu::arith {

// Evaluate two exact sparse DOT8 stages across each output row.
void swmmac_exact8_x86_64_v4(bool bf8_a, bool bf8_b, bool wave64, int key, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *index) {
  sparse_exact8::run(bf8_a, bf8_b, wave64, key, d, a, b, index);
}

} // namespace rocjitsu::arith
