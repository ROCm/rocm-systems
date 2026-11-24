// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CPU-specific empirical FP32 sine arithmetic for unmodified waves.

#include "math/trig.h"
#include "math/x86_64/trig_exact_x86_64.h"

namespace rocjitsu::arith {
void sin_exact_x86_64_v4(uint32_t *d, const uint32_t *a, unsigned denorm, bool quiet_snan) {
  trig::evaluate_wave(d, a, denorm, quiet_snan);
}
} // namespace rocjitsu::arith
