// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Exponent extraction from FP32 and FP64 encodings.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Extract floating-point exponents. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void frexp_exp_x86_64_v3(bool fp64, uint32_t exec_mask, uint32_t *d, const uint32_t *const *a,
                         bool flush_input);

} // namespace rocjitsu::arith
