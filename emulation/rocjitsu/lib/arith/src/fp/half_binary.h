// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 add/subtract/multiply, including packed forms and guest FP status and min/max selection.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include "fp/binary.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Output stages inherited by CDNA Compatibility; RDNA uses unbounded-exponent tininess.
struct HalfOutputPolicy {
  bool rounded_tininess = false;
  bool scale_before_rounding = false;
};

// Evaluate FP16 binary arithmetic. Preserve destination lanes outside exec_mask.
// compatibility preserves intermediate FP16 rounding; preserve_input_snan retains
// selected input NaN bits unless that result is clamped.
// Requires the named CPU features and validated instruction controls.
template <Binary Op, bool Packed = false>
void half_binary_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, bool compatibility,
                           bool preserve_input_snan, HalfOutputPolicy output = {});

// Ordinary/packed arithmetic with guest rounding, requiring AVX-512-v4.
// Preserve inactive/unselected storage; Compatibility retains the inherited output stages.
template <Binary Op, bool Packed = false>
void half_binary_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, bool compatibility,
                           bool preserve_input_snan, HalfOutputPolicy output = {});

} // namespace rocjitsu::arith
