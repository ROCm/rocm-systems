// SPDX-License-Identifier: MIT

// FP64 arithmetic with split-word VGPR operands and guest FP policies.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/float_bits.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Reinterpret FP64 bits as double without numeric conversion.
inline double as_double(uint64_t bits) { return FloatBits<double>::value(bits); }

// Return the FP64 bit pattern without numeric conversion.
inline uint64_t double_bits(double value) { return FloatBits<double>::bits(value); }

} // namespace rocjitsu::arith
