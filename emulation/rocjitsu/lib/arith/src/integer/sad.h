// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar and packed sums of absolute differences with integer accumulators.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_SAD_H_
#define RJ_ARITH_SAD_H_

#include <stdint.h>

namespace rocjitsu::arith {

enum class Sad { U8, HighU8, U16, U32, MaskedU8, QuadU16, MaskedQuadU16, MaskedQuadU32 };

// Accumulate absolute differences. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Sad Op>
void sad_x86_64_v3(uint32_t exec_mask, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *b, const uint32_t *const *c);

} // namespace rocjitsu::arith

#endif
