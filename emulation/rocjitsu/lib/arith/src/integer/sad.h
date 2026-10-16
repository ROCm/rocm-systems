// SPDX-License-Identifier: MIT

// Scalar and packed sums of absolute differences with integer accumulators.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_SAD_H_
#define RJ_ARITH_SAD_H_

#include <stdint.h>

namespace rocjitsu::arith {

enum class Sad { U8, HighU8, U16, U32, MaskedU8, QuadU16, MaskedQuadU16, MaskedQuadU32 };

// Return whether the SAD operation computes a quad of overlapping differences.
constexpr bool sad_quad(Sad op) { return op >= Sad::QuadU16; }

// Return whether SAD results are packed into 16-bit halves.
constexpr bool sad_packed(Sad op) { return op == Sad::QuadU16 || op == Sad::MaskedQuadU16; }

// Return whether zero source bytes suppress their SAD contributions.
constexpr bool sad_masked(Sad op) {
  return op == Sad::MaskedU8 || op == Sad::MaskedQuadU16 || op == Sad::MaskedQuadU32;
}

// Return the number of destination VGPRs required by this SAD operation.
constexpr int sad_outputs(Sad op) { return op == Sad::MaskedQuadU32 ? 4 : sad_quad(op) ? 2 : 1; }

// Accumulate absolute differences. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Sad Op>
void sad_x86_64_v3(uint32_t exec_mask, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *b, const uint32_t *const *c);

} // namespace rocjitsu::arith

#endif
