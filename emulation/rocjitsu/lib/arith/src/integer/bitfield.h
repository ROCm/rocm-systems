// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bitfield extraction/insertion and alignment of adjacent source words.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#ifndef RJ_ARITH_BITFIELD_H_
#define RJ_ARITH_BITFIELD_H_

#include <stdint.h>

namespace rocjitsu::arith {

enum class Bitfield {
  ExtractUnsigned,
  ExtractSigned,
  Insert,
  Mask,
  Reverse,
  AlignBit,
  AlignByte,
  Permute
};

// Extract, insert, or align bitfields. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Bitfield Op>
void bitfield_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                        const uint32_t *c);

// Extract, insert, or align bitfields. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Bitfield Op>
void bitfield_x86_64_v4(uint32_t exec_mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                        const uint32_t *c);

} // namespace rocjitsu::arith

#endif
