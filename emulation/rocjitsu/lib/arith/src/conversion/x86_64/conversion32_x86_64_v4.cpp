// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD conversion with guest rounding and denormal policies shared with scalar execution.
#include "conversion/float_conversion.h"
#include "conversion/x86_64/lanes.h"

namespace rocjitsu::arith {
// Convert a wave and preserve inactive lanes and unselected destination halves.
template <Conversion32 Op>
void conversion32_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a) {
  using V = conversion_simd::Lanes<32>;
  for (int lane = 0; lane < 32; lane += V::size()) {
    V raw(a + lane);
    V result = conversion_detail::convert32<Op>(raw, flags, mode);
    result.store(d + lane, exec_mask >> lane);
  }
}

template void conversion32_x86_64_v4<Conversion32::SignedToFloat>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::UnsignedToFloat>(uint64_t, uint32_t, uint32_t,
                                                                    uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::FloatToSigned>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::FloatToUnsigned>(uint64_t, uint32_t, uint32_t,
                                                                    uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::Nearest>(uint64_t, uint32_t, uint32_t,
                                                            uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::Floor>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *);

} // namespace rocjitsu::arith
