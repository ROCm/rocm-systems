// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD conversion with guest rounding and denormal policies shared with scalar execution.
#include "conversion/float_conversion.h"
#include "conversion/x86_64/lanes.h"

namespace rocjitsu::arith {
// Convert a wave and preserve inactive lanes and unselected destination halves.
template <Conversion16 Op>
void conversion16_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, bool quiet_nan) {
  using V = conversion_simd::Lanes<32>;
  for (int lane = 0; lane < 32; lane += V::size()) {
    V raw(a + lane);
    V result = conversion_detail::convert16<Op>(raw, flags, mode, quiet_nan);
    if constexpr (Op != Conversion16::HalfToFloat) {
      const int shift = (mode & RJ_ARITH_ALU_HIGH_D) ? 16 : 0;
      result = (V(d + lane) & V(~(0xffffU << shift))) | (result << shift);
    }
    result.store(d + lane, exec_mask >> lane);
  }
}

template void conversion16_x86_64_v3<Conversion16::SignedToHalf>(uint64_t, uint32_t, uint32_t,
                                                                 uint32_t *, const uint32_t *,
                                                                 bool);
template void conversion16_x86_64_v3<Conversion16::UnsignedToHalf>(uint64_t, uint32_t, uint32_t,
                                                                   uint32_t *, const uint32_t *,
                                                                   bool);
template void conversion16_x86_64_v3<Conversion16::HalfToSigned>(uint64_t, uint32_t, uint32_t,
                                                                 uint32_t *, const uint32_t *,
                                                                 bool);
template void conversion16_x86_64_v3<Conversion16::HalfToUnsigned>(uint64_t, uint32_t, uint32_t,
                                                                   uint32_t *, const uint32_t *,
                                                                   bool);
template void conversion16_x86_64_v3<Conversion16::FloatToHalf>(uint64_t, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *, bool);
template void conversion16_x86_64_v3<Conversion16::HalfToFloat>(uint64_t, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *, bool);

} // namespace rocjitsu::arith
