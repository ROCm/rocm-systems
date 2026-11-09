// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/vgpr_writeback.h"

#include <cassert>

namespace rocjitsu {
namespace amdgpu {
namespace vgpr_writeback {
namespace {

using Writer = void (*)(uint8_t *const *dst, bool sram_ecc, uint32_t wave, uint64_t exec,
                        const uint8_t *lanes);

/// One out-of-line instance per format, so write_back_as<F>()'s inlined levels
/// are compiled once here rather than into every caller. The group pointers
/// are built here, where the stride is a constant: they stay in registers,
/// where an array built by the caller was stored a pointer at a time and
/// reloaded as one vector, which stalls on store forwarding.
template <DstFormat F>
void write(uint8_t *const *dst, bool sram_ecc, uint32_t wave, uint64_t exec, const uint8_t *lanes) {
  // A wave32 response ends after two groups; its unread groups 2-3 repeat 0-1.
  const uint8_t *const upper = wave == 64 ? lanes + 32 * kStride<F> : lanes;
  const uint8_t *const groups[4] = {lanes, lanes + 16 * kStride<F>, upper, upper + 16 * kStride<F>};
  write_back_as<F>(dst, sram_ecc, wave, exec, groups);
}

template <uint8_t Elem, bool Sign> Writer narrow_writer(const DstFormat &f) {
  if (f.d16_hi)
    return &write<DstFormat{.elem_size = Elem, .sign_extend = Sign, .d16_hi = true}>;
  if (f.d16_lo)
    return &write<DstFormat{.elem_size = Elem, .sign_extend = Sign, .d16_lo = true}>;
  return &write<DstFormat{.elem_size = Elem, .sign_extend = Sign}>;
}

Writer writer(const DstFormat &f) {
  if (f.elem_size == 4) {
    switch (f.num_elems) {
    case 1:
      return &write<DstFormat{.num_elems = 1}>;
    case 2:
      return &write<DstFormat{.num_elems = 2}>;
    case 3:
      return &write<DstFormat{.num_elems = 3}>;
    default:
      return &write<DstFormat{.num_elems = 4}>;
    }
  }
  if (f.elem_size == 1)
    return f.sign_extend ? narrow_writer<1, true>(f) : narrow_writer<1, false>(f);
  return f.sign_extend ? narrow_writer<2, true>(f) : narrow_writer<2, false>(f);
}

/// The per-lane loop, for a format known only at run time.
void few_lanes(const DstFormat &f, uint8_t *const *dst, bool sram_ecc, uint64_t exec,
               const uint8_t *lanes) {
  const uint32_t stride = f.elem_size * f.num_elems;
  for (; exec; exec &= exec - 1) {
    const uint32_t l = std::countr_zero(exec);
    const uint8_t *src = lanes + l * stride;
    if (f.elem_size == 4) {
      for (uint32_t r = 0; r < f.num_elems; ++r)
        std::memcpy(dst[r] + 4 * l, src + 4 * r, 4);
      continue;
    }
    uint32_t h;
    if (f.elem_size == 1) {
      h = f.sign_extend ? static_cast<uint32_t>(static_cast<int8_t>(*src)) : *src;
    } else {
      uint16_t x;
      std::memcpy(&x, src, 2);
      h = f.sign_extend ? static_cast<uint32_t>(static_cast<int16_t>(x)) : x;
    }
    uint32_t v = h;
    if (f.d16_hi || f.d16_lo) {
      uint32_t old;
      std::memcpy(&old, dst[0] + 4 * l, 4);
      v = f.d16_hi ? (sram_ecc ? 0 : old & 0xFFFFu) | h << 16
                   : (sram_ecc ? 0 : old & 0xFFFF0000u) | (h & 0xFFFFu);
    }
    std::memcpy(dst[0] + 4 * l, &v, 4);
  }
}

} // namespace

void write_back(const DstFormat &f, uint8_t *const *dst, bool sram_ecc, uint32_t wave,
                uint64_t exec, const uint8_t *lanes) {
  assert(format_of(f.elem_size, f.num_elems, f.sign_extend, f.d16_hi, f.d16_lo) == f);
  assert((wave == 32 || wave == 64) && (exec & ~prefix(wave)) == 0);
  // A few lanes cost less one at a time than the dispatch, the group pointers
  // and the whole-wave loops do: up to two lanes of one register, four of
  // several, where each whole-wave chunk does more work.
  if (std::popcount(exec) <= (regs(f) > 1 ? 4 : 2))
    return few_lanes(f, dst, sram_ecc, exec, lanes);
  writer(f)(dst, sram_ecc, wave, exec, lanes);
}

} // namespace vgpr_writeback
} // namespace amdgpu
} // namespace rocjitsu
