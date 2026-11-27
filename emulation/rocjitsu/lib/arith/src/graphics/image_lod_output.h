// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Common GET_LOD sampler-control ordering, compiled in each arithmetic ISA domain.
#pragma once
#include "common/validation.h"
#include "graphics/image_lod_scalar.h"
#include "rj_arith/detail/rj_arith_graphics.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::image_lod_output {
// Quantize a sampler LOD to fixed8 with nearest-even ties.
inline double round_even8(double value) {
  const double scaled = value * 256, lower = __builtin_floor(scaled), fraction = scaled - lower;
  return (lower + (fraction > 0.5 || (fraction == 0.5 && __builtin_fmod(lower, 2.0) != 0))) / 256;
}
// Resolve the inherited footprint, bias, residency and mip-filter selection order.
template <bool Gfx10, typename Footprint>
inline void lod_values(const Footprint &footprint, const rj_arith_image_gradients &input,
                       const rj_arith_image_lod_controls &control, float &selected, float &raw) {
  const bool zero = footprint.norms[0] == 0 && footprint.norms[1] == 0;
  int32_t fixed = footprint.lods[0];
  if (control.max_anisotropy && control.dimensions < 3) {
    const int reduction = __builtin_ctz(control.max_anisotropy) * 256;
    fixed = zero ? -16384 : maximum(-16384, maximum(footprint.lods[1], fixed - reduction));
  }
  double value = fixed / 256.0 + control.bias;
  if (input.unbounded) {
    value = control.max_lod;
  }
  raw = float(!Gfx10 && zero && !control.max_anisotropy ? 0 : value);
  double lod = clamp(value, control.min_lod, control.max_lod);
  if constexpr (Gfx10) {
    lod =
        control.mip_filter ? clamp(lod, 0.0, double(control.last_level - control.first_level)) : 0;
  } else {
    lod = round_even8(maximum(lod, control.resident_min_lod - control.first_level));
    if (!control.mip_filter) {
      lod = __builtin_floor(control.resident_min_lod) - control.first_level;
    }
    lod = clamp(lod, 0.0, double(control.last_level - control.first_level));
  }
  double level = __builtin_floor(lod + (control.mip_filter == 1 ? 0.5 : 0));
  if (control.mip_filter == 2) {
    level +=
        image_lod::mip_fraction(uint32_t((lod - level) * 256), control.perf_mip, control.perf_mod) /
        256.0;
  }
  selected = float(level);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::image_lod_output
