// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IMAGE_GET_LOD numeric execution and the shared sampler footprint interface.
// Descriptor decoding, register observations and texture fetching stay in rocjitsu.
#include "graphics/image_lod.h"
#include "common/validation.h"
#include "graphics/image_lod_output.h"
#include "graphics/image_lod_scalar.h"
#include "rj_arith/detail/rj_arith_graphics.h"

// Reuse the footprint model across the library boundary without inline ISA symbols.
void rj_arith_detail_image_footprint([[maybe_unused]] uint64_t cpu_flags,
                                     const rj_arith_image_gradients *input, const uint32_t *extents,
                                     uint32_t dimensions, rj_arith_image_footprint *output) {
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(cpu_flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::image_footprint_single_x86_64_v3(input, extents, int(dimensions), output);
    return;
  }
#endif
  *output = rocjitsu::arith::image_lod::footprint(*input, extents, int(dimensions));
}
// Share the same mip remapping coefficients with the texture sampler.
uint32_t rj_arith_detail_image_mip_fraction(uint32_t fraction, uint32_t perf_mip,
                                            uint32_t perf_mod) {
  return rocjitsu::arith::image_lod::mip_fraction(fraction, perf_mip, perf_mod);
}

namespace {
using namespace rocjitsu::arith;
// Compute active quads before writes, including helper-lane data captured by the caller.
template <int Lanes, bool Gfx10>
int get_lod(uint64_t flags, ExecMask<Lanes> exec_mask, uint64_t mode, uint32_t *const *d,
            const rj_arith_image_gradients *input, const rj_arith_image_lod_controls *control) {
  if (int error = loose_only(flags, false)) {
    return error;
  }
  if (int error = validate(flags, mode)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  // Inactive quads need no valid source data. Fill them with zeroes before SIMD
  // evaluation rather than reading outside the instruction's source footprint.
  rj_arith_image_gradients gradients[Lanes / 4]{};
  for (int quad = 0; quad < Lanes / 4; ++quad) {
    if ((exec_mask >> (4 * quad)) & 15) {
      gradients[quad] = input[quad];
    }
  }
  uint32_t result[2 * (Lanes / 4)]{};
  bool vectorized = false;
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!vectorized && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    image_get_lod_x86_64_v3(Gfx10, Lanes / 4, gradients, *control, result);
    vectorized = true;
  }
#endif
  if (!vectorized) {
    for (int quad = 0; quad < Lanes / 4; ++quad) {
      if ((exec_mask >> (4 * quad)) & 15) {
        const auto footprint =
            image_lod::footprint(gradients[quad], control->extents, int(control->dimensions));
        float selected, raw;
        image_lod_output::lod_values<Gfx10>(footprint, gradients[quad], *control, selected, raw);
        result[quad] = as_bits(selected);
        result[Lanes / 4 + quad] = as_bits(raw);
      }
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane / 4];
      d[1][lane] = result[Lanes / 4 + lane / 4];
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace
// Return selected-mip and raw LOD using the inherited gfx11 policy.
int rj_arith_image_get_lod_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const rj_arith_image_gradients *quad_inputs,
                               const rj_arith_image_lod_controls *controls) {
  return get_lod<32, false>(flags, exec_mask, instruction_flags, d, quad_inputs, controls);
}
// Return selected-mip and raw LOD using the inherited gfx11 policy.
int rj_arith_image_get_lod_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const rj_arith_image_gradients *quad_inputs,
                               const rj_arith_image_lod_controls *controls) {
  return get_lod<64, false>(flags, exec_mask, instruction_flags, d, quad_inputs, controls);
}
// Return selected-mip and raw LOD using the inherited gfx10 policy.
int rj_arith_image_get_lod_w32_gfx10(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d,
                                     const rj_arith_image_gradients *quad_inputs,
                                     const rj_arith_image_lod_controls *controls) {
  return get_lod<32, true>(flags, exec_mask, instruction_flags, d, quad_inputs, controls);
}
// Return selected-mip and raw LOD using the inherited gfx10 policy.
int rj_arith_image_get_lod_w64_gfx10(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d,
                                     const rj_arith_image_gradients *quad_inputs,
                                     const rj_arith_image_lod_controls *controls) {
  return get_lod<64, true>(flags, exec_mask, instruction_flags, d, quad_inputs, controls);
}
