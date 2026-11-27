// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 four-quad footprint quantization and fixed-point LOD arithmetic.
#include "graphics/image_lod.h"
#include "graphics/image_lod_output.h"
#include "graphics/image_lod_scalar.h"
#include "graphics/x86_64/image_lod_simd.h"
namespace rocjitsu::arith {
// Evaluate a validated batch using the named CPU features.
void image_get_lod_x86_64_v3(bool gfx10, int count, const rj_arith_image_gradients *input,
                             const rj_arith_image_lod_controls &controls, uint32_t *output) {
  ImageLodAxes footprints[16];
  image_lod_simd::footprints(count, input, controls.extents, int(controls.dimensions), footprints);
  for (int quad = 0; quad < count; ++quad) {
    float selected, raw;
    if (gfx10) {
      image_lod_output::lod_values<true>(footprints[quad], input[quad], controls, selected, raw);
    } else {
      image_lod_output::lod_values<false>(footprints[quad], input[quad], controls, selected, raw);
    }
    output[quad] = as_bits(selected);
    output[count + quad] = as_bits(raw);
  }
}
// Keep native single-sample lowering when sampling cannot batch independent quads.
void image_footprint_single_x86_64_v3(const rj_arith_image_gradients *input,
                                      const uint32_t *extents, int dimensions,
                                      rj_arith_image_footprint *output) {
  *output = image_lod::footprint(*input, extents, dimensions);
}
} // namespace rocjitsu::arith
