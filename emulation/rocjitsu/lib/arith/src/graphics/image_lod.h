// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CPU backends for a batch of quad footprints. No descriptor or memory access.
#pragma once
#include "rj_arith/detail/rj_arith_graphics.h"
namespace rocjitsu::arith {
// Query execution needs only axis summaries, not the sampler's retained gradients.
struct ImageLodAxes {
  uint32_t norms[2];
  int32_t lods[2];
};

// Compute selected/raw LODs for a validated batch on the named CPU tier.
void image_get_lod_x86_64_v3(bool gfx10, int count, const rj_arith_image_gradients *input,
                             const rj_arith_image_lod_controls &controls, uint32_t *output);
// Single sampler footprint, lowered with the named CPU features.
void image_footprint_single_x86_64_v3(const rj_arith_image_gradients *input,
                                      const uint32_t *extents, int dimensions,
                                      rj_arith_image_footprint *output);
} // namespace rocjitsu::arith
