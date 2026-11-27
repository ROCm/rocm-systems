// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded derivative and footprint records shared by GET_LOD and the sampler.
// Image memory operations and descriptor interpretation remain outside arithmetic.
#ifndef RJ_ARITH_IMAGE_FOOTPRINT_H_
#define RJ_ARITH_IMAGE_FOOTPRINT_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct rj_arith_image_gradients {
  double dx[3], dy[3];
  // Cube unfolding tie directions: dx.u, dx.v, dy.u, dy.v; otherwise zero.
  int32_t rounding[4];
  uint32_t unbounded;
} rj_arith_image_gradients;

typedef struct rj_arith_image_footprint {
  int32_t fixed[6]; // dx.xyz followed by dy.xyz, in the common aligned exponent.
  uint32_t norms[2];
  int32_t lods[2]; // Major/minor in 1/256 units; zero footprint is {-8192,-16384}.
} rj_arith_image_footprint;

// Internal-to-rocjitsu arithmetic bridge for sampler reuse of the query model.
// This is not an additional GPU instruction entry point or a stable public API.
// CPU_FLAGS supplies a cached CPU level; its other fields are ignored. Extents
// are positive; dimensions is 1..3. No image-memory operation is performed.
void rj_arith_detail_image_footprint(uint64_t cpu_flags, const rj_arith_image_gradients *input,
                                     const uint32_t *extents, uint32_t dimensions,
                                     rj_arith_image_footprint *output);
// Fixed8 mip remapping: fraction=0..255, perf_mip=0..15, perf_mod=0..7.
uint32_t rj_arith_detail_image_mip_fraction(uint32_t fraction, uint32_t perf_mip,
                                            uint32_t perf_mod);
#ifdef __cplusplus
} // extern "C"
#endif
#endif // RJ_ARITH_IMAGE_FOOTPRINT_H_
