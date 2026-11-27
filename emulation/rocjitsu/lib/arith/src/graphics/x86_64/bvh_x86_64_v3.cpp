// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 compensated RTZ intersection arithmetic.
#define RJ_ARITH_BVH_SIMD 1
#include "graphics/bvh.h"
#include "graphics/bvh_kernel.h"
namespace rocjitsu::arith {
// Evaluate validated prefetched rays with the named CPU features.
void bvh_x86_64_v3(int lanes, uint64_t exec_mask, uint64_t fetch_mask, const void *nodes,
                   uint32_t node_stride, const rj_arith_bvh_ray *rays,
                   const rj_arith_bvh_controls &controls, uint32_t *result) {
  bvh::execute(lanes, exec_mask, fetch_mask, nodes, node_stride, rays, controls, result);
}
} // namespace rocjitsu::arith
