// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CPU-specific BVH arithmetic on prefetched node records.
#pragma once
#include "rj_arith/detail/rj_arith_graphics.h"
namespace rocjitsu::arith {
// Requires the named CPU tier; results are four words per active lane.
void bvh_x86_64_v3(int lanes, uint64_t exec_mask, uint64_t fetch_mask, const void *nodes,
                   uint32_t node_stride, const rj_arith_bvh_ray *rays,
                   const rj_arith_bvh_controls &controls, uint32_t *result);
// Requires the named CPU tier; results are four words per active lane.
void bvh_x86_64_v4(int lanes, uint64_t exec_mask, uint64_t fetch_mask, const void *nodes,
                   uint32_t node_stride, const rj_arith_bvh_ray *rays,
                   const rj_arith_bvh_controls &controls, uint32_t *result);
} // namespace rocjitsu::arith
