// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Public BVH intersection validation, CPU dispatch and overlap-safe publication.
#include "graphics/bvh.h"
#include "common/validation.h"
#include "graphics/bvh_kernel.h"
#include "rj_arith/detail/rj_arith_graphics.h"
namespace {
using namespace rocjitsu::arith;
// Validate before accessing operands and compute every result before committing D.
template <int Lanes>
int execute(uint64_t flags, ExecMask<Lanes> exec_mask, uint64_t mode, uint32_t *const *d,
            const void *nodes, uint32_t node_stride, ExecMask<Lanes> fetch_mask,
            const rj_arith_bvh_ray *rays, const rj_arith_bvh_controls *controls) {
  if (int error = loose_only(flags, false)) {
    return error;
  }
  if (int error = validate(flags, mode)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if (controls->grow > 255 || controls->sort > 3 || controls->sort_enabled > 1 ||
      controls->barycentrics > 1 || controls->pointer_flags > 1) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  uint32_t result[4 * Lanes];
  auto kernel = bvh::execute;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    kernel = bvh_x86_64_v4;
  } else
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
      if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    kernel = bvh_x86_64_v3;
  }
#endif
  kernel(Lanes, exec_mask, fetch_mask, nodes, node_stride, rays, *controls, result);
  if (exec_mask == ~ExecMask<Lanes>(0)) {
    for (int i = 0; i < 4; ++i) {
      __builtin_memcpy(d[i], result + i * Lanes, Lanes * sizeof(uint32_t));
    }
  } else {
    for (int lane = 0; lane < Lanes; ++lane) {
      if ((exec_mask >> lane) & 1) {
        for (int i = 0; i < 4; ++i) {
          d[i][lane] = result[i * Lanes + lane];
        }
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace
// Compute gfx11 IMAGE_BVH_INTERSECT_RAY from prefetched memory operands.
int rj_arith_image_bvh_intersect_ray_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const void *nodes, uint32_t node_stride,
                                         uint32_t fetch_mask, const rj_arith_bvh_ray *rays,
                                         const rj_arith_bvh_controls *controls) {
  return execute<32>(flags, exec_mask, instruction_flags, d, nodes, node_stride, fetch_mask, rays,
                     controls);
}
// Compute gfx11 IMAGE_BVH_INTERSECT_RAY from prefetched memory operands.
int rj_arith_image_bvh_intersect_ray_w64(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const void *nodes, uint32_t node_stride,
                                         uint64_t fetch_mask, const rj_arith_bvh_ray *rays,
                                         const rj_arith_bvh_controls *controls) {
  return execute<64>(flags, exec_mask, instruction_flags, d, nodes, node_stride, fetch_mask, rays,
                     controls);
}
