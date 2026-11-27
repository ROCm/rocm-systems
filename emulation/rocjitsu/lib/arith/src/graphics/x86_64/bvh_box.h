// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Four-child slab intersections, including parallel rays and ordered bounds.
#pragma once
#include "rj_arith/detail/rj_arith_graphics.h"
#include <immintrin.h>
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::bvh {
// Match scalar first-on-ties min/max, including unordered operands.
inline __m128 box_min(__m128 a, __m128 b) {
  return _mm_blendv_ps(a, b, _mm_cmp_ps(b, a, _CMP_LT_OQ));
}
inline __m128 box_max(__m128 a, __m128 b) {
  return _mm_blendv_ps(a, b, _mm_cmp_ps(a, b, _CMP_LT_OQ));
}
// Return valid-bound bits and near/far distances for each of a node's four children.
inline unsigned box_distances(const uint32_t *node, bool half, const rj_arith_bvh_ray &ray,
                              float *near_values, float *far_values) {
  auto near = _mm_set1_ps(-__builtin_inff()), far = _mm_set1_ps(__builtin_inff());
  auto valid = _mm_castsi128_ps(_mm_set1_epi32(-1));
  const auto bound = [&](int axis) {
    if (half) {
      auto bits = _mm_setr_epi32(int(node[4 + axis / 2]), int(node[7 + axis / 2]),
                                 int(node[10 + axis / 2]), int(node[13 + axis / 2]));
      bits = _mm_and_si128(_mm_srli_epi32(bits, (axis & 1) * 16), _mm_set1_epi32(0xffff));
      return _mm_cvtph_ps(_mm_packus_epi32(bits, bits));
    }
    return _mm_castsi128_ps(_mm_setr_epi32(int(node[4 + axis]), int(node[10 + axis]),
                                           int(node[16 + axis]), int(node[22 + axis])));
  };
  for (int axis = 0; axis < 3; ++axis) {
    const auto lo = bound(axis), hi = bound(axis + 3), origin = _mm_set1_ps(ray.origin[axis]);
    valid = _mm_and_ps(valid, _mm_cmp_ps(lo, hi, _CMP_LE_OQ));
    if (__builtin_isinf(ray.inverse[axis])) {
      valid = _mm_and_ps(valid, _mm_and_ps(_mm_cmp_ps(lo, origin, _CMP_LE_OQ),
                                           _mm_cmp_ps(origin, hi, _CMP_LE_OQ)));
    } else {
      const auto inverse = _mm_set1_ps(ray.inverse[axis]);
      const auto lower = _mm_mul_ps(_mm_sub_ps(lo, origin), inverse),
                 upper = _mm_mul_ps(_mm_sub_ps(hi, origin), inverse);
      near = box_max(near, box_min(lower, upper));
      far = box_min(far, box_max(lower, upper));
    }
  }
  _mm_storeu_ps(near_values, near);
  _mm_storeu_ps(far_values, far);
  return unsigned(_mm_movemask_ps(valid));
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::bvh
