/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_WARP_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_WARP_FUNCTIONS_H

#if !defined(__HIPCC_RTC__)
#include "device_library_decls.h"  // ockl warp functions
#endif                             // !defined(__HIPCC_RTC__)

#if defined(__has_attribute) && __has_attribute(maybe_undef)
#define MAYBE_UNDEF __attribute__((maybe_undef))
#else
#define MAYBE_UNDEF
#endif

__device__ static inline unsigned __hip_ds_bpermute(int index, unsigned src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)
              ? __builtin_amdgcn_ds_bpermute(index, tmp.i)
              : 0;
  return tmp.u;
}

__device__ static inline float __hip_ds_bpermutef(int index, float src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)
              ? __builtin_amdgcn_ds_bpermute(index, tmp.i)
              : 0;
  return tmp.f;
}

__device__ static inline unsigned __hip_ds_permute(int index, unsigned src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_permute)
              ? __builtin_amdgcn_ds_permute(index, tmp.i)
              : 0;
  return tmp.u;
}

__device__ static inline float __hip_ds_permutef(int index, float src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_permute)
              ? __builtin_amdgcn_ds_permute(index, tmp.i)
              : 0;
  return tmp.f;
}

#define __hip_ds_swizzle(src, pattern) __hip_ds_swizzle_N<(pattern)>((src))
#define __hip_ds_swizzlef(src, pattern) __hip_ds_swizzlef_N<(pattern)>((src))

template <int pattern> __device__ static inline unsigned __hip_ds_swizzle_N(unsigned int src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_swizzle)
              ? __builtin_amdgcn_ds_swizzle(tmp.i, pattern)
              : 0;
  return tmp.u;
}

template <int pattern> __device__ static inline float __hip_ds_swizzlef_N(float src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_swizzle)
              ? __builtin_amdgcn_ds_swizzle(tmp.i, pattern)
              : 0;
  return tmp.f;
}

#define __hip_move_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)                             \
  __hip_move_dpp_N<(dpp_ctrl), (row_mask), (bank_mask), (bound_ctrl)>((src))

template <int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl>
__device__ static inline int __hip_move_dpp_N(int src) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mov_dpp))
    return __builtin_amdgcn_mov_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
  __builtin_trap();
}

inline __device__ const struct {
  __device__ __attribute__((always_inline, const)) operator int() const noexcept {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_wavefrontsize))
      return __builtin_amdgcn_wavefrontsize();
    __builtin_trap();
  }
} warpSize{};

// warp vote function __all __any __ballot
__device__ inline int __all(int predicate) { return __ockl_wfall_i32(predicate); }

__device__ inline int __any(int predicate) { return __ockl_wfany_i32(predicate); }

__device__ inline unsigned long long int __ballot(int predicate) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ballot_w64))
    return __builtin_amdgcn_ballot_w64(predicate);
  __builtin_trap();
}

__device__ inline unsigned long long int __ballot64(int predicate) { return __ballot(predicate); }

// See amd_warp_sync_functions.h for an explanation of this preprocessor flag.
#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
// Since threads in a wave do not make independent progress, __activemask()
// always returns the exact active mask, i.e, all active threads in the wave.
__device__ inline unsigned long long __activemask() { return __ballot(true); }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS

__device__ static inline unsigned int __lane_id() {
  if (static_cast<int>(warpSize) == 32) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_lo))
      return __builtin_amdgcn_mbcnt_lo(-1, 0);
    __builtin_trap();
  }
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_hi))
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_lo))
      return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
  __builtin_trap();
}

__device__ inline int __shfl(MAYBE_UNDEF int var, int src_lane, int width = warpSize) {
#if __has_builtin(__builtin_amdgcn_permlane_bcast)
  // v_permlane_bcast_b32 reads VGPR[laneGroupBase + (src_lane[5:0] & (width - 1))], which is the
  // index computed below. width is only masked to 6 bits, and width - 1 is never wider than that.
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlane_bcast))
    return __builtin_amdgcn_permlane_bcast(var, src_lane, width);
#endif
  int self = __lane_id();
  int index = (src_lane & (width - 1)) + (self & ~(width - 1));
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}
__device__ inline unsigned int __shfl(MAYBE_UNDEF unsigned int var, int src_lane,
                                      int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.u;
}
__device__ inline float __shfl(MAYBE_UNDEF float var, int src_lane, int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.f;
}
__device__ inline double __shfl(MAYBE_UNDEF double var, int src_lane, int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl(MAYBE_UNDEF long var, int src_lane, int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl(static_cast<int>(var), src_lane, width));
#endif
}
__device__ inline unsigned long __shfl(MAYBE_UNDEF unsigned long var, int src_lane,
                                       int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl(static_cast<unsigned int>(var), src_lane, width));
#endif
}
__device__ inline long long __shfl(MAYBE_UNDEF long long var, int src_lane, int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline unsigned long long __shfl(MAYBE_UNDEF unsigned long long var, int src_lane,
                                            int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

__device__ inline int __shfl_up(MAYBE_UNDEF int var, unsigned int lane_delta,
                                int width = warpSize) {
#if __has_builtin(__builtin_amdgcn_permlane_up)
  // v_permlane_up_b32 returns the lane's own value when its position in the lane group is below
  // lane_delta, which is the same select as below. It additionally clamps lane_delta to the group
  // width, so an out-of-range delta yields the lane's own value rather than the wrapped index the
  // unsigned arithmetic below produces; that input is outside the shuffle contract either way.
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlane_up))
    return __builtin_amdgcn_permlane_up(var, lane_delta, width);
#endif
  int self = __lane_id();
  int index = self - lane_delta;
  index = (index < (self & ~(width - 1))) ? self : index;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}
__device__ inline unsigned int __shfl_up(MAYBE_UNDEF unsigned int var, unsigned int lane_delta,
                                         int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.u;
}
__device__ inline float __shfl_up(MAYBE_UNDEF float var, unsigned int lane_delta,
                                  int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.f;
}
__device__ inline double __shfl_up(MAYBE_UNDEF double var, unsigned int lane_delta,
                                   int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl_up(MAYBE_UNDEF long var, unsigned int lane_delta,
                                 int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl_up(static_cast<int>(var), lane_delta, width));
#endif
}

__device__ inline unsigned long __shfl_up(MAYBE_UNDEF unsigned long var, unsigned int lane_delta,
                                          int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl_up(static_cast<unsigned int>(var), lane_delta, width));
#endif
}

__device__ inline long long __shfl_up(MAYBE_UNDEF long long var, unsigned int lane_delta,
                                      int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");
  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

__device__ inline unsigned long long __shfl_up(MAYBE_UNDEF unsigned long long var,
                                               unsigned int lane_delta, int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");
  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

__device__ inline int __shfl_down(MAYBE_UNDEF int var, unsigned int lane_delta,
                                  int width = warpSize) {
#if __has_builtin(__builtin_amdgcn_permlane_down)
  // v_permlane_down_b32 returns the lane's own value when its position in the lane group plus
  // lane_delta reaches the group width, which is the same select as below. As with __shfl_up it
  // clamps lane_delta to the group width, so an out-of-range delta yields the lane's own value.
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlane_down))
    return __builtin_amdgcn_permlane_down(var, lane_delta, width);
#endif
  int self = __lane_id();
  int index = self + lane_delta;
  index = (int)((self & (width - 1)) + lane_delta) >= width ? self : index;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}
__device__ inline unsigned int __shfl_down(MAYBE_UNDEF unsigned int var, unsigned int lane_delta,
                                           int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.u;
}
__device__ inline float __shfl_down(MAYBE_UNDEF float var, unsigned int lane_delta,
                                    int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.f;
}
__device__ inline double __shfl_down(MAYBE_UNDEF double var, unsigned int lane_delta,
                                     int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl_down(MAYBE_UNDEF long var, unsigned int lane_delta,
                                   int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl_down(static_cast<int>(var), lane_delta, width));
#endif
}
__device__ inline unsigned long __shfl_down(MAYBE_UNDEF unsigned long var, unsigned int lane_delta,
                                            int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl_down(static_cast<unsigned int>(var), lane_delta, width));
#endif
}
__device__ inline long long __shfl_down(MAYBE_UNDEF long long var, unsigned int lane_delta,
                                        int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");
  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline unsigned long long __shfl_down(MAYBE_UNDEF unsigned long long var,
                                                 unsigned int lane_delta, int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");
  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

// True when (lane_mask, width) are both known at compile time and the xor cannot leave the
// width-aligned lane group. Every fixed-permutation back end below is gated on this.
//
// Both operands need their own __builtin_constant_p, and neither test is redundant:
//
//  - lane_mask, because the back ends select on it with a switch whose arms take it as a
//    template argument. Testing only the comparison is not enough - the optimiser will report
//    `lane_mask > 0 && lane_mask < width` as constant once it can bound the range, even when
//    lane_mask is still a register. The switch then survives as a runtime binary search
//    (measured: a 54-instruction reduction became 299).
//  - width, because otherwise the comparison survives as a runtime branch. If width is
//    divergent, that branch is divergent too, and the cross-lane instruction inside it runs
//    under a partial exec mask, reading lanes that are switched off. On gfx942 that silently
//    corrupts half the wave.
//
// Requiring both folds the condition away entirely, so neither failure is reachable by
// construction. It costs nothing in practice: warpSize folds to a literal, so the default
// `width` argument still takes the fast path.
//
// This must be expanded directly into an `if` condition. Assigning it to a local first makes
// clang evaluate __builtin_constant_p in the front end - where lane_mask is still a parameter -
// so it always yields false and every caller falls back to ds_bpermute.
#define __HIP_SHFL_XOR_FIXED(lane_mask, width)                                                     \
  (__builtin_constant_p(lane_mask) && __builtin_constant_p(width) && (lane_mask) > 0 &&            \
   (lane_mask) < (width))

// DPP is an operand modifier rather than an instruction, so GCNDPPCombine folds the move into
// the consuming VALU op and a shuffle that feeds arithmetic costs nothing at all. The fold needs
// row_mask and bank_mask fully enabled and either bound_ctrl set or old == 0 (GCNDPPCombine.cpp,
// "Combining rules"); __hip_move_dpp_N passes poison as old, so bound_ctrl must be true. That is
// also the form whose behaviour matches ds_bpermute when the source lane is inactive - both
// yield 0 - so this does not change the (undefined) partial-exec corner.
#define __HIP_SHFL_XOR_DPP(dpp_ctrl) __hip_move_dpp_N<(dpp_ctrl), 0xf, 0xf, true>(var)

// ds_swizzle_b32 in BITMASK_PERM mode: and_mask in [4:0], or_mask in [9:5], xor_mask in [14:10]
// (SIDefines.h). A pure xor by N is therefore (N << 10) | 0x1f. Groups are 32 lanes wide, so
// this covers every mask below 32 regardless of wavefront size.
#define __HIP_SHFL_XOR_SWIZZLE(lane_mask)                                                          \
  static_cast<int>(__hip_ds_swizzle_N<((lane_mask) << 10) | 0x1f>(static_cast<unsigned int>(var)))

__device__ inline int __shfl_xor(MAYBE_UNDEF int var, int lane_mask, int width = warpSize) {
  if (__HIP_SHFL_XOR_FIXED(lane_mask, width)) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_update_dpp)) {
#if __has_builtin(__builtin_amdgcn_permlane16)
      // row_xmask:N is "exchange with lane ^ N inside the row of 16", which is exactly this
      // function. It is gfx10 and later, but the restriction is on the dpp_ctrl value rather
      // than on the builtin, and it is only diagnosed in the backend, so is_invocable cannot
      // query it. permlane16 carries the matching "gfx10-insts" target feature.
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlane16)) {
        switch (lane_mask) {
          case 1: return __HIP_SHFL_XOR_DPP(0x161);
          case 2: return __HIP_SHFL_XOR_DPP(0x162);
          case 3: return __HIP_SHFL_XOR_DPP(0x163);
          case 4: return __HIP_SHFL_XOR_DPP(0x164);
          case 5: return __HIP_SHFL_XOR_DPP(0x165);
          case 6: return __HIP_SHFL_XOR_DPP(0x166);
          case 7: return __HIP_SHFL_XOR_DPP(0x167);
          case 8: return __HIP_SHFL_XOR_DPP(0x168);
          case 9: return __HIP_SHFL_XOR_DPP(0x169);
          case 10: return __HIP_SHFL_XOR_DPP(0x16a);
          case 11: return __HIP_SHFL_XOR_DPP(0x16b);
          case 12: return __HIP_SHFL_XOR_DPP(0x16c);
          case 13: return __HIP_SHFL_XOR_DPP(0x16d);
          case 14: return __HIP_SHFL_XOR_DPP(0x16e);
          case 15: return __HIP_SHFL_XOR_DPP(0x16f);
        }
      } else
#endif
      {
        // DPP16 has no general xor mode before gfx10. The masks it can still express are the
        // modes that happen to be involutions - a quad swap, and the two row mirrors.
        switch (lane_mask) {
          case 1: return __HIP_SHFL_XOR_DPP(0x0b1);   // quad_perm:[1,0,3,2]
          case 2: return __HIP_SHFL_XOR_DPP(0x04e);   // quad_perm:[2,3,0,1]
          case 3: return __HIP_SHFL_XOR_DPP(0x01b);   // quad_perm:[3,2,1,0]
          case 7: return __HIP_SHFL_XOR_DPP(0x141);   // row_half_mirror
          case 15: return __HIP_SHFL_XOR_DPP(0x140);  // row_mirror
        }
      }
    }
  }
#if __has_builtin(__builtin_amdgcn_permlane_xor)
  // v_permlane_xor_b32 computes (lane ^ lane_mask) and falls back to the lane's own value when
  // that leaves the lane group, which is the same select as below. Its extra
  // lane_mask >= wavefrontSize case also yields the lane's own value, matching the select below
  // because width never exceeds the wavefront size.
  //
  // It is a real instruction, so unlike DPP it cannot be absorbed by the consumer - hence it
  // sits below the DPP cases. It still beats ds_swizzle for what DPP could not reach, because
  // it stays on the VALU pipe, and unlike both it accepts a runtime lane_mask.
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlane_xor))
    return __builtin_amdgcn_permlane_xor(var, lane_mask, width);
#endif
  if (__HIP_SHFL_XOR_FIXED(lane_mask, width) && lane_mask < 32) {
    switch (lane_mask) {
      case 1: return __HIP_SHFL_XOR_SWIZZLE(1);
      case 2: return __HIP_SHFL_XOR_SWIZZLE(2);
      case 3: return __HIP_SHFL_XOR_SWIZZLE(3);
      case 4: return __HIP_SHFL_XOR_SWIZZLE(4);
      case 5: return __HIP_SHFL_XOR_SWIZZLE(5);
      case 6: return __HIP_SHFL_XOR_SWIZZLE(6);
      case 7: return __HIP_SHFL_XOR_SWIZZLE(7);
      case 8: return __HIP_SHFL_XOR_SWIZZLE(8);
      case 9: return __HIP_SHFL_XOR_SWIZZLE(9);
      case 10: return __HIP_SHFL_XOR_SWIZZLE(10);
      case 11: return __HIP_SHFL_XOR_SWIZZLE(11);
      case 12: return __HIP_SHFL_XOR_SWIZZLE(12);
      case 13: return __HIP_SHFL_XOR_SWIZZLE(13);
      case 14: return __HIP_SHFL_XOR_SWIZZLE(14);
      case 15: return __HIP_SHFL_XOR_SWIZZLE(15);
      case 16: return __HIP_SHFL_XOR_SWIZZLE(16);
      case 17: return __HIP_SHFL_XOR_SWIZZLE(17);
      case 18: return __HIP_SHFL_XOR_SWIZZLE(18);
      case 19: return __HIP_SHFL_XOR_SWIZZLE(19);
      case 20: return __HIP_SHFL_XOR_SWIZZLE(20);
      case 21: return __HIP_SHFL_XOR_SWIZZLE(21);
      case 22: return __HIP_SHFL_XOR_SWIZZLE(22);
      case 23: return __HIP_SHFL_XOR_SWIZZLE(23);
      case 24: return __HIP_SHFL_XOR_SWIZZLE(24);
      case 25: return __HIP_SHFL_XOR_SWIZZLE(25);
      case 26: return __HIP_SHFL_XOR_SWIZZLE(26);
      case 27: return __HIP_SHFL_XOR_SWIZZLE(27);
      case 28: return __HIP_SHFL_XOR_SWIZZLE(28);
      case 29: return __HIP_SHFL_XOR_SWIZZLE(29);
      case 30: return __HIP_SHFL_XOR_SWIZZLE(30);
      case 31: return __HIP_SHFL_XOR_SWIZZLE(31);
    }
  }
  // General case: a runtime lane_mask, a runtime width, or a mask that reaches outside the
  // 32-lane group that swizzle can address.
  int self = __lane_id();
  int index = self ^ lane_mask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}

#undef __HIP_SHFL_XOR_FIXED
#undef __HIP_SHFL_XOR_DPP
#undef __HIP_SHFL_XOR_SWIZZLE
__device__ inline unsigned int __shfl_xor(MAYBE_UNDEF unsigned int var, int lane_mask,
                                          int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.u;
}
__device__ inline float __shfl_xor(MAYBE_UNDEF float var, int lane_mask, int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.f;
}
__device__ inline double __shfl_xor(MAYBE_UNDEF double var, int lane_mask, int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl_xor(MAYBE_UNDEF long var, int lane_mask, int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl_xor(static_cast<int>(var), lane_mask, width));
#endif
}
__device__ inline unsigned long __shfl_xor(MAYBE_UNDEF unsigned long var, int lane_mask,
                                           int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl_xor(static_cast<unsigned int>(var), lane_mask, width));
#endif
}
__device__ inline long long __shfl_xor(MAYBE_UNDEF long long var, int lane_mask,
                                       int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");
  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline unsigned long long __shfl_xor(MAYBE_UNDEF unsigned long long var, int lane_mask,
                                                int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");
  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

#endif
