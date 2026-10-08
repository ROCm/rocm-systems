// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_ARITHMETIC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_ARITHMETIC_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/extract.h"
#include "util/data_types.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rocjitsu::amdgpu {

inline uint32_t wmma_c_modifier(uint32_t neg, uint32_t neg_hi) {
  return ((neg >> 2) & 0x1u) | (((neg_hi >> 2) & 0x1u) << 1);
}

inline float apply_wmma_c_modifier(float acc, uint32_t c_modifier) {
  if (c_modifier & 0x2u)
    acc = std::fabs(acc);
  if (c_modifier & 0x1u)
    acc = -acc;
  return acc;
}

/// FMA step with host-independent NaN source priority. Matrix instructions use
/// fused arithmetic, but scalar and packed host FMAs may select different NaN
/// operands after lowering. RocJITsu deterministically selects src0, then src1,
/// then the accumulator, quieting the selected signaling NaN.
inline float matrix_fma(float src0, float src1, float acc) {
  const float result = std::fma(src0, src1, acc);
  if (!std::isnan(result))
    return result;
  auto quiet = [](float value) {
    return std::bit_cast<float>(std::bit_cast<uint32_t>(value) | 0x00400000u);
  };
  if (std::isnan(src0))
    return quiet(src0);
  if (std::isnan(src1))
    return quiet(src1);
  if (std::isnan(acc))
    return quiet(acc);
  // Use RocJITsu's deterministic negative canonical NaN for invalid operations
  // such as Inf*0 and Inf + -Inf.
  return std::bit_cast<float>(0xFFC00000u);
}

// Preserve the focused helper name used by the BF16F32 contract tests and by
// callers that need the same deterministic matrix-FMA NaN behavior.
inline float wmma_bf16f32_fma(float src0, float src1, float acc) {
  return matrix_fma(src0, src1, acc);
}

template <typename Extract>
inline constexpr bool swmmac_product_is_exact_in_f32 =
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractF16> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractFp8Ocp> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractBf8Ocp> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractFp8Fnuz> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractBf8Fnuz> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractFp8> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractBf8> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractFp4> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractFp6> ||
    std::is_same_v<std::remove_cvref_t<Extract>, ExtractBf6>;

/// Use the cheaper split scalar MAC when both decoded inputs have a product
/// that is exactly representable in F32. F16 and narrower formats have at most
/// 11 significand bits and an exponent range whose products stay within F32,
/// so rounding the product separately cannot differ from an FMA. BF16 shares
/// F32's exponent range and must remain fused: its product can overflow before
/// a cancelling add. NaN results are replayed through matrix_fma by the caller.
template <typename ExtractA, typename ExtractB>
inline float swmmac_scalar_mac(float src0, float src1, float acc) {
  if constexpr (swmmac_product_is_exact_in_f32<ExtractA> &&
                swmmac_product_is_exact_in_f32<ExtractB>)
    return acc + src0 * src1;
  return std::fma(src0, src1, acc);
}

/// Re-evaluate an exceptional sparse matrix result with deterministic NaN
/// propagation. Keep this rare path out of the instruction executor so its
/// layout and extraction helpers remain small enough for the compiler to
/// inline and constant-propagate on ordinary finite inputs.
template <typename ExtractA, typename ExtractB>
RJ_NOINLINE float swmmac_replay_nan_vgpr(auto &cu, uint32_t M, uint32_t N, uint32_t K,
                                         uint32_t a_bits, uint32_t b_bits, uint32_t s0, uint32_t s1,
                                         uint32_t index_base, uint32_t index_entries,
                                         uint32_t index_key, uint32_t row, uint32_t col,
                                         ExtractA ea, ExtractB eb, float acc, uint32_t wave_size) {
  const uint32_t compressed_k = K / 2;
  for (uint32_t ck = 0; ck < compressed_k; ++ck) {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, a_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    const uint32_t dense_k = swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
    const auto a_loc = swmmac_a_input_loc(wave_size, M, K, row, ck, a_bits);
    const auto b_loc = swmmac_b_input_loc(wave_size, N, K, col, dense_k, b_bits);
    acc = matrix_fma(ea(cu, s0, a_loc), eb(cu, s1, b_loc), acc);
  }
  return acc;
}

RJ_NOINLINE inline float swmmac_replay_nan_buffers(const float *a, const float *b,
                                                   uint32_t compressed_k, uint32_t stride,
                                                   uint32_t col, float acc) {
  for (uint32_t ck = 0; ck < compressed_k; ++ck)
    acc = matrix_fma(a[ck], b[ck * stride + col], acc);
  return acc;
}

/// Round a matrix accumulator under MODE.FP16_OVFL. CDNA5 WMMA/SWMMAC
/// results of 16 bits or fewer saturate every infinity to signed MAX when the
/// mode bit is set, including infinity produced by finite arithmetic.
[[gnu::always_inline]] inline uint16_t wmma_round_f16(float val, bool fp16_ovfl) {
  if (fp16_ovfl && std::isinf(val))
    return std::signbit(val) ? uint16_t{0xFBFF} : uint16_t{0x7BFF};
  return util::f32_to_f16_mode(val, fp16_ovfl);
}

[[gnu::always_inline]] inline uint16_t wmma_round_bf16_rne(float val, bool fp16_ovfl) {
  if (fp16_ovfl && std::isinf(val))
    return std::signbit(val) ? uint16_t{0xFF7F} : uint16_t{0x7F7F};
  return util::f32_to_bf16_rne_mode(val, fp16_ovfl);
}

/// Pack an integer accumulator, saturating to int32 range when clamp is enabled.
inline uint32_t pack_i32_acc(int64_t acc, bool clamp) {
  if (clamp) {
    acc = std::clamp(acc, static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                     static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  }
  return static_cast<uint32_t>(acc);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_ARITHMETIC_H_
