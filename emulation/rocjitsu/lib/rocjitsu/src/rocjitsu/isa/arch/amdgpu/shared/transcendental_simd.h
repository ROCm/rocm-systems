// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file transcendental_simd.h
/// @brief Exact integer batch evaluation of F32 transcendental mappings.

#pragma once

#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu::transcendental {
enum class F32Operation { LOG, EXP, SIN, COS, RCP, RSQ, SQRT };

/// @brief Whether the host supports the exact eight-lane integer implementation.
bool f32_simd_available();

/// @brief Evaluate complete eight-lane chunks, preserving host floating-point state.
/// Input and output may be identical; otherwise they must not overlap.
/// Other counts use the same scalar mapping. RCP/RSQ always flush denormals
/// and follow quiet_snan for signaling NaNs.
void evaluate_f32_simd(F32Operation operation, const uint32_t *input, uint32_t *output,
                       std::size_t count, uint32_t denorm_mode, bool quiet_snan);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
namespace detail {
// Internal batch backend; the caller verifies the required host features.
void evaluate_reciprocal_f32_simd(bool rsq, const uint32_t *input, uint32_t *output,
                                  std::size_t count, bool quiet_snan);
} // namespace detail
#endif
} // namespace rocjitsu::amdgpu::transcendental
