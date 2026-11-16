// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_STATIC_RESOLVE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_STATIC_RESOLVE_H_

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {

/// True for the nine floating inline selectors (240–248). Literals use selector
/// 255, even when their payload has the same value as an inline selector.
inline bool is_inline_float_src(uint32_t selector) { return selector >= 240u && selector <= 248u; }

/// Numeric F16 bits for floating inline selectors 240–248. Registers, literals
/// and integer inlines return nullopt so callers preserve their raw carriers.
[[nodiscard]] inline std::optional<uint16_t> resolve_inline_float16_statically(int ev) {
  constexpr uint16_t values[] = {0x3800u, 0xB800u, 0x3C00u, 0xBC00u, 0x4000u,
                                 0xC000u, 0x4400u, 0xC400u, 0x3118u};
  if (is_inline_float_src(ev))
    return values[ev - 240];
  return std::nullopt;
}

/// Numeric BF16 bits for floating inline selectors 240–248. Registers, literals
/// and integer inlines return nullopt so callers preserve their raw carriers.
[[nodiscard]] inline std::optional<uint16_t> resolve_inline_bfloat16_statically(int ev) {
  constexpr uint16_t values[] = {0x3F00u, 0xBF00u, 0x3F80u, 0xBF80u, 0x4000u,
                                 0xC000u, 0x4080u, 0xC080u, 0x3E22u};
  if (is_inline_float_src(ev))
    return values[ev - 240];
  return std::nullopt;
}

// Wavefront-free subset of resolve_src_scalar: the value of an inline constant
// (small integers 0..64 / -1..-16 and the inline float constants), or nullopt
// for any other encoding value. Negative inline integers are sign-extended to
// 64 bits so 64-bit consumers see all-ones.
//
// This header is intentionally free of any rocjitsu/vm/ dependency so that
// model-side code (e.g. Operand::const_value) can use it without pulling in the
// simulator/execution layer. Keep the encoding-value handling here in sync with
// resolve_src_scalar() in scalar_operand_resolve.h, which includes this header.
inline std::optional<uint64_t> resolve_src_scalar_statically(int ev) {
  if (ev >= 128 && ev <= 192)
    return static_cast<uint64_t>(ev - 128);
  if (ev >= 193 && ev <= 208)
    return static_cast<uint64_t>(static_cast<int64_t>(-(ev - 192)));
  if (ev == 240)
    return static_cast<uint64_t>(0x3F000000u); // 0.5f
  if (ev == 241)
    return static_cast<uint64_t>(0xBF000000u); // -0.5f
  if (ev == 242)
    return static_cast<uint64_t>(0x3F800000u); // 1.0f
  if (ev == 243)
    return static_cast<uint64_t>(0xBF800000u); // -1.0f
  if (ev == 244)
    return static_cast<uint64_t>(0x40000000u); // 2.0f
  if (ev == 245)
    return static_cast<uint64_t>(0xC0000000u); // -2.0f
  if (ev == 246)
    return static_cast<uint64_t>(0x40800000u); // 4.0f
  if (ev == 247)
    return static_cast<uint64_t>(0xC0800000u); // -4.0f
  if (ev == 248)
    return static_cast<uint64_t>(0x3E22F983u); // 1/(2*pi)
  return std::nullopt;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_STATIC_RESOLVE_H_
