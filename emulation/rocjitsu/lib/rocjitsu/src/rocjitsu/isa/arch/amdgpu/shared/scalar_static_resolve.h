// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_STATIC_RESOLVE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_STATIC_RESOLVE_H_

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {

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
