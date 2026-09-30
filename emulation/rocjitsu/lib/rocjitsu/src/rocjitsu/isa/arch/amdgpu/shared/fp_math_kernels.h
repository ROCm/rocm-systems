// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu::fp_math {

// Bit-pattern packs are an internal backend boundary. Register storage and
// target-dependent vector types stay on their respective sides of it.
struct F32Bits8 {
  std::array<std::uint32_t, 8> lane;
};

struct F32Bits16 {
  std::array<std::uint32_t, 16> lane;
};

F32Bits8 exp_v3(F32Bits8 input, bool quiet_snan) noexcept;
F32Bits8 log_v3(F32Bits8 input, bool quiet_snan) noexcept;
F32Bits8 exp_v4x8(F32Bits8 input, bool quiet_snan) noexcept;
F32Bits8 log_v4x8(F32Bits8 input, bool quiet_snan) noexcept;
F32Bits16 exp_v4x16(F32Bits16 input, bool quiet_snan) noexcept;
F32Bits16 log_v4x16(F32Bits16 input, bool quiet_snan) noexcept;

} // namespace rocjitsu::amdgpu::fp_math
