// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <span>

namespace rocjitsu::consan {

/// @brief Build-owned standalone probe image for one concrete target/wave mode.
[[nodiscard]] std::span<const uint8_t> lds_probe_image(uint32_t target, uint32_t wave_size);

} // namespace rocjitsu::consan
