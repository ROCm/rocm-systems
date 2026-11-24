// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace rocjitsu::consan::test {

inline std::vector<uint32_t> flat_half_output(std::string_view target, uint32_t wave) {
  // CDNA SRAM ECC clears the inactive half of the destination.
  const bool preserve = target == "gfx1100" || target == "gfx1201";
  const uint32_t low_address = 1014 + 8 * wave;
  return {0x81233344u,
          0x8123u,
          0xffff8123u,
          preserve ? 0xdead8123u : 0x8123u,
          0x81230000u | (preserve ? low_address : 0u),
          0xabcd3344u};
}

} // namespace rocjitsu::consan::test
