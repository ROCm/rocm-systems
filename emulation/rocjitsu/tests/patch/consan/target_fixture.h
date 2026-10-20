// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/target.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace rocjitsu::consan::test {

struct TargetMode {
  const char *target;
  uint32_t wave_size;
};

inline constexpr std::array kTargetModes = {
    TargetMode{"gfx942", 64},  TargetMode{"gfx950", 64},  TargetMode{"gfx1100", 32},
    TargetMode{"gfx1100", 64}, TargetMode{"gfx1201", 32}, TargetMode{"gfx1201", 64},
    TargetMode{"gfx1250", 32},
};

inline std::filesystem::path target_fixture_path(TargetMode mode, std::string_view fixture) {
#ifdef CONSAN_FIXTURE_DIR
  return std::filesystem::path(CONSAN_FIXTURE_DIR) / "targets" / mode.target /
         ("wave" + std::to_string(mode.wave_size)) / (std::string(fixture) + ".hsaco");
#else
  (void)mode;
  (void)fixture;
  return {};
#endif
}

inline std::vector<uint8_t> read_fixture(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

} // namespace rocjitsu::consan::test
