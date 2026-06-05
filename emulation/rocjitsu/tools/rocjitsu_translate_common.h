// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TOOLS_ROCJITSU_TRANSLATE_COMMON_H_
#define ROCJITSU_TOOLS_ROCJITSU_TRANSLATE_COMMON_H_

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::tools {

[[nodiscard]] inline std::optional<rj_code_arch_t> parse_arch(std::string_view value) {
  if (value == "cdna1")
    return ROCJITSU_CODE_ARCH_CDNA1;
  if (value == "cdna2")
    return ROCJITSU_CODE_ARCH_CDNA2;
  if (value == "cdna3")
    return ROCJITSU_CODE_ARCH_CDNA3;
  if (value == "cdna4" || value == "gfx950")
    return ROCJITSU_CODE_ARCH_CDNA4;
  if (value == "rdna1")
    return ROCJITSU_CODE_ARCH_RDNA1;
  if (value == "rdna2")
    return ROCJITSU_CODE_ARCH_RDNA2;
  if (value == "rdna3")
    return ROCJITSU_CODE_ARCH_RDNA3;
  if (value == "rdna3_5")
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  if (value == "rdna4" || value == "gfx1200" || value == "gfx1201")
    return ROCJITSU_CODE_ARCH_RDNA4;
  if (value == "gfx1250")
    return ROCJITSU_CODE_ARCH_GFX1250;
  return std::nullopt;
}

[[nodiscard]] inline std::optional<uint32_t> parse_mach(std::string_view value) {
  if (value.empty())
    return 0;
  if (value == "gfx1200")
    return EF_AMDGPU_MACH_AMDGCN_GFX1200;
  if (value == "gfx1201")
    return EF_AMDGPU_MACH_AMDGCN_GFX1201;
  if (value == "gfx1250")
    return EF_AMDGPU_MACH_AMDGCN_GFX1250;

  try {
    size_t consumed = 0;
    unsigned long parsed = std::stoul(std::string(value), &consumed, 0);
    if (consumed != value.size() || parsed > std::numeric_limits<uint32_t>::max())
      return std::nullopt;
    return static_cast<uint32_t>(parsed);
  } catch (...) {
    return std::nullopt;
  }
}

[[nodiscard]] inline std::optional<std::vector<uint8_t>>
read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return std::nullopt;

  const auto size = input.tellg();
  if (size < 0)
    return std::nullopt;

  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.seekg(0);
  if (!bytes.empty())
    input.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!input && !bytes.empty())
    return std::nullopt;
  return bytes;
}

inline bool write_file(const std::filesystem::path &path, const std::vector<uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return false;
  if (!bytes.empty())
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(output);
}

} // namespace rocjitsu::tools

#endif // ROCJITSU_TOOLS_ROCJITSU_TRANSLATE_COMMON_H_
