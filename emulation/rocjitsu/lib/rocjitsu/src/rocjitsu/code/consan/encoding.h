// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::consan::wire {

[[nodiscard]] inline std::optional<uint32_t> read_u32(std::span<const uint8_t> bytes,
                                                      size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4)
    return std::nullopt;
  uint32_t value = 0;
  for (unsigned index = 0; index < 4; ++index)
    value |= uint32_t{bytes[offset + index]} << (8 * index);
  return value;
}

[[nodiscard]] inline std::optional<uint64_t> read_u64(std::span<const uint8_t> bytes,
                                                      size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    return std::nullopt;
  return uint64_t{*read_u32(bytes, offset)} | (uint64_t{*read_u32(bytes, offset + 4)} << 32);
}

inline bool write_u32(std::span<uint8_t> bytes, size_t offset, uint32_t value) {
  if (offset > bytes.size() || bytes.size() - offset < 4)
    return false;
  for (unsigned index = 0; index < 4; ++index)
    bytes[offset + index] = static_cast<uint8_t>(value >> (8 * index));
  return true;
}

inline bool write_u64(std::span<uint8_t> bytes, size_t offset, uint64_t value) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    return false;
  return write_u32(bytes, offset, static_cast<uint32_t>(value)) &&
         write_u32(bytes, offset + 4, static_cast<uint32_t>(value >> 32));
}

} // namespace rocjitsu::consan::wire
