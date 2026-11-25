// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::consan::wire {

[[nodiscard]] inline std::optional<uint32_t> read_u32(std::span<const uint8_t> bytes,
                                                      size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4) {
    return std::nullopt;
  }
  uint32_t value = 0;
  for (unsigned index = 0; index < 4; ++index) {
    value |= uint32_t{bytes[offset + index]} << (8 * index);
  }
  return value;
}

[[nodiscard]] inline std::optional<uint64_t> read_u64(std::span<const uint8_t> bytes,
                                                      size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8) {
    return std::nullopt;
  }
  return uint64_t{*read_u32(bytes, offset)} | (uint64_t{*read_u32(bytes, offset + 4)} << 32);
}

inline bool write_u32(std::span<uint8_t> bytes, size_t offset, uint32_t value) {
  if (offset > bytes.size() || bytes.size() - offset < 4) {
    return false;
  }
  for (unsigned index = 0; index < 4; ++index) {
    bytes[offset + index] = static_cast<uint8_t>(value >> (8 * index));
  }
  return true;
}

inline bool write_u64(std::span<uint8_t> bytes, size_t offset, uint64_t value) {
  if (offset > bytes.size() || bytes.size() - offset < 8) {
    return false;
  }
  return write_u32(bytes, offset, static_cast<uint32_t>(value)) &&
         write_u32(bytes, offset + 4, static_cast<uint32_t>(value >> 32));
}

/// @brief Owning little-endian encoder for variable-length ConSan facts.
class Encoder {
public:
  void u32(uint32_t value) {
    const size_t offset = bytes_.size();
    bytes_.resize(offset + 4);
    write_u32(bytes_, offset, value);
  }
  void u64(uint64_t value) {
    const size_t offset = bytes_.size();
    bytes_.resize(offset + 8);
    write_u64(bytes_, offset, value);
  }
  void bytes(std::span<const uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void string(std::string_view value) {
    u32(static_cast<uint32_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  std::vector<uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<uint8_t> bytes_;
};

/// @brief Bounds-checked reader for little-endian ConSan facts.
class Decoder {
public:
  explicit Decoder(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  bool u32(uint32_t &value) {
    const std::optional<uint32_t> read = read_u32(bytes_, offset_);
    if (!read) {
      return false;
    }
    value = *read;
    offset_ += 4;
    return true;
  }
  bool u64(uint64_t &value) {
    const std::optional<uint64_t> read = read_u64(bytes_, offset_);
    if (!read) {
      return false;
    }
    value = *read;
    offset_ += 8;
    return true;
  }
  bool bytes(std::span<uint8_t> value) {
    if (value.size() > remaining()) {
      return false;
    }
    std::copy_n(bytes_.begin() + offset_, value.size(), value.begin());
    offset_ += value.size();
    return true;
  }
  bool string(std::string &value, uint32_t limit) {
    uint32_t size = 0;
    if (!u32(size) || size > limit || size > remaining()) {
      return false;
    }
    value.assign(reinterpret_cast<const char *>(bytes_.data() + offset_), size);
    offset_ += size;
    return true;
  }
  std::span<const uint8_t> remaining_bytes() const { return bytes_.subspan(offset_); }
  bool skip(uint64_t size) {
    if (size > remaining()) {
      return false;
    }
    offset_ += size;
    return true;
  }
  size_t remaining() const { return bytes_.size() - offset_; }

private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

} // namespace rocjitsu::consan::wire
