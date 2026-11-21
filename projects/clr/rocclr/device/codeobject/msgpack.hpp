/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace amd::codeobject {

/**
 * Pull reader for MessagePack data, as used by the code object metadata note
 * (NT_AMDGPU_HSA_METADATA). It reads one value at a time from a byte range, checks every read
 * against the range, and returns strings as views into it. Nothing is copied or allocated.
 */
class MsgPackReader {
 public:
  MsgPackReader(const char* data, size_t size) : cur_(data), end_(data + size) {}

  //! Reads a map header; @p count is the number of key/value pairs that follow.
  bool readMapSize(uint32_t* count);

  //! Reads an array header; @p count is the number of values that follow.
  bool readArraySize(uint32_t* count);

  //! Reads a string.
  bool readString(std::string_view* value);

  //! Reads an unsigned integer, or a signed one that isn't negative.
  bool readUint(uint64_t* value);

  //! Reads a boolean, or an integer (non-zero is true).
  bool readBool(bool* value);

  //! Skips one value, including the contents of maps and arrays.
  bool skip();

  //! @returns The current read position.
  const char* position() const { return cur_; }

  //! @returns The bytes left to read. Every value takes at least one, so this bounds the
  //! element count of any container still to come; cap reserve() calls with it.
  size_t remaining() const { return static_cast<size_t>(end_ - cur_); }

 private:
  bool readBigEndian(size_t bytes, uint64_t* value);
  bool readContainerSize(uint8_t fixBase, uint8_t fixMask, uint8_t tag16, uint8_t tag32,
                         uint32_t* count);
  bool advance(uint64_t bytes);

  const char* cur_;  //!< Next byte to read
  const char* end_;  //!< One past the last readable byte
};

}  // namespace amd::codeobject
