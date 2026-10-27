// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_FILE_IO_H_
#define ROCJITSU_CODE_FILE_IO_H_

#include "util/result.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace rocjitsu::detail {

/// @brief Read the contents of a regular file.
/// @returns The file contents, or failure if the path is missing, is not a regular
/// file, or an I/O error occurs. An empty regular file succeeds with an empty vector.
/// Reject files exceeding max_bytes before allocating the input buffer.
/// @throws std::bad_alloc if memory allocation fails.
util::FailureOr<std::vector<char>>
read_file_bytes(std::string_view path, size_t max_bytes = std::numeric_limits<size_t>::max());

bool fits_in_bounds(uint64_t offset, uint64_t size, size_t limit);

template <typename T> bool read_value(const char *data, size_t size, size_t &offset, T &value) {
  if (!fits_in_bounds(offset, sizeof(T), size))
    return false;
  std::memcpy(&value, data + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

} // namespace rocjitsu::detail

#endif // ROCJITSU_CODE_FILE_IO_H_
