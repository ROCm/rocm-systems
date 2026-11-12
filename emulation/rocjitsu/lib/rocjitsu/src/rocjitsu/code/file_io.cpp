// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/file_io.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>

namespace rocjitsu::detail {

bool fits_in_bounds(uint64_t offset, uint64_t size, size_t limit) {
  return offset <= limit && size <= limit - offset;
}

util::FailureOr<std::vector<char>> read_file_bytes(std::string_view path) {
  std::string path_str(path);
  std::error_code error;
  if (!std::filesystem::is_regular_file(path_str, error))
    return util::Result::failure();

  std::ifstream file(path_str, std::ios::binary | std::ios::ate);
  if (!file)
    return util::Result::failure();

  std::ifstream::pos_type end = file.tellg();
  if (end < 0)
    return util::Result::failure();

  std::vector<char> bytes(static_cast<size_t>(end));
  file.seekg(0, std::ios::beg);
  if (!file)
    return util::Result::failure();
  if (!bytes.empty() && !file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
    return util::Result::failure();

  return bytes;
}

} // namespace rocjitsu::detail
