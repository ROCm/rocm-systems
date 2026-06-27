// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/file_io.h"

#include <cstddef>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>

namespace rocjitsu::detail {

bool fits_in_bounds(uint64_t offset, uint64_t size, size_t limit) {
  return offset <= limit && size <= limit - offset;
}

util::inline_vector<char, 0> read_file_bytes(std::string_view path) {
  std::string path_str(path);
  std::ifstream file(path_str, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("failed to open " + path_str);

  std::ifstream::pos_type end = file.tellg();
  if (end < 0)
    throw std::runtime_error("failed to size " + path_str);
  if (static_cast<uint64_t>(end) > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("file too large " + path_str);

  util::inline_vector<char, 0> bytes(static_cast<uint32_t>(end));
  file.seekg(0, std::ios::beg);
  if (!file)
    throw std::runtime_error("failed to seek " + path_str);
  if (!bytes.empty() && !file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error("failed to read " + path_str);

  return bytes;
}

} // namespace rocjitsu::detail
