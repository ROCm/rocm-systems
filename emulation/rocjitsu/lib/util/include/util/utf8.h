// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace util {

/// @brief Accept UTF-8 scalar encodings without normalizing or replacing bytes.
inline bool is_valid_utf8(std::string_view text) {
  for (size_t offset = 0; offset < text.size();) {
    const uint8_t lead = static_cast<uint8_t>(text[offset++]);
    if (lead < 0x80)
      continue;
    const uint32_t continuation_bytes = lead >= 0xc2 && lead <= 0xdf   ? 1
                                        : lead >= 0xe0 && lead <= 0xef ? 2
                                        : lead >= 0xf0 && lead <= 0xf4 ? 3
                                                                       : 0;
    if (continuation_bytes == 0 || continuation_bytes > text.size() - offset)
      return false;
    uint32_t code_point = lead & (0x3f >> continuation_bytes);
    for (uint32_t index = 0; index < continuation_bytes; ++index) {
      const uint8_t byte = static_cast<uint8_t>(text[offset++]);
      if ((byte & 0xc0) != 0x80)
        return false;
      code_point = (code_point << 6) | (byte & 0x3f);
    }
    const uint32_t minimum = continuation_bytes == 1   ? 0x80
                             : continuation_bytes == 2 ? 0x800
                                                       : 0x10000;
    if (code_point < minimum || code_point > 0x10ffff ||
        (code_point >= 0xd800 && code_point <= 0xdfff))
      return false;
  }
  return true;
}

} // namespace util
