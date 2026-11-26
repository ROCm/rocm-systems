/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef UNIQUE_ID_HEX_HPP
#define UNIQUE_ID_HEX_HPP

#include <charconv>
#include <cstddef>
#include <cstdio>
#include <string>

#include <rccl/rccl.h>

namespace RcclUnitTesting {

// Encodes an ncclUniqueId as lowercase hex, two characters per byte, to hand it to another process.
inline std::string uniqueIdToHex(const ncclUniqueId& id) {
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&id);
  std::string out;
  out.reserve(sizeof(ncclUniqueId) * 2);
  char buf[3];
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    snprintf(buf, sizeof(buf), "%02x", bytes[i]);
    out += buf;
  }
  return out;
}

// Decodes the output of uniqueIdToHex(). Returns false if the length is wrong or a byte is not exactly two hex
// digits.
inline bool uniqueIdFromHex(const std::string& hex, ncclUniqueId& id) {
  if (hex.size() != sizeof(ncclUniqueId) * 2) return false;
  unsigned char* bytes = reinterpret_cast<unsigned char*>(&id);
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    const char* pair = hex.data() + i * 2;
    unsigned int v = 0;
    const auto [end, ec] = std::from_chars(pair, pair + 2, v, 16);
    if (ec != std::errc() || end != pair + 2) {
      return false;
    }
    bytes[i] = static_cast<unsigned char>(v);
  }
  return true;
}

}  // namespace RcclUnitTesting

#endif  // UNIQUE_ID_HEX_HPP
