/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once


#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace hrr {

constexpr size_t inline_capacity(size_t field_bytes, size_t elem_size) {
  return elem_size ? field_bytes / elem_size : 0;
}

// True if a record claiming `count` elements fits its inline field. A count
// larger than the field is a malformed record, not a request to read beyond it.
constexpr bool inline_count_fits(uint64_t count, size_t field_bytes, size_t elem_size) {
  return count <= inline_capacity(field_bytes, elem_size);
}

// A string held in a fixed-size character field. The archive may omit the NUL,
// so the text ends at the first NUL or the end of the field.
inline std::string bounded_cstr(const uint8_t* field, size_t field_bytes) {
  const void* nul = std::memchr(field, '\0', field_bytes);
  const size_t n = nul ? static_cast<size_t>(static_cast<const uint8_t*>(nul) - field)
                       : field_bytes;
  return std::string(reinterpret_cast<const char*>(field), n);
}

// True if the dependency list in a graph-node record holds every dependency the call
// was made with. Capture keeps at most what its inline array holds.
constexpr bool deps_complete(uint64_t recorded, uint64_t stored) {
  return stored >= recorded;
}

}  // namespace hrr
