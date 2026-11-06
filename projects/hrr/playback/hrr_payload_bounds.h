/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// Bounds on a count the archive supplies for an array stored inside the record.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace hrr {

// Number of `elem_size`-byte elements an inline field of `field_bytes` holds.
constexpr size_t inline_capacity(size_t field_bytes, size_t elem_size) {
  return elem_size ? field_bytes / elem_size : 0;
}

// True if a record claiming `count` elements fits its inline field. A count
// larger than the field is a malformed record, not a request to read beyond it.
constexpr bool inline_count_fits(uint64_t count, size_t field_bytes, size_t elem_size) {
  return count <= inline_capacity(field_bytes, elem_size);
}

// A string held in a fixed-size character field of a record. The archive need
// not have put a NUL in the field, so the text is whatever precedes the first
inline std::string bounded_cstr(const uint8_t* field, size_t field_bytes) {
  const void* nul = std::memchr(field, '\0', field_bytes);
  const size_t n = nul ? static_cast<size_t>(static_cast<const uint8_t*>(nul) - field)
                       : field_bytes;
  return std::string(reinterpret_cast<const char*>(field), n);
}

// True if the dependency list stored in a graph-node record holds every
// dependency the call was made with. Capture keeps at most what the inline
constexpr bool deps_complete(uint64_t recorded, uint64_t stored) {
  return stored >= recorded;
}

}  // namespace hrr
