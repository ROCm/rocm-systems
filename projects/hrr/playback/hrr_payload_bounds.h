/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// Bounds on a count the archive supplies for an array stored inside the record.
//
// Several recorded API structs carry a small array inline (for example up to 16
// graph-node handles in `pDependencies_bytes`) next to a count field. The count
// is archive data: nothing makes it agree with how many elements the record
// actually holds. A handler that copies `count * sizeof(element)` bytes out of
// the inline field reads past the record when the count is larger. These
// helpers are header-only, with no HIP dependency, so the rule can be tested
// without a GPU.

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
// NUL within `field_bytes`, or all of the field when there is none. The result
// is a std::string, which is NUL-terminated inside its own buffer, so a callee
// that reads it as a C string stays inside it.
inline std::string bounded_cstr(const uint8_t* field, size_t field_bytes) {
  const void* nul = std::memchr(field, '\0', field_bytes);
  const size_t n = nul ? static_cast<size_t>(static_cast<const uint8_t*>(nul) - field)
                       : field_bytes;
  return std::string(reinterpret_cast<const char*>(field), n);
}

}  // namespace hrr
