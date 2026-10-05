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

}  // namespace hrr
