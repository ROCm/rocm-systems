////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef AMD_HSA_NOTE_BOUNDS_HPP_
#define AMD_HSA_NOTE_BOUNDS_HPP_

#include <cstddef>
#include <cstdint>

namespace rocr {
namespace amd {
namespace hsa {
namespace code {
namespace detail {

// Returns false when claimed_size would cause GetNoteString to read past desc_size.
inline bool IsNoteStringSizeWithinDescriptor(uint32_t desc_size, size_t field_offset,
                                             uint16_t claimed_size) {
  if (desc_size < field_offset) { return false; }
  return claimed_size <= desc_size - field_offset;
}

// Bounds a trailing variable-length field within room already carved out of the descriptor.
inline bool IsNoteStringSizeWithinRoom(size_t room, size_t offset, uint16_t claimed_size) {
  if (offset > room) { return false; }
  return claimed_size <= room - offset;
}

}  // namespace detail
}  // namespace code
}  // namespace hsa
}  // namespace amd
}  // namespace rocr

#endif  // AMD_HSA_NOTE_BOUNDS_HPP_
