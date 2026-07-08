////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "amd_hsa_note_bounds.hpp"
#include "inc/amd_hsa_elf.h"

int main() {
  using rocr::amd::hsa::code::detail::IsNoteStringSizeWithinDescriptor;
  using rocr::amd::hsa::code::detail::IsNoteStringSizeWithinRoom;

  static_assert(sizeof(amdgpu_hsa_note_isa_t) >
                    offsetof(amdgpu_hsa_note_isa_t, vendor_and_architecture_name));
  static_assert(sizeof(amdgpu_hsa_note_producer_t) >
                    offsetof(amdgpu_hsa_note_producer_t, producer_name));
  static_assert(sizeof(amdgpu_hsa_note_producer_options_t) >
                    offsetof(amdgpu_hsa_note_producer_options_t, producer_options));

  const size_t isa_name_offset =
      offsetof(amdgpu_hsa_note_isa_t, vendor_and_architecture_name);
  const size_t producer_name_offset =
      offsetof(amdgpu_hsa_note_producer_t, producer_name);
  const size_t options_offset =
      offsetof(amdgpu_hsa_note_producer_options_t, producer_options);

  // Legitimate ISA note: vendor "AMD\0" (4) + arch "gfx906\0" (7) in 32-byte descriptor.
  const uint32_t isa_desc_size = 32;
  const size_t isa_name_room = isa_desc_size - isa_name_offset;
  assert(IsNoteStringSizeWithinDescriptor(isa_desc_size, isa_name_offset, 4));
  assert(IsNoteStringSizeWithinRoom(isa_name_room, 4, 7));

  // Oversized vendor_name_size must be rejected.
  assert(!IsNoteStringSizeWithinDescriptor(isa_desc_size, isa_name_offset, 0xffff));

  // Architecture name that would read past the descriptor.
  assert(!IsNoteStringSizeWithinRoom(isa_name_room, 4, 0xffff));

  // Legitimate producer note.
  const uint32_t producer_desc_size = 24;
  assert(IsNoteStringSizeWithinDescriptor(producer_desc_size, producer_name_offset, 8));

  // Oversized producer_name_size must be rejected.
  assert(!IsNoteStringSizeWithinDescriptor(producer_desc_size, producer_name_offset, 0xffff));

  // Legitimate producer-options note.
  const uint32_t options_desc_size = 20;
  assert(IsNoteStringSizeWithinDescriptor(options_desc_size, options_offset, 6));
  assert(!IsNoteStringSizeWithinDescriptor(options_desc_size, options_offset, 0xffff));

  return 0;
}
