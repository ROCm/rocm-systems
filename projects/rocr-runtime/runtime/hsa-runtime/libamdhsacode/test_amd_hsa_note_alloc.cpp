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
#include <string>

#include "inc/amd_hsa_elf.h"

// Mirrors detail::kMaxAmdNoteBufferSize and the size checks in amd_hsa_code.cpp.
constexpr size_t kMaxAmdNoteBufferSize = 4096;

inline bool FitsInAmdNoteBuffer(size_t size) {
  return size <= kMaxAmdNoteBufferSize;
}

size_t IsaNoteBufferSize(const std::string& vendor_name,
                         const std::string& architecture_name) {
  return sizeof(amdgpu_hsa_note_isa_t) + vendor_name.length() +
         architecture_name.length() + 1;
}

size_t ProducerNoteBufferSize(const std::string& producer) {
  return sizeof(amdgpu_hsa_note_producer_t) + producer.length();
}

size_t ProducerOptionsNoteBufferSize(const std::string& options) {
  return sizeof(amdgpu_hsa_note_producer_options_t) + options.length();
}

int main() {
  assert(FitsInAmdNoteBuffer(0));
  assert(FitsInAmdNoteBuffer(kMaxAmdNoteBufferSize));
  assert(!FitsInAmdNoteBuffer(kMaxAmdNoteBufferSize + 1));

  const std::string vendor = "AMD";
  const std::string architecture = "AMDGPU";
  assert(FitsInAmdNoteBuffer(IsaNoteBufferSize(vendor, architecture)));
  assert(FitsInAmdNoteBuffer(ProducerNoteBufferSize("clang")));
  assert(FitsInAmdNoteBuffer(ProducerOptionsNoteBufferSize("-O2")));

  const std::string oversized(5000, 'x');
  assert(!FitsInAmdNoteBuffer(IsaNoteBufferSize(oversized, oversized)));
  assert(!FitsInAmdNoteBuffer(ProducerNoteBufferSize(oversized)));
  assert(!FitsInAmdNoteBuffer(ProducerOptionsNoteBufferSize(oversized)));

  return 0;
}
