////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#include <cassert>

#include "amd_hsa_alloc_bounds.hpp"

int main() {
  using rocr::amd::hsa::code::detail::IsWithinAmdNoteBufferLimit;
  using rocr::amd::hsa::code::detail::kMaxAmdNoteBufferSize;

  assert(IsWithinAmdNoteBufferLimit(0));
  assert(IsWithinAmdNoteBufferLimit(kMaxAmdNoteBufferSize));
  assert(!IsWithinAmdNoteBufferLimit(kMaxAmdNoteBufferSize + 1));
  return 0;
}
