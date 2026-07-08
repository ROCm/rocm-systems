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
  using rocr::amd::hsa::code::detail::IsWithinSectionPrintLimit;
  using rocr::amd::hsa::code::detail::kMaxSectionPrintSize;

  assert(IsWithinSectionPrintLimit(0));
  assert(IsWithinSectionPrintLimit(kMaxSectionPrintSize));
  assert(!IsWithinSectionPrintLimit(kMaxSectionPrintSize + 1));
  return 0;
}
