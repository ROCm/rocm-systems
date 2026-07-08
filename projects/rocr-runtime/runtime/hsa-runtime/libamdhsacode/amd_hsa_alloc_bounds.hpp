////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef AMD_HSA_ALLOC_BOUNDS_HPP_
#define AMD_HSA_ALLOC_BOUNDS_HPP_

#include <cstddef>

namespace rocr {
namespace amd {
namespace hsa {
namespace code {
namespace detail {

constexpr std::size_t kMaxAmdNoteBufferSize = 4096;
constexpr std::size_t kMaxSectionPrintSize = 1024 * 1024;

inline bool IsWithinAmdNoteBufferLimit(std::size_t size) {
  return size <= kMaxAmdNoteBufferSize;
}

inline bool IsWithinSectionPrintLimit(std::size_t size) {
  return size <= kMaxSectionPrintSize;
}

}  // namespace detail
}  // namespace code
}  // namespace hsa
}  // namespace amd
}  // namespace rocr

#endif  // AMD_HSA_ALLOC_BOUNDS_HPP_
