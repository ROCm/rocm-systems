/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

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
