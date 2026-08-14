/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// GuardedBuffer<N> — a fixed-size character buffer flanked by 64-bit canaries
// on each side. Any out-of-bounds write by the parser under test corrupts a
// canary, which CanariesIntact() detects.
//
// Canary values are deliberately recognizable constants (0xDEADBEEFDEADBEEF
// and 0xCAFEBABECAFEBABE) so that when a test fails, the exact corruption
// pattern is visible in gdb / core dumps.
//
// Used by:
//   test_v5_buffer_bounds.cc  — CWE-120 Classic Buffer Overflow
//   test_v5_homoglyph.cc      — to confirm UTF-8 bytes don't overflow
//   test_v5_dos.cc            — to confirm 1MB inputs stay bounded

#ifndef AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_GUARDED_BUFFER_H_
#define AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_GUARDED_BUFFER_H_

#include <cstddef>
#include <cstdint>

namespace amdsmi_test {

template <size_t N>
struct GuardedBuffer {
  uint64_t pre_canary = 0xDEADBEEFDEADBEEFULL;
  char buf[N] = {0};
  uint64_t post_canary = 0xCAFEBABECAFEBABEULL;

  bool CanariesIntact() const {
    return pre_canary == 0xDEADBEEFDEADBEEFULL &&
           post_canary == 0xCAFEBABECAFEBABEULL;
  }
};

}  // namespace amdsmi_test

#endif  // AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_GUARDED_BUFFER_H_
