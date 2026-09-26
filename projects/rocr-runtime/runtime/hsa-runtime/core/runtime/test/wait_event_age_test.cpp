////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// Host-only table-driven test for WaitEventAgeBytes (no GPU / driver required).
// The event_age buffer stores one uint64_t per unique signal; the sizing helper
// must therefore return unique_evts * sizeof(uint64_t) so the _alloca allocation
// covers the subsequent memset (which uses sizeof(uint64_t)).
//
// Build/run:
//   c++ -std=c++17 -I projects/rocr-runtime/runtime/hsa-runtime \
//       projects/rocr-runtime/runtime/hsa-runtime/core/runtime/test/wait_event_age_test.cpp \
//       -o t && ./t

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "core/inc/wait_event_age.h"

using rocr::core::WaitEventAgeBytes;

namespace {

struct Case {
  const char* description;   // what this row exercises + expected outcome
  uint32_t    unique_evts;   // input: number of unique signals
  std::size_t expected;      // expected bytes = unique_evts * sizeof(uint64_t)
};

}  // namespace

int main() {
  int failures = 0;

  const Case cases[] = {
      {"boundary: 0 events -> 0 bytes",            0,    0},
      {"positive: 1 event  -> 8 bytes",            1,    8},
      {"positive: 2 events -> 16 bytes",           2,    16},
      {"boundary: 3 events -> 24 bytes",           3,    24},
      {"positive: 64 events -> 512 bytes",         64,   512},
      {"corner: 1024 events -> 8192 bytes",        1024, 8192},
  };

  for (const Case& c : cases) {
    const std::size_t got = WaitEventAgeBytes(c.unique_evts);
    if (got != c.expected) {
      ++failures;
      std::printf("FAIL: %s (got %zu, expected %zu)\n", c.description, got, c.expected);
    }
  }

  // Regression: the allocation MUST cover the memset, which writes
  // unique_evts * sizeof(uint64_t) bytes. Pre-fix the helper used
  // sizeof(uint32_t) (4) per element, half the required size -> stack overflow.
  for (uint32_t n : {1u, 2u, 7u, 100u}) {
    const std::size_t alloc = WaitEventAgeBytes(n);
    const std::size_t memset_bytes = static_cast<std::size_t>(n) * sizeof(uint64_t);
    if (alloc < memset_bytes) {
      ++failures;
      std::printf("FAIL: regression: alloc %zu < memset %zu for n=%u (under-allocation)\n",
                  alloc, memset_bytes, n);
    }
  }

  if (failures != 0) {
    std::printf("%d FAILED\n", failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
