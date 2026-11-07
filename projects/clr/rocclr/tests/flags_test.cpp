// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "top.hpp"
#include "utils/flags.hpp"

#include <cstdio>
#include <initializer_list>
#include <limits>

// Exercise the production implementation without linking a GPU runtime. Only
// the unrelated OS and logging dependencies used by Flag::init are substituted.
#define OS_HPP_
namespace amd {
class Os {
 public:
  static int getProcessId() { return 0; }
};
FILE* outFile = stderr;
void EnableAsyncLogging(bool) {}
}  // namespace amd
#include "utils/flags.cpp"

namespace {
int failures = 0;

void check(bool condition, const char* message, const char* input) {
  if (!condition) {
    std::fprintf(stderr, "%s: [%s]\n", message, input);
    ++failures;
  }
}
}  // namespace

int main() {
  struct ValidCase {
    const char* input;
    uint expected;
  };
  const ValidCase valid[] = {
      {"0x2008", 0x2008u},
      {"0x7fffffff", 0x7fffffffu},
      {"0xffffffff", std::numeric_limits<uint>::max()},
      {"0XFFFFFFFF", std::numeric_limits<uint>::max()},
      {"0x80000000", 0x80000000u},
      {"0x0", 0},
      {"0x00002008", 0x2008u},
      {" +0X2008 \t", 0x2008u},
      {"\t0x2008\n", 0x2008u},
      {"0", 0},
      {"8200", 8200},
      {"0008200", 8200},
      {"08", 8},
      {" +8200", 8200},
      {"-1", std::numeric_limits<uint>::max()},
  };
  for (const auto& test : valid) {
    uint mask = 42;
    amd::Flag flag{"AMD_LOG_MASK", &mask, amd::Flag::Tuint, true};
    check(flag.setValue(test.input), "Valid mask rejected", test.input);
    check(mask == test.expected, "Incorrect mask value", test.input);
    check(!flag.isDefault_, "Explicit mask still marked default", test.input);
  }

  const char* invalid[] = {
      "0x",    "0X",   "0xg",         "0x2008junk",
      "0x1 2", "-0x1", "0x100000000", "0xffffffffffffffffffffffffffffffff",
  };
  for (const char* input : invalid) {
    for (bool isDefault : {false, true}) {
      uint mask = 42;
      amd::Flag flag{"AMD_LOG_MASK", &mask, amd::Flag::Tuint, isDefault};
      check(!flag.setValue(input), "Invalid mask accepted", input);
      check(mask == 42, "Invalid mask changed the value", input);
      check(flag.isDefault_ == isDefault, "Invalid mask changed default status", input);
    }
  }

  uint other = 42;
  amd::Flag otherFlag{"GPU_MAX_USWC_ALLOC_SIZE", &other, amd::Flag::Tuint, true};
  check(otherFlag.setValue("-1") && other == std::numeric_limits<uint>::max(),
        "Unsigned decimal sentinel changed", "-1");
  check(otherFlag.setValue("010") && other == 10, "Leading-zero decimal changed", "010");
  check(otherFlag.setValue("0x10") && other == 0, "Unrelated flag parsing changed", "0x10");

  amd::Flag constantFlag{"AMD_LOG_MASK", nullptr, amd::Flag::Tuint, true};
  check(!constantFlag.setValue("0x2008") && constantFlag.isDefault_, "Constant flag changed",
        "0x2008");

  return failures == 0 ? 0 : 1;
}
