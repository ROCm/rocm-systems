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

// Edge cases: integer arithmetic, npos handling, multi-marker lines,
// and identity preservation.
//
// This file is a catch-all for small invariants that don't map cleanly to
// one CWE but that a skeptical reviewer will want to see pinned down:
//
//   1. std::string::find returning npos must not cause underflow when we
//      add strlen(type_name) + 1 to it (CWE-190 Integer Overflow).
//      CVE precedent: CVE-2017-1000253 (Linux PIE ELF loader integer
//      miscomputation).
//      https://nvd.nist.gov/vuln/detail/CVE-2017-1000253
//
//   2. Empty input and empty type_name must be safely rejected, not
//      dereferenced or looped on.
//
//   3. If the cgroup line contains BOTH type markers (e.g. a host with
//      nested docker-in-lxc), each parser call returns its own match
//      independently. Order-of-iteration in the caller (fdinfo.cc)
//      determines which type_name is tried first.
//
//   4. The parser must preserve the ID's case exactly — upper/lower
//      mixing is common in LXC names and hex IDs can arrive in either
//      case from debug tools.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct EdgeCase {
  std::string line;
  const char* type_name;
  std::string expected;
  const char* desc;
};
}  // namespace

TEST(ContainerIdParser_EdgeCases, Invariants) {
  const std::vector<EdgeCase> cases = {
      {"0::/kubepods/besteffort/abc", "docker", "",
       "type not found, no integer underflow (docker)"},
      {"0::/kubepods/besteffort/abc", "lxc", "",
       "type not found, no integer underflow (lxc)"},
      {"0::/docker", "docker", "", "type at exact end of line, no over-read"},
      {"0::/docker/", "docker", "", "separator but no id"},
      {"", "docker", "", "empty line"},
      {"0::/docker/abc", "", "", "empty type_name safely rejected"},
      {"0::/docker/first/lxc/second", "docker", "first",
       "multiple markers resolve independently (docker)"},
      {"0::/docker/first/lxc/second", "lxc", "second",
       "multiple markers resolve independently (lxc)"},
      {"0::/lxc/AbC_dEf-123", "lxc", "AbC_dEf-123", "id preserves mixed case"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, c.type_name), c.expected)
        << "edge case: " << c.desc;
  }
}
