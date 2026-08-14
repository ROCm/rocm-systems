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

// CWE-400 Uncontrolled Resource Consumption (DoS resistance).
//
// Threat: a parser that allocates or iterates proportionally to input
// size can be turned into a denial-of-service primitive by feeding it
// pathologically large input. For /proc parsers this matters because:
//
//   - /proc content is kernel-controlled but the kernel imposes few
//     hard limits on line length (typically bounded by PAGE_SIZE per
//     read, but accumulated across reads a line can be arbitrarily long).
//   - Container runtimes and user-namespace setups can produce very
//     long cgroup paths (deeply nested systemd slices).
//   - Even if the input is "trusted" (root reading /proc of processes
//     it owns), a pathological input should degrade gracefully, not
//     hang the tool.
//
// CWE-400:  https://cwe.mitre.org/data/definitions/400.html
// CWE-407:  https://cwe.mitre.org/data/definitions/407.html  (algorithmic complexity)
//
// Relevant CVEs where oversize /proc or file content caused issues:
//   - CVE-2022-0185   fsconfig slab overflow via oversized legacy_parse_param
//     https://nvd.nist.gov/vuln/detail/CVE-2022-0185
//   - CVE-2018-1000613  BouncyCastle unbounded allocation
//     https://nvd.nist.gov/vuln/detail/CVE-2018-1000613
//
// Parser defense: the charset consume loop is bounded by
// AMDSMI_MAX_CONTAINER_ID_LENGTH (64) iterations. The only unbounded
// operation is std::string::find() for the type_name itself, which is
// O(n*m) but with tiny m (<=6 chars); acceptable worst case verified below.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct SizeCase {
  size_t fill;
  char ch;
  const char* desc;
};
}  // namespace

// Regardless of input length, the extracted ID is capped at
// AMDSMI_MAX_CONTAINER_ID_LENGTH.
TEST(ContainerIdParser_DoS, OversizeInputStaysBounded) {
  const std::vector<SizeCase> cases = {
      {4096, 'a', "4 KB line"},
      {65536, 'b', "64 KB line"},
  };
  for (const auto& c : cases) {
    std::string line = "0::/docker/";
    line.append(c.fill, c.ch);
    EXPECT_EQ(ExtractIdString(line, "docker").size(),
              static_cast<size_t>(AMDSMI_MAX_CONTAINER_ID_LENGTH))
        << "size case: " << c.desc;
  }
}

TEST(ContainerIdParser_DoS, BoundedIterations_ConstantTimeWorstCase) {
  // 1 MB input; 1000 extractions must complete under a generous wall-clock
  // bound. Tests that the parser degrades gracefully under adversarial size.
  std::string line = "0::/docker/";
  line.append(1 << 20, 'c');
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    auto s = ExtractIdString(line, "docker");
    (void)s;
  }
  auto dt = std::chrono::steady_clock::now() - t0;
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
  EXPECT_LT(ms, 2000) << "parser is not bounded enough for DoS resistance";
}
