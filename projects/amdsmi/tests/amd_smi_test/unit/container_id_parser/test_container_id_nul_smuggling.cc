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

// CWE-626 NUL-byte Truncation / Smuggling tests.
//
// Threat: std::string stores embedded NUL bytes faithfully, but any code
// that later does `container_name.c_str()` or passes the pointer to a C
// API sees only the bytes up to the first NUL. This creates a parser
// differential where:
//
//   - The C++ parser reads "abc\0malicious"
//   - The C-string consumer sees "abc"
//   - The Python/Rust binding (which also uses c_str under the hood) sees "abc"
//   - But the in-memory std::string holds the full "abc\0malicious"
//
// The attacker uses this to bypass allow-lists, get-truncated logging,
// or exfiltrate bytes into contexts that *do* read past the NUL.
//
// CWE-626: https://cwe.mitre.org/data/definitions/626.html
//
// Historical CVEs in this class:
//   - CVE-2006-7243  PHP magic-quotes NUL-byte path traversal
//     https://nvd.nist.gov/vuln/detail/CVE-2006-7243
//   - CVE-2010-1916  PHP mysql.trace_mode NUL-byte truncation
//   - Many Go/Rust/C FFI bugs where embedded NUL crossed a language boundary
//     with different semantics (e.g. Go's os.Open vs syscall.Open).
//
// Parser defense: '\0' is not in [a-zA-Z0-9_-], so the charset loop halts at
// the first NUL. No NUL ever reaches the output buffer.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct NulCase {
  std::string line;
  std::string expected;
  const char* desc;
};
}  // namespace

// '\0' is not in [a-zA-Z0-9_-], so the charset scan halts at the first NUL
// and nothing past it is smuggled into the output.
TEST(ContainerIdParser_NulSmuggling, HaltsAtEmbeddedNul) {
  const std::vector<NulCase> cases = {
      {std::string("0::/docker/abc") + '\0' + "smuggled", "abc",
       "NUL mid-ID (CVE-2006-7243)"},
      {std::string("0::/docker/") + '\0' + "xyz", "",
       "NUL at ID start yields empty"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, "docker"), c.expected)
        << "nul case: " << c.desc;
  }
}
