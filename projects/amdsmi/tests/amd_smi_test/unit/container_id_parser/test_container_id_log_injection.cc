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

// CWE-117 Log Injection / Terminal Hijack tests.
//
// Threat: the container ID eventually appears in log output, CLI output,
// APIs that might be consumed by HTML dashboards, and downstream tools.
// If an attacker can embed control bytes in the ID (newline, CR, ANSI
// escape sequences, BEL, etc.) they can:
//
//   - Forge additional log lines          (\n, \r\n)
//   - Clear the user's terminal           (\x1b[2J)
//   - Rewrite the terminal title          (\x1b]0;...\x07)
//   - Overwrite previous log content      (\r, \x08 backspace)
//   - Hide tool output                    (BEL, cursor-move escapes)
//
// CWE-117:  https://cwe.mitre.org/data/definitions/117.html
//
// Related public CVEs (same bug class):
//   - CVE-2017-8816  curl NTLM log injection via Authorization header
//     https://nvd.nist.gov/vuln/detail/CVE-2017-8816
//   - CVE-2021-23385  Flask-RESTX CRLF injection
//     https://nvd.nist.gov/vuln/detail/CVE-2021-23385
//   - CVE-2020-11022  jQuery HTML injection (analogous sanitization class)
//
// Background: OWASP Log Injection cheatsheet
//   https://owasp.org/www-community/attacks/Log_Injection
//
// Parser defense: charset whitelist [a-zA-Z0-9_-] rejects ALL of these control
// bytes. This file pins down each specific control byte we must reject.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct InjectionCase {
  std::string line;
  std::string expected;
  const char* desc;
};
}  // namespace

// The charset whitelist halts at the first control byte, so none reach the
// output. type_name is "docker" for every case.
TEST(ContainerIdParser_LogInjection, RejectsControlByteAtFirstOccurrence) {
  const std::vector<InjectionCase> cases = {
      {"0::/docker/abc\nFAKE_LOG_ENTRY", "abc", "newline (CVE-2017-8816)"},
      {"0::/docker/abc\rOVERWRITE", "abc", "carriage return (CVE-2021-23385)"},
      {"0::/docker/abc\x1b[2Jpayload", "abc", "ANSI CSI clear-screen"},
      {"0::/docker/abc\x1b]0;TITLE\x07payload", "abc", "ANSI OSC title-change"},
      {std::string("0::/docker/abc\x07") + "def", "abc", "bell"},
      {std::string("0::/docker/abc\x08") + "def", "abc", "backspace"},
      {"0::/docker/abc\tdef", "abc", "tab"},
  };
  for (const auto& c : cases) {
    const std::string got = ExtractIdString(c.line, "docker");
    EXPECT_EQ(got, c.expected) << "control byte: " << c.desc;
  }
}
