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

// CWE-1007 Visual Identity Spoofing / Trojan Source tests.
//
// Threat: Unicode contains several categories of chars designed to trick
// humans (or text-rendering systems) into misreading strings:
//
//   - Bidirectional overrides (U+202A-U+202E, U+2066-U+2069) reorder
//     characters at display time — the attacker-visible string is
//     different from the byte sequence.
//   - Zero-width chars (U+200B-U+200D, U+FEFF) are invisible but present.
//   - Homoglyphs: Cyrillic 'а' (U+0430) renders identically to Latin 'a'
//     (U+0061) but is a completely different codepoint.
//
// If these land in a container ID that is then logged or shown to a human
// operator, the operator cannot distinguish "docker-abc" from
// "docker-а b c" (Cyrillic).
//
// CWE-1007: https://cwe.mitre.org/data/definitions/1007.html
//
// Public exploit / research:
//   - CVE-2021-42574  "Trojan Source" — BiDi override attacks in source code
//     https://trojansource.codes/
//     https://nvd.nist.gov/vuln/detail/CVE-2021-42574
//   - Unicode TR#36 Security Considerations
//     https://www.unicode.org/reports/tr36/
//
// Parser defense: the charset whitelist is strictly ASCII [a-zA-Z0-9_-]. Every
// non-ASCII byte (>= 0x80) is rejected. Because all relevant BiDi/zero-
// width/homoglyph chars are encoded as multi-byte UTF-8 with high bit set,
// they never survive charset validation.
//
// This file pins down the specific UTF-8 byte sequences we must reject.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
struct HomoglyphCase {
  std::string line;
  std::string expected;
  const char* desc;
};
}  // namespace

// Multi-byte UTF-8 sequences all have the high bit set and so fail the ASCII
// charset check, truncating the ID at the first non-ASCII byte.
TEST(ContainerIdParser_Homoglyph, RejectsMultiByteUtf8AtFirstByte) {
  const std::vector<HomoglyphCase> cases = {
      {std::string("0::/docker/abc") + "\xE2\x80\xAE" + "evilsuffix", "abc",
       "U+202E BiDi RTL override (CVE-2021-42574 Trojan Source)"},
      {std::string("0::/docker/abc") + "\xE2\x80\x8B", "abc",
       "U+200B zero-width space"},
      {std::string("0::/docker/abc") + "\xEF\xBB\xBF", "abc", "U+FEFF UTF-8 BOM"},
      {std::string("0::/docker/abc") + "\xD0\xB0" + "def", "abc",
       "U+0430 Cyrillic homoglyph for Latin 'a'"},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ExtractIdString(c.line, "docker"), c.expected)
        << "homoglyph: " << c.desc;
  }
}

TEST(ContainerIdParser_Homoglyph, Rejects_AllHighBitBytes_Exhaustive) {
  for (int b = 0x80; b <= 0xFF; ++b) {
    std::string line = "0::/docker/abc";
    line.push_back(static_cast<char>(b));
    line += "def";
    EXPECT_EQ(ExtractIdString(line, "docker"), "abc")
        << "byte 0x" << std::hex << b << " leaked through charset filter";
  }
}
