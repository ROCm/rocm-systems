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

// CWE-120 / CWE-787 Buffer bounds (canary-protected).
//
// Threat: a classic off-by-one or unchecked length in the parser overwrites
// bytes past the end of info.container_name[AMDSMI_MAX_STRING_LENGTH],
// corrupting adjacent struct fields or heap metadata.
//
// Attack surface in this parser:
//   - Input line can be arbitrarily long
//   - Output is a fixed-size char array
//   - ExtractIdInto() takes an explicit out_cap that MUST be respected
//
// CWE-120: https://cwe.mitre.org/data/definitions/120.html  (classic overflow)
// CWE-787: https://cwe.mitre.org/data/definitions/787.html  (out-of-bounds write)
//
// Methodology: wrap the output buffer in GuardedBuffer<N>, which places
// recognizable 64-bit canary values on each side. After every call we
// verify the canaries are intact. Any off-by-one write smashes a canary
// and fails the test with a diagnosable pattern in gdb.
//
// Canary pattern references:
//   - StackGuard/ProPolice   https://en.wikipedia.org/wiki/Stack_buffer_overflow#Canaries
//   - OWASP Buffer Overflow  https://owasp.org/www-community/vulnerabilities/Buffer_Overflow
//
// The tests cover:
//   1. Exactly-max-length input fills the output without smashing canaries.
//   2. Overlong input (far beyond the cap) is safely truncated.
//   3. Smaller-than-max output buffer is respected (out_cap parameter works).
//   4. Zero-capacity output buffer writes nothing.
//   5. Cross-product: every attack input from other test files leaves the
//      canary intact.

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "fixtures.h"
#include "guarded_buffer.h"
#include "container_id_test_util.h"

using amdsmi_test::ExtractIdInto;
using amdsmi_test::GuardedBuffer;

TEST(ContainerIdParser_BufferBounds, ExactlyMaxLength_NullTerminated) {
  const std::string line =
      std::string("0::/docker/") + amdsmi_test::kDocker64;
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  size_t n = ExtractIdInto(line, "docker", gb.buf, sizeof(gb.buf));
  EXPECT_EQ(n, static_cast<size_t>(AMDSMI_MAX_CONTAINER_ID_LENGTH));
  EXPECT_EQ(gb.buf[AMDSMI_MAX_CONTAINER_ID_LENGTH], '\0');
  EXPECT_TRUE(gb.CanariesIntact());
}

TEST(ContainerIdParser_BufferBounds, OverlongInput_CanariesIntact_CWE120) {
  std::string line = "0::/docker/";
  line.append(1024, 'z');
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  ExtractIdInto(line, "docker", gb.buf, sizeof(gb.buf));
  EXPECT_TRUE(gb.CanariesIntact());
}

TEST(ContainerIdParser_BufferBounds, SmallOutputBuffer_RespectsCapacity) {
  const std::string line =
      std::string("0::/docker/") + amdsmi_test::kDocker64;
  GuardedBuffer<16> gb;
  size_t n = ExtractIdInto(line, "docker", gb.buf, sizeof(gb.buf));
  EXPECT_EQ(n, 15u);  // 16-byte cap minus one NUL terminator
  EXPECT_EQ(gb.buf[15], '\0');
  EXPECT_TRUE(gb.CanariesIntact());
}

TEST(ContainerIdParser_BufferBounds, ZeroCapacityOutputBuffer_NoWrite) {
  const std::string line =
      std::string("0::/docker/") + amdsmi_test::kDocker64;
  GuardedBuffer<1> gb;
  size_t n = ExtractIdInto(line, "docker", gb.buf, 0);
  EXPECT_EQ(n, 0u);
  EXPECT_TRUE(gb.CanariesIntact());
}

TEST(ContainerIdParser_BufferBounds,
     AllAttackInputs_CanariesIntact_CrossProduct) {
  // Sweep every known-adversarial input through the parser; canaries must
  // survive every one. New attack classes should be appended here.
  const std::vector<std::string> attacks = {
      "0::/docker/abc\nFAKE",
      "0::/docker/abc\rcr",
      std::string("0::/docker/abc") + '\0' + "nul",
      "0::/docker/abc\x1b[2Jterm",
      "0::/docker/abc;rm",
      "0::/docker/abc`id`",
      "0::/docker/abc$(id)",
      "0::/docker/abc|cmd",
      "0::/docker/abc&bg",
      "0::/docker/" + std::string(10000, 'x'),
      std::string("0::/docker/\xE2\x80\xAE") + "bidi",
  };
  for (const auto& line : attacks) {
    GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
    ExtractIdInto(line, "docker", gb.buf, sizeof(gb.buf));
    EXPECT_TRUE(gb.CanariesIntact())
        << "canary smashed by input of " << line.size() << " bytes";
  }
}
