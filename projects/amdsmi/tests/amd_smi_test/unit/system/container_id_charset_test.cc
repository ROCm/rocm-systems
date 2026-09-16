// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Charset coverage for amd::smi::ExtractContainerId. The ID reaches log and
// CLI output, so the scan must halt at the first byte outside [a-zA-Z0-9_-].
// Asserted over all 256 byte values rather than a sample: control bytes, NUL,
// path separators, shell metacharacters and UTF-8 are all just bytes here.

#include <gtest/gtest.h>

#include <string>

#include "amd_smi/impl/amd_smi_container_id_parser.h"
#include "container_id_test_util.h"

using amdsmi_test::ExtractIdString;

namespace {
// Spelled out rather than delegated to IsContainerIdChar, so the test and the
// implementation are able to disagree.
bool ExpectedIdChar(int b) {
  return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b == '_' ||
         b == '-';
}
}  // namespace

TEST(SystemUnit, ContainerIdHaltsAtFirstByteOutsideCharset) {
  for (int b = 0; b <= 0xFF; ++b) {
    const char byte = static_cast<char>(b);
    std::string line = "0::/docker/abc";
    line.push_back(byte);
    line += "def";

    const bool accepted = ExpectedIdChar(b);
    const std::string expected = accepted ? std::string("abc") + byte + "def" : "abc";
    EXPECT_EQ(ExtractIdString(line, "docker/"), expected)
        << "byte 0x" << std::hex << b
        << (accepted ? " wrongly halted the scan" : " leaked through the charset filter");
  }
}

// The same rule at the first byte of the ID: there is no ID at all, and the
// output must be an empty string rather than a partial one.
TEST(SystemUnit, ContainerIdIsEmptyWhenFirstByteIsOutsideCharset) {
  for (int b = 0; b <= 0xFF; ++b) {
    if (ExpectedIdChar(b)) continue;
    std::string line = "0::/docker/";
    line.push_back(static_cast<char>(b));
    line += "smuggled";
    EXPECT_EQ(ExtractIdString(line, "docker/"), "")
        << "byte 0x" << std::hex << b << " started an ID";
  }
}

// The charset predicates themselves, against literal ranges. Without this the
// suite only ever asks the implementation to agree with itself: widening
// IsLowerHexChar to accept 'g' leaves every other assertion in the suite green.
TEST(SystemUnit, ContainerIdCharsetPredicatesMatchTheirRanges) {
  for (int b = 0; b <= 0xFF; ++b) {
    const auto byte = static_cast<unsigned char>(b);
    EXPECT_EQ(amd::smi::IsContainerIdChar(byte), ExpectedIdChar(b))
        << "IsContainerIdChar(0x" << std::hex << b << ")";
    EXPECT_EQ(amd::smi::IsLowerHexChar(byte), (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f'))
        << "IsLowerHexChar(0x" << std::hex << b << ")";
  }
}
