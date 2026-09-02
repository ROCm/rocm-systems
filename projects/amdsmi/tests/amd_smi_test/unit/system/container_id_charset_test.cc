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

TEST(SystemUnit, ContainerIdHaltsAtFirstByteOutsideCharset) {
  for (int b = 0; b <= 0xFF; ++b) {
    const char byte = static_cast<char>(b);
    std::string line = "0::/docker/abc";
    line.push_back(byte);
    line += "def";

    const bool accepted = amd::smi::IsContainerIdChar(static_cast<unsigned char>(b));
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
    if (amd::smi::IsContainerIdChar(static_cast<unsigned char>(b))) continue;
    std::string line = "0::/docker/";
    line.push_back(static_cast<char>(b));
    line += "smuggled";
    EXPECT_EQ(ExtractIdString(line, "docker/"), "")
        << "byte 0x" << std::hex << b << " started an ID";
  }
}
