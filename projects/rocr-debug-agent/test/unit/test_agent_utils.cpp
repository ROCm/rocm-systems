// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "agent_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace amd::debug_agent;

/* The bytes are stored least significant first, and hex_string prints the
   most significant byte first.  */
TEST (AgentUtilsTest, HexStringReversesBytes)
{
  EXPECT_EQ (hex_string (std::vector<uint8_t>{ 0x01, 0xab }), "ab01");
}

/* A file:// URI has protocol "file" and the decoded path after "://".  */
TEST (AgentUtilsTest, ParseFileUri)
{
  parsed_uri_t uri = parse_code_object_uri ("file:///tmp/a.out");
  EXPECT_EQ (uri.protocol, "file");
  EXPECT_EQ (uri.decoded_path, "/tmp/a.out");
}
