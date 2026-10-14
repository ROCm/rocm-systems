// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "json_output.h"

#include <gtest/gtest.h>
#include <sstream>

TEST(RocjitsuCliJsonTest, QuotesPrintableText) {
  EXPECT_EQ(rocjitsu::cli::json_quoted("runtime_dir"), "\"runtime_dir\"");
}

TEST(RocjitsuCliJsonTest, EscapesJsonControlCharacters) {
  EXPECT_EQ(rocjitsu::cli::json_quoted("quote \" slash \\ tab\t newline\n byte\x01"),
            "\"quote \\\" slash \\\\ tab\\t newline\\n byte\\u0001\"");
}

TEST(RocjitsuCliJsonTest, WritesStringFieldsWithOptionalComma) {
  std::ostringstream out;
  rocjitsu::cli::write_json_string_field(out, "path", "/tmp/rocjitsu", true);
  rocjitsu::cli::write_json_string_field(out, "last", "value", false);

  EXPECT_EQ(out.str(), "  \"path\": \"/tmp/rocjitsu\",\n  \"last\": \"value\"\n");
}
