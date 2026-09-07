// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// Host-only unit test for aql_profile::StableErrorString. Models the way
// hsa_ven_amd_aqlprofile_error_string() returns a pointer derived from
// Logger::LastMessage() -- a reference to a per-thread buffer that later
// messages overwrite. Build/run standalone:
//   c++ -std=c++17 -I projects/aqlprofile/src \
//       projects/aqlprofile/src/core/tests/error_string_util_test.cpp \
//       -lgtest -lgtest_main -o t && ./t

#include <string>

#include <gtest/gtest.h>

#include "core/error_string_util.h"

namespace {

struct Case {
  const char* description;  // what this row exercises + expected outcome
  const char* first;        // message the pointer is created from
  const char* second;       // message logged afterwards (overwrites the buffer)
};

// The returned pointer must keep reading the message it was created from even
// after the source buffer is overwritten by a later message.
TEST(StableErrorStringTest, PointerSurvivesBufferOverwrite) {
  const Case cases[] = {
      {"positive: same-length overwrite", "aaaa", "bbbb"},
      {"boundary: shorter overwrite", "hello", "hi"},
      {"corner: empty then non-empty", "", "later error"},
      {"positive: typical messages", "invalid event", "out of resources"},
  };
  for (const Case& c : cases) {
    // Reserve so the assignments below reuse the same heap buffer in place --
    // this mirrors Logger reusing message_[tid] and keeps the test well-defined.
    std::string buffer;
    buffer.reserve(64);
    buffer = c.first;

    const char* p = aql_profile::StableErrorString(buffer);

    buffer = c.second;  // a later message overwrites the source buffer

    EXPECT_STREQ(p, c.first) << c.description;
  }
}

// strerror-style contract: a fresh call reflects the newest message.
TEST(StableErrorStringTest, FreshCallReflectsNewMessage) {
  std::string buffer;
  buffer.reserve(64);
  buffer = "first";
  EXPECT_STREQ(aql_profile::StableErrorString(buffer), "first");
  buffer = "second";
  EXPECT_STREQ(aql_profile::StableErrorString(buffer), "second");
}

}  // namespace
