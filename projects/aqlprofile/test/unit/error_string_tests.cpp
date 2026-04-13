// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "core/logger.h"

// Static member definitions for Logger.
namespace aql_profile {
Logger::mutex_t Logger::mutex_;
Logger* Logger::instance_ = nullptr;
}  // namespace aql_profile

namespace {

// Pre-fix: aliases Logger::message_[tid].
const char* buggy_error_string() { return aql_profile::Logger::LastMessage().c_str(); }

// Post-fix: thread-local copy; stable until next call on the same thread.
const char* fixed_error_string() {
  thread_local std::string last_error_copy;
  last_error_copy = aql_profile::Logger::LastMessage();
  return last_error_copy.c_str();
}

}  // namespace

class AqlprofileErrorStringDangleTest : public ::testing::Test {
 protected:
  void SetUp() override { aql_profile::Logger::Destroy(); }
  void TearDown() override { aql_profile::Logger::Destroy(); }
};

// Pins the pre-fix aliasing behavior.
TEST_F(AqlprofileErrorStringDangleTest, BuggyPatternAliasesLogger) {
  ERR_LOGGING << "first-error-message-alpha";

  const char* p = buggy_error_string();
  const std::string s1 = p;
  ASSERT_NE(s1.find("first-error-message-alpha"), std::string::npos);

  ERR_LOGGING << "second-error-message-beta";

  const std::string pview(p);
  EXPECT_NE(pview, s1);
  EXPECT_NE(pview.find("second-error-message-beta"), std::string::npos);
}

// Fix: pointer stable until next call on the same thread.
TEST_F(AqlprofileErrorStringDangleTest, FixedPatternIsStableUntilNextCall) {
  ERR_LOGGING << "first-error-message-alpha";

  const char* p = fixed_error_string();
  const std::string s1 = p;
  ASSERT_NE(s1.find("first-error-message-alpha"), std::string::npos);

  ERR_LOGGING << "second-error-message-beta";

  const std::string pview(p);
  EXPECT_EQ(pview, s1);
  EXPECT_NE(pview.find("first-error-message-alpha"), std::string::npos);

  const char* q = fixed_error_string();
  const std::string s2 = q;
  EXPECT_NE(s2.find("second-error-message-beta"), std::string::npos);
}
