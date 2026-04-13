// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Regression test for hsa_ven_amd_aqlprofile_error_string() pointer lifetime.
// The pre-fix implementation aliased Logger::message_[tid]; a subsequent log
// on the same thread invalidated the caller's pointer.

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "../logger.h"

// Static member definitions (header-only Logger; matches logger_tests.cpp).
namespace aql_profile
{
Logger::mutex_t Logger::mutex_;
Logger*         Logger::instance_ = nullptr;
}  // namespace aql_profile

// Mirror the two call-site patterns. We don't link aqlprofile-lib (requires
// HSA runtime), so we replicate the wrappers here against the same Logger.
namespace
{
// Pre-fix: aliases Logger::message_[tid].
const char*
buggy_error_string()
{
    return aql_profile::Logger::LastMessage().c_str();
}

// Post-fix: thread-local copy; stable until next call on the same thread.
const char*
fixed_error_string()
{
    thread_local std::string last_error_copy;
    last_error_copy = aql_profile::Logger::LastMessage();
    return last_error_copy.c_str();
}
}  // namespace

class ErrorStringDangleTest : public ::testing::Test
{
protected:
    void SetUp() override { aql_profile::Logger::Destroy(); }
    void TearDown() override { aql_profile::Logger::Destroy(); }
};

// Pins the pre-fix aliasing behavior: a subsequent log on the same thread
// mutates what the caller's pointer reads. Production error paths omit the
// trailing Logger::endl, matched here.
TEST_F(ErrorStringDangleTest, BuggyPatternAliasesLogger)
{
    ERR_LOGGING << "first-error-message-alpha";

    const char*       p  = buggy_error_string();
    const std::string s1 = p;
    ASSERT_NE(s1.find("first-error-message-alpha"), std::string::npos);

    ERR_LOGGING << "second-error-message-beta";

    const std::string pview(p);
    EXPECT_NE(pview, s1);
    EXPECT_NE(pview.find("second-error-message-beta"), std::string::npos);
}

// Fix: pointer stable until next call on the same thread.
TEST_F(ErrorStringDangleTest, FixedPatternIsStableUntilNextCall)
{
    ERR_LOGGING << "first-error-message-alpha";

    const char*       p  = fixed_error_string();
    const std::string s1 = p;
    ASSERT_NE(s1.find("first-error-message-alpha"), std::string::npos);

    ERR_LOGGING << "second-error-message-beta";

    const std::string pview(p);
    EXPECT_EQ(pview, s1);
    EXPECT_NE(pview.find("first-error-message-alpha"), std::string::npos);

    // A second call on the same thread may invalidate p (documented).
    const char*       q  = fixed_error_string();
    const std::string s2 = q;
    EXPECT_NE(s2.find("second-error-message-beta"), std::string::npos);
}
