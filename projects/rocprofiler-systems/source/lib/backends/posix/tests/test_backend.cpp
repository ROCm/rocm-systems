// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/posix/backend.hpp"

#include <gtest/gtest.h>

#include <cerrno>
#include <signal.h>
#include <sys/wait.h>

// Covers the posix backend helpers that are pure functions of their input. The system
// call wrappers are mocked in test_dynamic_dependency_listing.cpp.
namespace
{
using ::rocprofsys::backends::posix::backend;

TEST(posix_backend_test, exited_successfully_on_exit_code_zero)
{
    EXPECT_TRUE(backend::exited_successfully(W_EXITCODE(0, 0)));
}

TEST(posix_backend_test, not_exited_successfully_on_nonzero_exit_code)
{
    EXPECT_FALSE(backend::exited_successfully(W_EXITCODE(1, 0)));
}

TEST(posix_backend_test, not_exited_successfully_when_killed_by_signal)
{
    EXPECT_FALSE(backend::exited_successfully(W_EXITCODE(0, SIGKILL)));
}

TEST(posix_backend_test, last_call_interrupted_when_errno_is_eintr)
{
    errno = EINTR;
    EXPECT_TRUE(backend::last_call_interrupted());
}

TEST(posix_backend_test, last_call_not_interrupted_on_other_errors)
{
    errno = EIO;
    EXPECT_FALSE(backend::last_call_interrupted());
}
}  // namespace
