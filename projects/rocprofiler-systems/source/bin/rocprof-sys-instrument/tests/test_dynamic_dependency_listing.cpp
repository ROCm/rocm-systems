// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-instrument/dynamic_dependency_listing.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <sys/poll.h>
#include <sys/types.h>
#include <unistd.h>

namespace
{
namespace detail = ::rocprofsys::instrument::detail;

using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SetArrayArgument;
using ::testing::StrEq;
using ::testing::StrictMock;

constexpr int  k_read_fd  = 3;
constexpr int  k_write_fd = 4;
constexpr auto k_pipe_fds = std::array<int, 2>{ k_read_fd, k_write_fd };

struct gmock_posix
{
    MOCK_METHOD(int, pipe, (int* fds));
    MOCK_METHOD(pid_t, fork, ());
    MOCK_METHOD(int, close, (int descriptor));
    MOCK_METHOD(int, dup2, (int old_fd, int new_fd));
    MOCK_METHOD(int, execve, (const char* path, char* const* argv, char* const* envp));
    MOCK_METHOD(void, exit_immediately, (int status));
    MOCK_METHOD(int, poll, (pollfd * fds, nfds_t nfds, int timeout_ms));
    MOCK_METHOD(ssize_t, read, (int descriptor, void* buf, size_t count));
    MOCK_METHOD(int, kill, (pid_t pid, int sig));
    MOCK_METHOD(pid_t, waitpid, (pid_t pid, int* status, int options));
};

std::unique_ptr<StrictMock<gmock_posix>> g_mock;

/// Thrown by mock_posix_backend::exit_immediately in place of ending the process.
struct child_exited : std::exception
{};

/// posix_backend stand-in: every call forwards to g_mock.
struct mock_posix_backend
{
    static int   pipe(int* fds) { return g_mock->pipe(fds); }
    static pid_t fork() { return g_mock->fork(); }
    static int   close(int descriptor) { return g_mock->close(descriptor); }
    static int   dup2(int old_fd, int new_fd) { return g_mock->dup2(old_fd, new_fd); }
    static int   execve(const char* path, char* const* argv, char* const* envp)
    {
        return g_mock->execve(path, argv, envp);
    }
    // The code under test relies on this never returning ([[noreturn]]), but the mock
    // does return - so throw to leave the code under test
    [[noreturn]] static void exit_immediately(int status)
    {
        g_mock->exit_immediately(status);
        throw child_exited{};
    }
    static int poll(pollfd* fds, nfds_t nfds, int timeout_ms)
    {
        return g_mock->poll(fds, nfds, timeout_ms);
    }
    static ssize_t read(int descriptor, void* buf, size_t count)
    {
        return g_mock->read(descriptor, buf, count);
    }
    static int   kill(pid_t pid, int sig) { return g_mock->kill(pid, sig); }
    static pid_t waitpid(pid_t pid, int* status, int options)
    {
        return g_mock->waitpid(pid, status, options);
    }
};

/// Points ::environ at a test-owned array and restores the real one on scope exit.
class environ_guard
{
public:
    explicit environ_guard(char** replacement)
    : m_saved{ ::environ }
    {
        ::environ = replacement;
    }
    ~environ_guard() { ::environ = m_saved; }

    environ_guard(const environ_guard&)            = delete;
    environ_guard& operator=(const environ_guard&) = delete;
    environ_guard(environ_guard&&)                 = delete;
    environ_guard& operator=(environ_guard&&)      = delete;

private:
    char** m_saved;
};

// NOLINTNEXTLINE(readability-identifier-naming)
class dynamic_dependency_listing_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock = std::make_unique<StrictMock<gmock_posix>>(); }
    void TearDown() override { g_mock.reset(); }
};

TEST_F(dynamic_dependency_listing_test, create_pipe_returns_fds_on_success)
{
    EXPECT_CALL(*g_mock, pipe(NotNull()))
        .WillOnce(
            DoAll(SetArrayArgument<0>(k_pipe_fds.begin(), k_pipe_fds.end()), Return(0)));

    const auto fds = detail::create_pipe<mock_posix_backend>();

    ASSERT_TRUE(fds.has_value());
    EXPECT_EQ(fds->read_fd, k_read_fd);
    EXPECT_EQ(fds->write_fd, k_write_fd);
}

TEST_F(dynamic_dependency_listing_test, create_pipe_returns_nullopt_when_pipe_fails)
{
    EXPECT_CALL(*g_mock, pipe(NotNull())).WillOnce(Return(-1));

    EXPECT_EQ(detail::create_pipe<mock_posix_backend>(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, envp_is_environment_plus_listing_var_and_null)
{
    auto       var_a    = std::string{ "ROCPROFSYS_TEST_A=1" };
    auto       var_b    = std::string{ "ROCPROFSYS_TEST_B=2" };
    auto       test_env = std::array<char*, 3>{ var_a.data(), var_b.data(), nullptr };
    const auto guard    = environ_guard{ test_env.data() };

    const auto envp = detail::create_dependency_listing_envp();

    // Environment entries are the same pointers, not copies of the strings
    EXPECT_THAT(envp, ElementsAre(var_a.data(), var_b.data(),
                                  StrEq("LD_TRACE_LOADED_OBJECTS=1"), IsNull()));
}

TEST_F(dynamic_dependency_listing_test, envp_handles_null_environ)
{
    const auto guard = environ_guard{ nullptr };

    const auto envp = detail::create_dependency_listing_envp();

    EXPECT_THAT(envp, ElementsAre(StrEq("LD_TRACE_LOADED_OBJECTS=1"), IsNull()));
}

TEST_F(dynamic_dependency_listing_test, envp_handles_empty_environ)
{
    auto       test_env = std::array<char*, 1>{ nullptr };
    const auto guard    = environ_guard{ test_env.data() };

    const auto envp = detail::create_dependency_listing_envp();

    EXPECT_THAT(envp, ElementsAre(StrEq("LD_TRACE_LOADED_OBJECTS=1"), IsNull()));
}
}  // namespace
