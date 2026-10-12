// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-instrument/dynamic_dependency_listing.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace detail = ::rocprofsys::instrument::detail;

using ::testing::_;
using ::testing::AllOf;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Gt;
using ::testing::InSequence;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Le;
using ::testing::NotNull;
using ::testing::Optional;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::SetArrayArgument;
using ::testing::StrEq;
using ::testing::StrictMock;
using ::testing::Truly;

constexpr int  k_read_fd  = 3;
constexpr int  k_write_fd = 4;
constexpr auto k_pipe_fds = std::array<int, 2>{ k_read_fd, k_write_fd };
constexpr auto k_pipe = detail::pipe_fds{ .read_fd = k_read_fd, .write_fd = k_write_fd };
constexpr auto k_exe_path  = std::string_view{ "/path/to/app" };
constexpr auto k_timeout   = std::chrono::milliseconds{ 1000 };
constexpr int  k_child_pid = 1234;
// Opaque to the code under test: only passed from wait_for_exit to exited_successfully
constexpr int k_child_status = 42;

// What the loader prints for LD_TRACE_LOADED_OBJECTS=1, and the paths parsed from it
constexpr auto k_loader_output =
    std::string_view{ "\tlinux-vdso.so.1 (0x00007ffd)\n"
                      "\tlibz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f01)\n"
                      "\tlibmissing.so => not found\n"
                      "\t/lib64/ld-linux-x86-64.so.2 (0x00007f02)\n" };
constexpr auto k_loader_paths =
    std::array<std::string_view, 2>{ "/lib/x86_64-linux-gnu/libz.so.1",
                                     "/lib64/ld-linux-x86-64.so.2" };

struct gmock_posix
{
    MOCK_METHOD(int, pipe, (int* fds));
    MOCK_METHOD(int, fork, ());
    MOCK_METHOD(int, close, (int descriptor));
    MOCK_METHOD(int, redirect_stdout, (int descriptor));
    MOCK_METHOD(int, execve, (const char* path, char* const* argv, char* const* envp));
    MOCK_METHOD(void, exit_immediately, (int status));
    MOCK_METHOD(int, wait_readable, (int descriptor, std::chrono::milliseconds timeout));
    MOCK_METHOD(std::ptrdiff_t, read, (int descriptor, void* buf, std::size_t count));
    MOCK_METHOD(int, force_kill, (int pid));
    MOCK_METHOD(int, wait_for_exit, (int pid, int* status));
    MOCK_METHOD(bool, exited_successfully, (int status));
    MOCK_METHOD(bool, last_call_interrupted, ());
    MOCK_METHOD(char* const*, environ, ());
};

std::unique_ptr<StrictMock<gmock_posix>> g_mock;

/// Thrown by mock_posix_backend::exit_immediately in place of ending the process.
struct child_exited : std::exception
{};

/// backends::posix::backend stand-in: every call forwards to g_mock.
struct mock_posix_backend
{
    using pid_type = int;

    static int      pipe(int* fds) { return g_mock->pipe(fds); }
    static pid_type fork() { return g_mock->fork(); }
    static int      close(int descriptor) { return g_mock->close(descriptor); }
    static int      redirect_stdout(int descriptor)
    {
        return g_mock->redirect_stdout(descriptor);
    }
    static int execve(const char* path, char* const* argv, char* const* envp)
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
    static int wait_readable(int descriptor, std::chrono::milliseconds timeout)
    {
        return g_mock->wait_readable(descriptor, timeout);
    }
    static std::ptrdiff_t read(int descriptor, void* buf, std::size_t count)
    {
        return g_mock->read(descriptor, buf, count);
    }
    static int      force_kill(pid_type pid) { return g_mock->force_kill(pid); }
    static pid_type wait_for_exit(pid_type pid, int* status)
    {
        return g_mock->wait_for_exit(pid, status);
    }
    static bool exited_successfully(int status)
    {
        return g_mock->exited_successfully(status);
    }
    static bool last_call_interrupted() { return g_mock->last_call_interrupted(); }
    static char* const* environ() { return g_mock->environ(); }
};

/// Action for read(): copies data into the caller's buffer and returns its size.
/// Needed because read() writes through a void* buffer, and gmock's SetArrayArgument
/// (used for pipe() in the create_pipe tests) can't copy into void*.
auto
read_returns(std::string_view data)
{
    return [data](int /*descriptor*/, void* buf, std::size_t count) -> std::ptrdiff_t {
        const auto size = std::min(data.size(), count);
        std::memcpy(buf, data.data(), size);
        return static_cast<std::ptrdiff_t>(size);
    };
}

/// Runs read_pipe_until_eof on a copy of k_pipe and checks that both ends were
/// invalidated - the function must do that on every path.
std::optional<std::string>
run_read_pipe_until_eof(std::chrono::milliseconds timeout = k_timeout)
{
    auto fds = k_pipe;
    auto out = detail::read_pipe_until_eof<mock_posix_backend>(fds, timeout);
    EXPECT_EQ(fds.read_fd, detail::k_invalid_fd);
    EXPECT_EQ(fds.write_fd, detail::k_invalid_fd);
    return out;
}

/// Expects pipe() to succeed with k_pipe_fds, then the environment to be read, then
/// fork() to return pid.
void
expect_pipe_and_fork(int pid)
{
    EXPECT_CALL(*g_mock, pipe(NotNull()))
        .WillOnce(
            DoAll(SetArrayArgument<0>(k_pipe_fds.begin(), k_pipe_fds.end()), Return(0)));
    // Environment contents are covered by the create_dependency_listing_envp tests
    EXPECT_CALL(*g_mock, environ()).WillOnce(Return(nullptr));
    EXPECT_CALL(*g_mock, fork()).WillOnce(Return(pid));
}

/// Expects the parent to read output from the pipe, then EOF.
void
expect_parent_reads(std::string_view output)
{
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns(output));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));
}

/// Action for wait_for_exit(): reports k_child_status through the out-parameter and
/// returns the child.
auto
wait_for_exit_reports_status()
{
    return DoAll(SetArgPointee<1>(k_child_status), Return(k_child_pid));
}

/// Expects the child to be reaped, and the backend to judge its status as success.
void
expect_child_reaped(bool success)
{
    EXPECT_CALL(*g_mock, wait_for_exit(k_child_pid, NotNull()))
        .WillOnce(wait_for_exit_reports_status());
    EXPECT_CALL(*g_mock, exited_successfully(k_child_status)).WillOnce(Return(success));
}

std::optional<std::vector<std::string>>
run_read_dynamic_dependencies()
{
    return rocprofsys::instrument::read_dynamic_dependencies<mock_posix_backend>(
        std::string{ k_exe_path }, k_timeout);
}

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
    const auto test_env = std::array<char*, 3>{ var_a.data(), var_b.data(), nullptr };
    EXPECT_CALL(*g_mock, environ()).WillOnce(Return(test_env.data()));

    const auto envp = detail::create_dependency_listing_envp<mock_posix_backend>();

    // Environment entries are the same pointers, not copies of the strings
    EXPECT_THAT(envp, ElementsAre(var_a.data(), var_b.data(),
                                  StrEq("LD_TRACE_LOADED_OBJECTS=1"), IsNull()));
}

TEST_F(dynamic_dependency_listing_test, envp_handles_null_environment)
{
    EXPECT_CALL(*g_mock, environ()).WillOnce(Return(nullptr));

    const auto envp = detail::create_dependency_listing_envp<mock_posix_backend>();

    EXPECT_THAT(envp, ElementsAre(StrEq("LD_TRACE_LOADED_OBJECTS=1"), IsNull()));
}

TEST_F(dynamic_dependency_listing_test, envp_handles_empty_environment)
{
    const auto test_env = std::array<char*, 1>{ nullptr };
    EXPECT_CALL(*g_mock, environ()).WillOnce(Return(test_env.data()));

    const auto envp = detail::create_dependency_listing_envp<mock_posix_backend>();

    EXPECT_THAT(envp, ElementsAre(StrEq("LD_TRACE_LOADED_OBJECTS=1"), IsNull()));
}

TEST_F(dynamic_dependency_listing_test, exec_redirects_stdout_then_execs_target)
{
    auto       var_a = std::string{ "ROCPROFSYS_TEST_A=1" };
    const auto envp  = std::vector<char*>{ var_a.data(), nullptr };
    // argv must be { k_exe_path, nullptr }
    const auto is_exe_argv = Truly([](char* const* argv) {
        return argv[0] != nullptr && std::string_view{ argv[0] } == k_exe_path &&
               argv[1] == nullptr;
    });

    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_read_fd));
    EXPECT_CALL(*g_mock, redirect_stdout(k_write_fd)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, execve(StrEq(k_exe_path), is_exe_argv, envp.data()))
        .WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, exit_immediately(detail::k_exec_failure_status));

    EXPECT_THROW(detail::exec_with_stdout_to_pipe<mock_posix_backend>(
                     std::string{ k_exe_path }, k_pipe, envp),
                 child_exited);
}

TEST_F(dynamic_dependency_listing_test, exec_exits_127_without_exec_when_redirect_fails)
{
    const auto envp = std::vector<char*>{ nullptr };

    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_read_fd));
    EXPECT_CALL(*g_mock, redirect_stdout(k_write_fd)).WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, exit_immediately(detail::k_exec_failure_status));

    EXPECT_THROW(detail::exec_with_stdout_to_pipe<mock_posix_backend>(
                     std::string{ k_exe_path }, k_pipe, envp),
                 child_exited);
}
TEST_F(dynamic_dependency_listing_test, read_pipe_concatenates_chunks_until_eof)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("abc"));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("def"));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "abcdef");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_empty_string_on_immediate_eof)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_waits_on_read_fd_with_bounded_timeout)
{
    const auto is_bounded_timeout =
        AllOf(Gt(std::chrono::milliseconds{ 0 }), Le(k_timeout));

    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, is_bounded_timeout))
        .WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_nullopt_on_wait_timeout)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, read_pipe_retries_wait_when_interrupted)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, last_call_interrupted()).WillOnce(Return(true));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("abc"));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "abc");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_nullopt_on_wait_error)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, last_call_interrupted()).WillOnce(Return(false));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, read_pipe_retries_read_when_interrupted)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, last_call_interrupted()).WillOnce(Return(true));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("abc"));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "abc");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_nullopt_on_read_error)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, last_call_interrupted()).WillOnce(Return(false));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test,
       read_pipe_returns_nullopt_without_waiting_when_timeout_is_zero)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(std::chrono::milliseconds{ 0 }), std::nullopt);
}
TEST_F(dynamic_dependency_listing_test, read_deps_returns_nullopt_when_pipe_fails)
{
    EXPECT_CALL(*g_mock, pipe(NotNull())).WillOnce(Return(-1));

    EXPECT_EQ(run_read_dynamic_dependencies(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, read_deps_closes_both_fds_when_fork_fails)
{
    const InSequence seq;
    expect_pipe_and_fork(-1);
    EXPECT_CALL(*g_mock, close(k_read_fd));
    EXPECT_CALL(*g_mock, close(k_write_fd));

    EXPECT_EQ(run_read_dynamic_dependencies(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_returns_absolute_paths_from_loader_output)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads(k_loader_output);
    expect_child_reaped(true);

    EXPECT_THAT(run_read_dynamic_dependencies(),
                Optional(ElementsAreArray(k_loader_paths)));
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_returns_empty_list_when_output_has_no_paths)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads("\tlinux-vdso.so.1 (0x00007ffd)\n");
    expect_child_reaped(true);

    EXPECT_THAT(run_read_dynamic_dependencies(), Optional(IsEmpty()));
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_kills_child_and_returns_nullopt_on_timeout)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, wait_readable(k_read_fd, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));
    EXPECT_CALL(*g_mock, force_kill(k_child_pid));
    // No exited_successfully call: the missing output alone decides the result
    EXPECT_CALL(*g_mock, wait_for_exit(k_child_pid, NotNull()))
        .WillOnce(wait_for_exit_reports_status());

    EXPECT_EQ(run_read_dynamic_dependencies(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_returns_nullopt_when_child_not_successful)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads(k_loader_output);
    expect_child_reaped(false);

    EXPECT_EQ(run_read_dynamic_dependencies(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, read_deps_retries_wait_for_exit_when_interrupted)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads(k_loader_output);
    EXPECT_CALL(*g_mock, wait_for_exit(k_child_pid, NotNull())).WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, last_call_interrupted()).WillOnce(Return(true));
    expect_child_reaped(true);

    EXPECT_THAT(run_read_dynamic_dependencies(),
                Optional(ElementsAreArray(k_loader_paths)));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_F(dynamic_dependency_listing_test, read_deps_child_execs_target_with_listing_env)
{
    const auto has_listing_var = Truly([](char* const* envp) {
        while(*envp != nullptr &&
              std::string_view{ *envp } != "LD_TRACE_LOADED_OBJECTS=1")
        {
            ++envp;
        }
        return *envp != nullptr;
    });

    const InSequence seq;
    expect_pipe_and_fork(0);
    EXPECT_CALL(*g_mock, close(k_read_fd));
    EXPECT_CALL(*g_mock, redirect_stdout(k_write_fd)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, execve(StrEq(k_exe_path), _, has_listing_var))
        .WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, exit_immediately(detail::k_exec_failure_status));

    EXPECT_THROW(run_read_dynamic_dependencies(), child_exited);
}
}  // namespace
