// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-instrument/dynamic_dependency_listing.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{
namespace detail = ::rocprofsys::instrument::detail;

using ::testing::_;
using ::testing::AllOf;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Field;
using ::testing::Gt;
using ::testing::InSequence;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Le;
using ::testing::NotNull;
using ::testing::Optional;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::SetArrayArgument;
using ::testing::SetErrnoAndReturn;
using ::testing::StrEq;
using ::testing::StrictMock;
using ::testing::Truly;

constexpr int   k_read_fd  = 3;
constexpr int   k_write_fd = 4;
constexpr auto  k_pipe_fds = std::array<int, 2>{ k_read_fd, k_write_fd };
constexpr auto  k_pipe = detail::pipe_fds{ .read_fd = k_read_fd, .write_fd = k_write_fd };
constexpr auto  k_exe_path  = std::string_view{ "/path/to/app" };
constexpr auto  k_timeout   = std::chrono::milliseconds{ 1000 };
constexpr pid_t k_child_pid = 1234;

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

/// Action for read(): copies data into the caller's buffer and returns its size.
/// Needed because read() writes through a void* buffer, and gmock's SetArrayArgument
/// (used for pipe() in the create_pipe tests) can't copy into void*.
auto
read_returns(std::string_view data)
{
    return [data](int /*descriptor*/, void* buf, size_t count) -> ssize_t {
        const auto size = std::min(data.size(), count);
        std::memcpy(buf, data.data(), size);
        return static_cast<ssize_t>(size);
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

/// Expects pipe() to succeed with k_pipe_fds, then fork() to return pid.
void
expect_pipe_and_fork(pid_t pid)
{
    EXPECT_CALL(*g_mock, pipe(NotNull()))
        .WillOnce(
            DoAll(SetArrayArgument<0>(k_pipe_fds.begin(), k_pipe_fds.end()), Return(0)));
    EXPECT_CALL(*g_mock, fork()).WillOnce(Return(pid));
}

/// Expects the parent to read output from the pipe, then EOF.
void
expect_parent_reads(std::string_view output)
{
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns(output));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));
}

/// Action for waitpid(): reports status through the out-parameter and returns the child.
auto
waitpid_reports(int status)
{
    return DoAll(SetArgPointee<1>(status), Return(k_child_pid));
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
    EXPECT_CALL(*g_mock, dup2(k_write_fd, STDOUT_FILENO)).WillOnce(Return(STDOUT_FILENO));
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, execve(StrEq(k_exe_path), is_exe_argv, envp.data()))
        .WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, exit_immediately(detail::k_exec_failure_status));

    EXPECT_THROW(detail::exec_with_stdout_to_pipe<mock_posix_backend>(
                     std::string{ k_exe_path }, k_pipe, envp),
                 child_exited);
}

TEST_F(dynamic_dependency_listing_test, exec_exits_127_without_exec_when_dup2_fails)
{
    const auto envp = std::vector<char*>{ nullptr };

    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_read_fd));
    EXPECT_CALL(*g_mock, dup2(k_write_fd, STDOUT_FILENO)).WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, exit_immediately(detail::k_exec_failure_status));

    EXPECT_THROW(detail::exec_with_stdout_to_pipe<mock_posix_backend>(
                     std::string{ k_exe_path }, k_pipe, envp),
                 child_exited);
}
TEST_F(dynamic_dependency_listing_test, read_pipe_concatenates_chunks_until_eof)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("abc"));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("def"));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "abcdef");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_empty_string_on_immediate_eof)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_polls_read_fd_with_bounded_timeout)
{
    const auto is_read_fd_for_input =
        Pointee(AllOf(Field(&pollfd::fd, k_read_fd), Field(&pollfd::events, POLLIN)));

    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock,
                poll(is_read_fd_for_input, 1, AllOf(Gt(0), Le(k_timeout.count()))))
        .WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_nullopt_on_poll_timeout)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, read_pipe_retries_poll_after_eintr)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(SetErrnoAndReturn(EINTR, -1));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("abc"));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "abc");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_nullopt_on_poll_error)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(SetErrnoAndReturn(EBADF, -1));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, read_pipe_retries_read_after_eintr)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(SetErrnoAndReturn(EINTR, -1));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(read_returns("abc"));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), "abc");
}

TEST_F(dynamic_dependency_listing_test, read_pipe_returns_nullopt_on_read_error)
{
    const InSequence seq;
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(1));
    EXPECT_CALL(*g_mock, read(k_read_fd, _, _)).WillOnce(SetErrnoAndReturn(EIO, -1));
    EXPECT_CALL(*g_mock, close(k_read_fd));

    EXPECT_EQ(run_read_pipe_until_eof(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test,
       read_pipe_returns_nullopt_without_polling_when_timeout_is_zero)
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
    EXPECT_CALL(*g_mock, waitpid(k_child_pid, NotNull(), 0))
        .WillOnce(waitpid_reports(W_EXITCODE(0, 0)));

    EXPECT_THAT(run_read_dynamic_dependencies(),
                Optional(ElementsAreArray(k_loader_paths)));
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_returns_empty_list_when_output_has_no_paths)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads("\tlinux-vdso.so.1 (0x00007ffd)\n");
    EXPECT_CALL(*g_mock, waitpid(k_child_pid, NotNull(), 0))
        .WillOnce(waitpid_reports(W_EXITCODE(0, 0)));

    EXPECT_THAT(run_read_dynamic_dependencies(), Optional(IsEmpty()));
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_kills_child_and_returns_nullopt_on_timeout)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, poll(_, 1, _)).WillOnce(Return(0));
    EXPECT_CALL(*g_mock, close(k_read_fd));
    EXPECT_CALL(*g_mock, kill(k_child_pid, SIGKILL));
    EXPECT_CALL(*g_mock, waitpid(k_child_pid, NotNull(), 0))
        .WillOnce(waitpid_reports(W_EXITCODE(0, SIGKILL)));

    EXPECT_EQ(run_read_dynamic_dependencies(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_returns_nullopt_when_child_exits_nonzero)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads(k_loader_output);
    EXPECT_CALL(*g_mock, waitpid(k_child_pid, NotNull(), 0))
        .WillOnce(waitpid_reports(W_EXITCODE(1, 0)));

    EXPECT_EQ(run_read_dynamic_dependencies(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test,
       read_deps_returns_nullopt_when_child_killed_by_signal)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads(k_loader_output);
    EXPECT_CALL(*g_mock, waitpid(k_child_pid, NotNull(), 0))
        .WillOnce(waitpid_reports(W_EXITCODE(0, SIGSEGV)));

    EXPECT_EQ(run_read_dynamic_dependencies(), std::nullopt);
}

TEST_F(dynamic_dependency_listing_test, read_deps_retries_waitpid_after_eintr)
{
    const InSequence seq;
    expect_pipe_and_fork(k_child_pid);
    expect_parent_reads(k_loader_output);
    EXPECT_CALL(*g_mock, waitpid(k_child_pid, NotNull(), 0))
        .WillOnce(SetErrnoAndReturn(EINTR, -1))
        .WillOnce(waitpid_reports(W_EXITCODE(0, 0)));

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
    EXPECT_CALL(*g_mock, dup2(k_write_fd, STDOUT_FILENO)).WillOnce(Return(STDOUT_FILENO));
    EXPECT_CALL(*g_mock, close(k_write_fd));
    EXPECT_CALL(*g_mock, execve(StrEq(k_exe_path), _, has_listing_var))
        .WillOnce(Return(-1));
    EXPECT_CALL(*g_mock, exit_immediately(detail::k_exec_failure_status));

    EXPECT_THROW(run_read_dynamic_dependencies(), child_exited);
}
}  // namespace
