// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <signal.h>
#include <sys/poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rocprofsys::backends::posix
{
/// Backend for read_dynamic_dependencies that makes the real system calls. Every POSIX
/// type, constant and macro the listing needs stays behind this struct.
struct backend
{
    // Define pid_type as pid_t; "pid_type = pid_t" triggers clang-tidy
    using pid_type = decltype(::fork());

    static int      pipe(int* fds) { return ::pipe(fds); }
    static pid_type fork() { return ::fork(); }
    static int      close(int descriptor) { return ::close(descriptor); }
    /// Make descriptor this process's stdout. Returns <0 on error.
    static int redirect_stdout(int descriptor)
    {
        return ::dup2(descriptor, STDOUT_FILENO);
    }
    static int execve(const char* path, char* const* argv, char* const* envp)
    {
        return ::execve(path, argv, envp);
    }
    // Must stay ::_exit, not std::exit: the forked child must not run the parent's
    // atexit handlers or flush its stdio buffers
    [[noreturn]] static void exit_immediately(int status) { ::_exit(status); }
    /// Wait until descriptor has data or EOF to read.
    /// Returns >0 when readable, 0 on timeout, <0 on error.
    static int wait_readable(int descriptor, std::chrono::milliseconds timeout)
    {
        auto pfd = pollfd{ .fd = descriptor, .events = POLLIN, .revents = 0 };
        return ::poll(&pfd, 1, static_cast<int>(timeout.count()));
    }
    static ssize_t read(int descriptor, void* buf, size_t count)
    {
        return ::read(descriptor, buf, count);
    }
    /// Kill pid with a signal it cannot catch or ignore
    static int force_kill(pid_type pid) { return ::kill(pid, SIGKILL); }
    /// Block until pid ends and store its status. Returns <0 on error.
    static pid_type wait_for_exit(pid_type pid, int* status)
    {
        return ::waitpid(pid, status, 0);
    }
    /// True if a wait_for_exit status reports a normal exit with code 0
    static bool exited_successfully(int status)
    {
        // Included <sys/wait.h> is correct for W* macros, but <stdlib.h> arrives earlier
        // through the standard library headers, so glibc defines the W* macros there,
        // <sys/wait.h> skips them, and clang-tidy reports no direct include.
        // NOLINTNEXTLINE(misc-include-cleaner)
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    /// True if the last failed call was interrupted by a signal and can be retried
    static bool last_call_interrupted() { return errno == EINTR; }
    /// The process environment: NULL-terminated, may itself be NULL
    static char* const* environ() { return ::environ; }
};
}  // namespace rocprofsys::backends::posix
