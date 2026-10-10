// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/delimit.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <optional>
#include <signal.h>
#include <string>
#include <sys/poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocprofsys::instrument
{
/// Backend for read_dynamic_dependencies that makes the real system calls
struct posix_backend
{
    static int   pipe(int* fds) { return ::pipe(fds); }
    static pid_t fork() { return ::fork(); }
    static int   close(int descriptor) { return ::close(descriptor); }
    static int   dup2(int old_fd, int new_fd) { return ::dup2(old_fd, new_fd); }
    static int   execve(const char* path, char* const* argv, char* const* envp)
    {
        return ::execve(path, argv, envp);
    }
    // Must stay ::_exit, not std::exit: the forked child must not run the parent's
    // atexit handlers or flush its stdio buffers
    [[noreturn]] static void exit_immediately(int status) { ::_exit(status); }
    static int               poll(pollfd* fds, nfds_t nfds, int timeout_ms)
    {
        return ::poll(fds, nfds, timeout_ms);
    }
    static ssize_t read(int descriptor, void* buf, size_t count)
    {
        return ::read(descriptor, buf, count);
    }
    static int   kill(pid_t pid, int sig) { return ::kill(pid, sig); }
    static pid_t waitpid(pid_t pid, int* status, int options)
    {
        return ::waitpid(pid, status, options);
    }
};

namespace detail
{
inline constexpr int k_invalid_fd          = -1;
inline constexpr int k_exec_failure_status = 127;

struct pipe_fds
{
    int read_fd  = k_invalid_fd;
    int write_fd = k_invalid_fd;
};

template <typename Backend>
[[nodiscard]] std::optional<pipe_fds>
create_pipe()
{
    auto fds = std::array<int, 2>{ k_invalid_fd, k_invalid_fd };
    if(Backend::pipe(fds.data()) != 0)
    {
        return std::nullopt;
    }

    return pipe_fds{ .read_fd = fds.at(0), .write_fd = fds.at(1) };
}

/// The current environment plus LD_TRACE_LOADED_OBJECTS, NULL-terminated.
/// The result points into environ - only valid while the environment is left untouched.
[[nodiscard]] inline std::vector<char*>
create_dependency_listing_envp()
{
    auto envp = std::vector<char*>{};

    // Copy ptrs to existing environment until nullptr terminator is reached
    for(char* const* var = ::environ; var != nullptr && *var != nullptr; ++var)
    {
        envp.emplace_back(*var);
    }
    // Add the loader's listing env var and nullptr terminator
    envp.emplace_back(const_cast<char*>("LD_TRACE_LOADED_OBJECTS=1"));
    envp.emplace_back(nullptr);

    return envp;
}

/// Redirect stdout to the pipe's write end, then become exe_path with the given
/// environment. Never returns: on success the process image is replaced, on failure exits
/// with 127. Runs in the child between fork and exec, where only async-signal-safe
/// calls are allowed (no allocation, no environment changes).
template <typename Backend>
[[noreturn]] void
exec_with_stdout_to_pipe(const std::string& exe_path, pipe_fds fds,
                         const std::vector<char*>& envp)
{
    Backend::close(fds.read_fd);
    if(Backend::dup2(fds.write_fd, STDOUT_FILENO) < 0)
    {
        Backend::exit_immediately(k_exec_failure_status);
    }
    Backend::close(fds.write_fd);

    auto argv = std::array<char*, 2>{ const_cast<char*>(exe_path.c_str()), nullptr };
    Backend::execve(exe_path.c_str(), argv.data(), envp.data());

    // Return from execve means it failed
    Backend::exit_immediately(k_exec_failure_status);
}

/// Read the pipe until EOF and return everything received. Closes both ends and
/// invalidates fds. Returns std::nullopt on error or if EOF is not reached before
/// timeout.
template <typename Backend>
[[nodiscard]] std::optional<std::string>
read_pipe_until_eof(pipe_fds& fds, std::chrono::milliseconds timeout)
{
    constexpr size_t k_read_buffer_size = 4096;

    // Close our copy of the write end, otherwise read() never sees EOF
    Backend::close(fds.write_fd);
    fds.write_fd = k_invalid_fd;

    const auto deadline = std::chrono::steady_clock::now() + timeout;

    auto out = std::string{};
    auto buf = std::array<char, k_read_buffer_size>{};
    auto eof = false;
    while(!eof)
    {
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if(remaining.count() <= 0)
        {
            break;
        }

        // Wait until the pipe is readable (data or EOF) or the deadline passes, so read()
        // below never blocks
        auto      pfd   = pollfd{ .fd = fds.read_fd, .events = POLLIN, .revents = 0 };
        const int ready = Backend::poll(&pfd, 1, static_cast<int>(remaining.count()));
        if(ready < 0 && errno == EINTR)
        {
            continue;  // interrupted: retry with the time left
        }
        if(ready <= 0)
        {
            break;  // timed out or poll error
        }

        // poll said readable, so read() should return at once: >0 data, 0 EOF, <0 error
        const auto bytes_read = Backend::read(fds.read_fd, buf.data(), buf.size());
        if(bytes_read > 0)
        {
            out.append(buf.data(), static_cast<std::size_t>(bytes_read));
        }
        else if(bytes_read == 0)
        {
            eof = true;
        }
        else if(errno != EINTR)  // EINTR: retry with time left
        {
            break;  // read error
        }
    }

    Backend::close(fds.read_fd);
    fds.read_fd = k_invalid_fd;

    // Anything but EOF (timeout, error) means the output can't be trusted
    return eof ? std::optional{ std::move(out) } : std::nullopt;
}
}  // namespace detail

/// Get the shared libraries that exe_path actually loads at startup, as absolute paths.
/// exe_path must be dynamically linked: the listing works by running it with
/// LD_TRACE_LOADED_OBJECTS=1, which makes the dynamic loader print every resolved
/// dependency and exit before main(). A static executable has no loader to intercept it,
/// so it would run for real instead.
template <typename Backend = posix_backend>
[[nodiscard]] std::optional<std::vector<std::string>>
read_dynamic_dependencies(const std::string& exe_path, std::chrono::milliseconds timeout)
{
    auto fds = detail::create_pipe<Backend>();
    if(!fds)
    {
        return std::nullopt;
    }

    // Build environment for listing deps. LD_LIBRARY_PATH and LD_PRELOAD are copied from
    // current env, so the later-listed lib paths will match what a real run would load.
    // Built before forking: between fork and exec the child must not allocate or modify
    // the environment.
    const auto listing_envp = detail::create_dependency_listing_envp();

    const auto pid = Backend::fork();
    if(pid < 0)
    {
        Backend::close(fds->read_fd);
        Backend::close(fds->write_fd);
        return std::nullopt;
    }

    if(pid == 0)  // we are in the child
    {
        detail::exec_with_stdout_to_pipe<Backend>(exe_path, *fds, listing_envp);
    }

    // If we are here - we are in the parent

    const auto loader_output = detail::read_pipe_until_eof<Backend>(*fds, timeout);
    if(!loader_output)
    {
        // Timeout or read error: the child may still be running. Kill it so the waitpid
        // below returns.
        Backend::kill(pid, SIGKILL);
    }

    // Reap the child
    auto status = 0;
    while(Backend::waitpid(pid, &status, 0) < 0 && errno == EINTR)
    {
        // Retry if interrupted by a signal
    }
    // Included <sys/wait.h> is correct for W* macros, but <stdlib.h> arrives earlier
    // through the standard library headers, so glibc defines the W* macros there,
    // <sys/wait.h> skips them, and clang-tidy reports no direct include.
    // NOLINTNEXTLINE(misc-include-cleaner)
    if(!loader_output || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        return std::nullopt;
    }

    // Loader output lines we need:
    //   <soname> => <abs path> (<addr>)
    //   <abs path> (<addr>)
    // Parse and keep only abs paths.
    auto out = rocprofsys::delimit(*loader_output, " \n\t=>");
    std::erase_if(out, [](const std::string& item) { return !item.starts_with('/'); });
    return out;
}
}  // namespace rocprofsys::instrument
