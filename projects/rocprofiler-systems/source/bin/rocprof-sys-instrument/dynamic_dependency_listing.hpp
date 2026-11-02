// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/delimit.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::instrument
{
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
/// The result points into Backend::environ() - only valid while the environment is
/// left untouched.
template <typename Backend>
[[nodiscard]] std::vector<char*>
create_dependency_listing_envp()
{
    auto envp = std::vector<char*>{};

    // Copy ptrs to existing environment until nullptr terminator is reached
    for(char* const* var = Backend::environ(); var != nullptr && *var != nullptr; ++var)
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
    if(Backend::redirect_stdout(fds.write_fd) < 0)
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
        const int ready = Backend::wait_readable(fds.read_fd, remaining);
        if(ready < 0 && Backend::last_call_interrupted())
        {
            continue;  // interrupted: retry with the time left
        }
        if(ready <= 0)
        {
            break;  // timed out or wait error
        }

        // Pipe is readable, so read() should return at once: >0 data, 0 EOF, <0 error
        const auto bytes_read = Backend::read(fds.read_fd, buf.data(), buf.size());
        if(bytes_read > 0)
        {
            out.append(buf.data(), static_cast<std::size_t>(bytes_read));
        }
        else if(bytes_read == 0)
        {
            eof = true;
        }
        else if(!Backend::last_call_interrupted())  // interrupted: retry with time left
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
template <typename Backend>
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
    const auto listing_envp = detail::create_dependency_listing_envp<Backend>();

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
        // Timeout or read error: the child may still be running. Kill it so wait_for_exit
        // below returns.
        Backend::force_kill(pid);
    }

    // Reap the child
    auto status = 0;
    while(Backend::wait_for_exit(pid, &status) < 0 && Backend::last_call_interrupted())
    {
        // Retry if interrupted by a signal
    }
    if(!loader_output || !Backend::exited_successfully(status))
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
