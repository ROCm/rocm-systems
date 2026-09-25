// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "common/path.hpp"
#include "common/string_utility.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <rocprofiler-sdk-rocattach/rocattach.h>

namespace
{
using rocprofsys::utility::string::to_integral;

struct session
{
    std::size_t id  = 0;
    pid_t       pid = 0;
};

struct active_sessions
{
public:
    const session& add(pid_t pid)
    {
        return m_sessions.emplace_back(session{ .id = ++m_last_id, .pid = pid });
    }

    [[nodiscard]] std::optional<session> find(std::size_t session_id) const
    {
        const auto itr = std::ranges::find(m_sessions, session_id, &session::id);
        if(itr == m_sessions.end())
        {
            return std::nullopt;
        }
        return *itr;
    }

    void remove(std::size_t session_id)
    {
        std::erase_if(m_sessions, [session_id](const session& entry) {
            return entry.id == session_id;
        });
    }

    [[nodiscard]] const std::vector<session>& active() const noexcept
    {
        return m_sessions;
    }

private:
    std::vector<session> m_sessions;
    std::size_t          m_last_id = 0;
};

struct attach_options
{
    std::vector<pid_t>       pids;
    std::string              output_path;
    std::vector<std::string> profile_format;
};

void
print_usage(const char* prog_name)
{
    std::cout
        << "Usage: " << prog_name << " -p <pid>[,<pid>...] [OPTIONS]\n"
        << "\n"
        << "Attach to one or more running processes for profiling.\n"
        << "\n"
        << "Options:\n"
        << "  -p, --pid PID[,PID,...]\n"
        << "                       Process ID(s) to attach to (required, repeatable)\n"
        << "  -o, --output PATH    Output path for profiling results\n"
        << "  -F, --format FORMAT[,FORMAT,...]\n"
        << "                       Output format(s): perfetto, rocpd\n"
        << "  -h, --help           Show this help message\n"
        << "\n"
        << "Environment variables:\n"
        << "  ROCPROFSYS_OUTPUT_PATH       Output directory for profiling data\n"
        << "  ROCPROFSYS_TRACE             Enable perfetto trace output\n"
        << "  ROCPROFSYS_USE_ROCPD         Enable rocpd database output\n"
        << "  ROCPROF_ATTACH_TOOL_LIBRARY  Path to the tool library\n"
        << "\n"
        << "Once attached, enter a session id to detach from that process, 'a' to\n"
        << "detach from all of them, or ENTER when only one session is left.\n";
}

void
setup_session_env()
{
    const auto* output_use_current_time_env_name =
        rocprofsys::env_vars::OUTPUT_USE_CURRENT_TIME;
    const auto* reattach_add_session_id_env_name =
        rocprofsys::env_vars::REATTACH_ADD_SESSION_ID;

    // enable the use of the current time for the output path
    setenv(output_use_current_time_env_name, "true", 1);

    // enable the re-attach to add a session ID to the output path
    setenv(reattach_add_session_id_env_name, "true", 1);
}

/**
 * Picks the tool library path, as seen by process @p pid: the user's
 * ROCPROF_ATTACH_TOOL_LIBRARY, else this installation's library if the target can see it,
 * else the library shipped next to the target's own librocprofiler-register (e.g. inside
 * a container with its own ROCm installation).
 */
std::optional<std::string>
resolve_tool_library(pid_t pid, const std::optional<std::string>& user_library)
{
    namespace path = rocprofsys::common::path;

    if(user_library)
    {
        if(!path::is_missing_in_target(pid, *user_library))
        {
            return user_library;
        }
        LOG_ERROR("Tool library '{}' from ROCPROF_ATTACH_TOOL_LIBRARY does not exist in "
                  "the mount namespace of process {}.",
                  *user_library, pid);
        return std::nullopt;
    }

    const auto own_library = path::get_internal_libpath("librocprof-sys-dl.so");
    if(!path::is_missing_in_target(pid, own_library))
    {
        return own_library;
    }

    if(auto target_library = path::find_library_in_loaded_dir(
           pid, "librocprof-sys-dl.so", "librocprofiler-register.so"))
    {
        LOG_INFO("Found tool library in the ROCm installation of process {}", pid);
        return target_library;
    }

    LOG_ERROR(
        "Tool library '{}' does not exist in the mount namespace of process {}, and "
        "no librocprof-sys-dl.so was found in its ROCm installation. Set "
        "ROCPROF_ATTACH_TOOL_LIBRARY to the path as seen by the target.",
        own_library, pid);
    return std::nullopt;
}

void
setup_output_env(const std::string& output_path)
{
    const auto* existing_output_path = getenv(rocprofsys::env_vars::OUTPUT_PATH);
    if(output_path.empty() && existing_output_path != nullptr)
    {
        LOG_INFO("Output path: {}", existing_output_path);
        return;
    }

    const auto* const pwd = getenv("PWD");
    const auto        output =
        output_path.empty() ? fmt::format("{}/rocprof-sys-output", pwd) : output_path;

    setenv(rocprofsys::env_vars::OUTPUT_PATH, output.c_str(), 1);
    LOG_INFO("Output path: {}", output);
}

void
setup_output_format_env(const std::vector<std::string>& formats)
{
    if(formats.empty())
    {
        return;
    }

    auto const has_format = [&formats](const std::string& fmt) {
        return std::ranges::find(formats, fmt) != formats.end();
    };

    // setenv("ROCPROFSYS_PROFILE", "false", 1);

    if(has_format("perfetto") || has_format("rocpd"))
    {
        setenv(rocprofsys::env_vars::TRACE, has_format("perfetto") ? "true" : "false", 1);
        setenv(rocprofsys::env_vars::USE_ROCPD, has_format("rocpd") ? "true" : "false",
               1);
    }

    LOG_INFO("Output format: {}", fmt::join(formats, " "));
}

bool
is_option(const char* arg, const char* short_opt, const char* long_opt)
{
    return std::strcmp(arg, short_opt) == 0 || std::strcmp(arg, long_opt) == 0;
}

const char*
consume_arg(int& i, int argc, char* argv[], const char* opt_name)
{
    if(i + 1 >= argc)
    {
        LOG_ERROR("{} requires an argument.", opt_name);
        std::exit(EXIT_FAILURE);
    }
    return argv[++i];
}

void
parse_pids(attach_options& opts, const char* arg)
{
    std::string       token;
    std::stringstream stream(arg);
    while(std::getline(stream, token, ','))
    {
        const auto pid = to_integral<pid_t>(token);
        if(!pid || *pid <= 0)
        {
            LOG_ERROR("Invalid PID '{}'. PIDs must be positive integers.", token);
            std::exit(EXIT_FAILURE);
        }

        if(std::ranges::find(opts.pids, *pid) != opts.pids.end())
        {
            LOG_WARNING("PID {} was given more than once; attaching to it once.", *pid);
        }
        else
        {
            opts.pids.push_back(*pid);
        }
    }
}

void
parse_formats(attach_options& opts, const char* arg)
{
    std::string       token;
    std::stringstream ss(arg);
    while(std::getline(ss, token, ','))
    {
        if(token == "perfetto" || token == "rocpd")
        {
            opts.profile_format.push_back(token);
        }
        else
        {
            LOG_ERROR("Invalid format '{}'. Valid options: perfetto, rocpd", token);
            std::exit(EXIT_FAILURE);
        }
    }
}

attach_options
parse_args(int argc, char* argv[])
{
    attach_options opts;

    for(int i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if(is_option(arg, "-h", "--help"))
        {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }

        if(is_option(arg, "-p", "--pid"))
        {
            parse_pids(opts, consume_arg(i, argc, argv, "-p/--pid"));
            continue;
        }

        if(is_option(arg, "-o", "--output"))
        {
            opts.output_path = consume_arg(i, argc, argv, "-o/--output");
            continue;
        }

        if(is_option(arg, "-F", "--format"))
        {
            parse_formats(opts, consume_arg(i, argc, argv, "-F/--format"));
            continue;
        }

        LOG_ERROR("Unknown option '{}'.", arg);
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    return opts;
}

bool
attach_all(const std::vector<pid_t>& pids, active_sessions& sessions)
{
    // read once: the loop below overwrites this variable for every target
    const auto* user_library_env = std::getenv("ROCPROF_ATTACH_TOOL_LIBRARY");
    const auto  user_library =
        user_library_env ? std::optional<std::string>{ user_library_env } : std::nullopt;

    bool all_attached = true;
    for(const auto pid : pids)
    {
        const auto tool_library = resolve_tool_library(pid, user_library);
        if(!tool_library)
        {
            all_attached = false;
            continue;
        }
        LOG_INFO("Using tool library: {}", *tool_library);
        setenv("ROCPROF_ATTACH_TOOL_LIBRARY", tool_library->c_str(), 1);
        setenv("ROCP_TOOL_LIBRARIES", tool_library->c_str(), 1);

        LOG_INFO("Trying to attach to process {}", pid);
        if(rocattach_attach(pid) != ROCATTACH_STATUS_SUCCESS)
        {
            LOG_ERROR("Failed to attach to process {}", pid);
            all_attached = false;
            continue;
        }

        const auto& attached = sessions.add(pid);
        LOG_INFO("Attached to process {} (session {})", pid, attached.id);
    }
    return all_attached;
}

bool
detach(active_sessions& sessions, session target)
{
    const auto status = rocattach_detach(target.pid);
    sessions.remove(target.id);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        LOG_ERROR("Failed to detach from process {} (session {})", target.pid, target.id);
        return false;
    }
    LOG_INFO("Detached from process {} (session {})", target.pid, target.id);
    return true;
}

bool
detach_all(active_sessions& sessions)
{
    bool all_detached = true;
    while(!sessions.active().empty())
    {
        all_detached = (detach(sessions, sessions.active().front()) && all_detached);
    }
    return all_detached;
}

void
print_sessions(const active_sessions& sessions)
{
    std::cout << "\nActive sessions:\n";
    for(const auto& entry : sessions.active())
        std::cout << fmt::format("  [{}] pid {}\n", entry.id, entry.pid);
    std::cout << "Enter a session id to detach, or 'a' to detach all: " << std::flush;
}

// An empty line detaches only when a single session remains, so a stray ENTER never
// detaches one of several targets by accident.
bool
handle_prompt_input(std::string_view line, active_sessions& sessions)
{
    const auto input = rocprofsys::utility::string::trim(line);
    if(input == "a" || (input.empty() && sessions.active().size() == 1))
    {
        return detach_all(sessions);
    }
    if(input.empty())
    {
        return true;
    }

    const auto session_id = to_integral<std::size_t>(input);
    const auto target     = session_id ? sessions.find(*session_id) : std::nullopt;
    if(!target)
    {
        LOG_WARNING("No active session '{}'", input);
        return true;
    }
    return detach(sessions, *target);
}

bool
run_detach_prompt(active_sessions& sessions)
{
    bool        all_detached = true;
    std::string line;
    while(!sessions.active().empty())
    {
        print_sessions(sessions);
        if(!std::getline(std::cin, line)) return detach_all(sessions) && all_detached;

        all_detached = handle_prompt_input(line, sessions) && all_detached;
    }
    return all_detached;
}

void
print_banner()
{
    std::cout << R"(
  ____   ___   ____ __  __   ______   ______ _____ _____ __  __ ____       _  _____ _____  _    ____ _   _
 |  _ \ / _ \ / ___|  \/  | / ___\ \ / / ___|_   _| ____|  \/  / ___|     / \|_   _|_   _|/ \  / ___| | | |
 | |_) | | | | |   | |\/| | \___ \\ V /\___ \ | | |  _| | |\/| \___ \    / _ \ | |   | | / _ \| |   | |_| |
 |  _ <| |_| | |___| |  | |  ___) || |  ___) || | | |___| |  | |___) |  / ___ \| |   | |/ ___ \ |___|  _  |
 |_| \_\\___/ \____|_|  |_| |____/ |_| |____/ |_| |_____|_|  |_|____/  /_/   \_\_|   |_/_/   \_\____|_| |_|

)" << "\n";
}

}  // namespace

int
main(int argc, char* argv[])
{
    print_banner();
    if(argc < 2)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    auto const opts = parse_args(argc, argv);

    if(opts.pids.empty())
    {
        LOG_ERROR("-p <pid> is required.");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    setup_session_env();
    setup_output_env(opts.output_path);
    setup_output_format_env(opts.profile_format);

    active_sessions sessions;
    const bool      all_attached = attach_all(opts.pids, sessions);
    if(sessions.active().empty())
    {
        return EXIT_FAILURE;
    }

    const bool all_detached = run_detach_prompt(sessions);
    return (all_attached && all_detached) ? EXIT_SUCCESS : EXIT_FAILURE;
}
