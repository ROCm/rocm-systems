// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// HIP API tracing on Windows is out-of-process: amdhip64 is an ETW provider and
// rocprofiler-sdk is the consumer. The consumer does not have to live inside the application,
// and Windows offers no supported way to put it there, so rocprofv3 runs the tool in this
// process and the application in a child.
//
// This binary exists only to sequence that correctly. rocprofv3 owns all policy; the command
// line here is deliberately minimal.

#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <string>
#include <vector>

namespace
{
// Distinct from any exit code the application can produce through normal means, so rocprofv3
// can tell "the launcher could not start anything" from "the application failed".
constexpr int usage_error_code  = 125;
constexpr int launch_error_code = 126;

void
print_usage()
{
    std::fprintf(stderr,
                 "usage: rocprofv3-launch [--tool-library <path>]... -- <application> [args]\n");
}

void
report_error(const char* what)
{
    std::fprintf(
        stderr, "rocprofv3-launch: %s failed with Windows error %lu\n", what, ::GetLastError());
}

// CreateProcess takes one command line, not an argument vector, and the CRT of the child
// re-splits it. Reproduce the quoting rules that split expects so arguments containing spaces
// or quotes survive the round trip.
std::string
quote_argument(const std::string& arg)
{
    if(!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) return arg;

    auto quoted = std::string{"\""};
    for(auto itr = arg.begin();; ++itr)
    {
        auto backslashes = size_t{0};
        while(itr != arg.end() && *itr == '\\')
        {
            ++itr;
            ++backslashes;
        }

        if(itr == arg.end())
        {
            // Backslashes immediately before the closing quote would escape it.
            quoted.append(backslashes * 2, '\\');
            break;
        }

        if(*itr == '"')
            quoted.append((backslashes * 2) + 1, '\\');
        else
            quoted.append(backslashes, '\\');

        quoted.push_back(*itr);
    }
    quoted.push_back('"');

    return quoted;
}

// rocprofiler_force_configure needs a configure function to drive initialization, but this
// process is only the host: the tool libraries it loads are what create contexts.
rocprofiler_tool_configure_result_t*
host_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name = "rocprofv3-launch";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), nullptr, nullptr, nullptr};

    return &cfg;
}
}  // namespace

int
main(int argc, char** argv)
{
    auto tool_libraries = std::vector<std::string>{};
    auto app_args       = std::vector<std::string>{};

    int idx = 1;
    for(; idx < argc; ++idx)
    {
        auto arg = std::string{argv[idx]};

        if(arg == "--")
        {
            ++idx;
            break;
        }

        if(arg == "--tool-library")
        {
            if(++idx >= argc)
            {
                std::fprintf(stderr, "rocprofv3-launch: --tool-library requires a path\n");
                print_usage();
                return usage_error_code;
            }
            tool_libraries.emplace_back(argv[idx]);
        }
        else
        {
            std::fprintf(stderr, "rocprofv3-launch: unrecognized option '%s'\n", arg.c_str());
            print_usage();
            return usage_error_code;
        }
    }

    for(; idx < argc; ++idx)
        app_args.emplace_back(argv[idx]);

    if(app_args.empty())
    {
        std::fprintf(stderr, "rocprofv3-launch: no application specified\n");
        print_usage();
        return usage_error_code;
    }

    auto command_line = std::string{};
    for(const auto& itr : app_args)
    {
        if(!command_line.empty()) command_line.push_back(' ');
        command_line += quote_argument(itr);
    }

    // CreateProcess may modify the command line in place, so it cannot be a string literal or
    // the buffer of a std::string.
    auto command_line_buffer = std::vector<char>{command_line.begin(), command_line.end()};
    command_line_buffer.push_back('\0');

    auto startup_info = STARTUPINFOA{};
    startup_info.cb   = sizeof(startup_info);

    auto process_info = PROCESS_INFORMATION{};

    // Suspended: the ETW session has to be consuming before the application reaches its first
    // HIP call, and the child's pid -- which the session filters on -- only exists once the
    // process does.
    if(::CreateProcessA(nullptr,
                        command_line_buffer.data(),
                        nullptr,
                        nullptr,
                        TRUE,
                        CREATE_SUSPENDED,
                        nullptr,
                        nullptr,
                        &startup_info,
                        &process_info) == FALSE)
    {
        report_error("CreateProcess");
        return launch_error_code;
    }

    // A job with KILL_ON_JOB_CLOSE ties the child's lifetime to this process: if the launcher
    // dies -- including on the tool-initialization failure path below, which leaves the child
    // suspended -- the application cannot be orphaned and the ETW session cannot be leaked.
    // ETW has 64 session slots system-wide and a leaked one survives until reboot.
    auto job = ::CreateJobObjectA(nullptr, nullptr);
    if(job != nullptr)
    {
        auto limits                             = JOBOBJECT_EXTENDED_LIMIT_INFORMATION{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        if(::SetInformationJobObject(
               job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) == FALSE ||
           ::AssignProcessToJobObject(job, process_info.hProcess) == FALSE)
        {
            report_error("job object setup");
        }
    }
    else
    {
        report_error("CreateJobObject");
    }

    // Set after CreateProcess so the child does not inherit them. Both configure the consumer
    // in this process; an application that happens to link rocprofiler-sdk would otherwise
    // start a second, redundant session of its own.
    if(!tool_libraries.empty())
    {
        auto joined = std::string{};
        for(const auto& itr : tool_libraries)
        {
            if(!joined.empty()) joined.push_back(';');
            joined += itr;
        }

        if(::_putenv_s("ROCP_TOOL_LIBRARIES", joined.c_str()) != 0)
        {
            std::fprintf(stderr, "rocprofv3-launch: could not set ROCP_TOOL_LIBRARIES\n");
            return launch_error_code;
        }
    }

    if(::_putenv_s("ROCPROF_ETW_TARGET_PID", std::to_string(process_info.dwProcessId).c_str()) != 0)
    {
        std::fprintf(stderr, "rocprofv3-launch: could not set ROCPROF_ETW_TARGET_PID\n");
        return launch_error_code;
    }

    // Loads the tool libraries and runs their initialize, which starts a context and opens the
    // ETW consumer session. The session does not report itself open until ProcessTrace has
    // attached, so the consumer is provably live before the child runs.
    if(auto status = rocprofiler_force_configure(&host_configure);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "rocprofv3-launch: rocprofiler_force_configure failed: %s\n",
                     rocprofiler_get_status_string(status));
        return launch_error_code;
    }

    if(::ResumeThread(process_info.hThread) == static_cast<DWORD>(-1))
    {
        report_error("ResumeThread");
        return launch_error_code;
    }

    ::WaitForSingleObject(process_info.hProcess, INFINITE);

    auto exit_code = DWORD{0};
    if(::GetExitCodeProcess(process_info.hProcess, &exit_code) == FALSE)
    {
        report_error("GetExitCodeProcess");
        exit_code = launch_error_code;
    }

    ::CloseHandle(process_info.hThread);
    ::CloseHandle(process_info.hProcess);

    // Stops the ETW session, decodes what it collected and runs every tool's finalizer, which is
    // what produces the output. This cannot be left to the atexit handler registration installs:
    // that handler belongs to rocprofiler-sdk.dll, so it runs at DLL_PROCESS_DETACH, by which
    // point the process has terminated every thread but this one and a finalizer that flushes a
    // buffer would wait forever on a worker that no longer exists.
    if(auto status = rocprofiler_finalize(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "rocprofv3-launch: rocprofiler_finalize failed: %s\n",
                     rocprofiler_get_status_string(status));
        return launch_error_code;
    }

    // The application's exit code passes through unchanged, including exception codes such as
    // STATUS_ACCESS_VIOLATION: unlike the in-process Linux tool, a crashed application cannot
    // take the trace down with it.
    return static_cast<int>(exit_code);
}
