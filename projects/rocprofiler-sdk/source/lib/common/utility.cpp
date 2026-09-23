// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "lib/common/utility.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"

#if !defined(_WIN32)
#    include <unistd.h>
#else
#    include <windows.h>
//
#    include <shellapi.h>
#    include <tlhelp32.h>
#endif
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace
{
#if !defined(_WIN32)
std::string_view
get_clock_name(clockid_t _id)
{
#    define CLOCK_NAME_CASE_STATEMENT(NAME)                                                        \
        case NAME: return #NAME;
    switch(_id)
    {
        CLOCK_NAME_CASE_STATEMENT(CLOCK_REALTIME)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_MONOTONIC)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_PROCESS_CPUTIME_ID)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_THREAD_CPUTIME_ID)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_MONOTONIC_RAW)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_REALTIME_COARSE)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_MONOTONIC_COARSE)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_BOOTTIME)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_REALTIME_ALARM)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_BOOTTIME_ALARM)
        CLOCK_NAME_CASE_STATEMENT(CLOCK_TAI)
        default: break;
    }
    return "CLOCK_UNKNOWN";
#    undef CLOCK_NAME_CASE_STATEMENT
}
#endif

auto _process_init_ns = timestamp_ns();
}  // namespace

uint64_t
get_clock_period_ns_impl(clockid_t _clk_id)
{
#if !defined(_WIN32)
    constexpr auto nanosec = std::nano::den;

    struct timespec ts;
    auto            ret = clock_getres(_clk_id, &ts);

    if(ROCPROFILER_UNLIKELY(ret != 0))
    {
        auto _err = errno;
        ROCP_FATAL << "error getting clock resolution for " << get_clock_name(_clk_id) << ": "
                   << strerror(_err);
    }
    else if(ROCPROFILER_UNLIKELY(ts.tv_sec != 0 ||
                                 ts.tv_nsec >= std::numeric_limits<uint32_t>::max()))
    {
        ROCP_FATAL << "clock_getres(" << get_clock_name(_clk_id)
                   << ") returned very low frequency (<1Hz)";
    }

    return (static_cast<uint64_t>(ts.tv_sec) * nanosec) + static_cast<uint64_t>(ts.tv_nsec);
#else
    (void) _clk_id;
    return 1;  // QPC already returns ns in get_ticks; period = 1 ns
#endif
}

pid_t
get_traced_pid()
{
#if !defined(_WIN32)
    return get_pid();
#else
    // Set by rocprofv3-launch for the child it created. Same variable the ETW consumer filters
    // on, so the two can never disagree about which process a run describes.
    return static_cast<pid_t>(get_env("ROCPROF_ETW_TARGET_PID", static_cast<uint32_t>(get_pid())));
#endif
}

pid_t
get_ppid()
{
#if !defined(_WIN32)
    return ::getppid();
#else
    auto _snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if(_snapshot == INVALID_HANDLE_VALUE) return 0;

    auto _self    = static_cast<DWORD>(get_pid());
    auto _ppid    = pid_t{0};
    auto _entry   = PROCESSENTRY32W{};
    _entry.dwSize = sizeof(_entry);

    if(::Process32FirstW(_snapshot, &_entry) != 0)
    {
        do
        {
            if(_entry.th32ProcessID == _self)
            {
                _ppid = static_cast<pid_t>(_entry.th32ParentProcessID);
                break;
            }
        } while(::Process32NextW(_snapshot, &_entry) != 0);
    }

    ::CloseHandle(_snapshot);
    return _ppid;
#endif
}

uint64_t
get_process_start_time_ns(pid_t _pid)
{
#if !defined(_WIN32)
    if(_pid == getpid()) return _process_init_ns;
#else
    if(_pid == static_cast<pid_t>(::_getpid())) return _process_init_ns;
#endif
    return 0;
}

std::vector<std::string>
read_command_line(pid_t _pid)
{
    auto _cmdline = std::vector<std::string>{};
#if !defined(_WIN32)
    auto fcmdline = std::stringstream{};
    fcmdline << "/proc/" << _pid << "/cmdline";
    auto ifs = std::ifstream{fcmdline.str().c_str()};
    if(ifs)
    {
        char        cstr;
        std::string sarg;
        while(!ifs.eof())
        {
            ifs >> cstr;
            if(!ifs.eof())
            {
                if(cstr != '\0')
                {
                    sarg += cstr;
                }
                else
                {
                    _cmdline.push_back(sarg);
                    sarg = "";
                }
            }
        }
        ifs.close();
    }
#else
    // Only the calling process can be asked for its command line without debug privileges, so
    // anything else reports nothing rather than guessing.
    if(_pid != get_pid()) return _cmdline;

    auto  _argc = int{0};
    auto* _argv = ::CommandLineToArgvW(::GetCommandLineW(), &_argc);
    if(_argv == nullptr) return _cmdline;

    _cmdline.reserve(_argc);
    for(int i = 0; i < _argc; ++i)
    {
        auto _len = ::WideCharToMultiByte(CP_UTF8, 0, _argv[i], -1, nullptr, 0, nullptr, nullptr);
        if(_len <= 1) continue;

        auto _str = std::string(static_cast<size_t>(_len) - 1, '\0');
        ::WideCharToMultiByte(CP_UTF8, 0, _argv[i], -1, _str.data(), _len, nullptr, nullptr);
        _cmdline.emplace_back(std::move(_str));
    }

    ::LocalFree(_argv);
#endif
    return _cmdline;
}
}  // namespace common
}  // namespace rocprofiler

namespace
{
std::atomic<bool>&
debugger_block()
{
    static std::atomic<bool> block = {true};
    return block;
}
}  // namespace

extern "C" {
void
rocprofiler_debugger_block()
{
    while(debugger_block().load() == true)
    {};
    // debugger_block().exchange(true);
}

void
rocprofiler_debugger_continue()
{
    debugger_block().exchange(false);
}
}
