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
#include "lib/common/logging.hpp"

#include <unistd.h>
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
std::string_view
get_clock_name(clockid_t _id)
{
#define CLOCK_NAME_CASE_STATEMENT(NAME)                                                            \
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
#undef CLOCK_NAME_CASE_STATEMENT
}

auto _process_init_ns = timestamp_ns();
}  // namespace

uint64_t
get_clock_period_ns_impl(clockid_t _clk_id)
{
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
}

uint64_t
get_process_start_time_ns(pid_t _pid)
{
    if(_pid == getpid()) return _process_init_ns;
    return 0;
}

std::vector<std::string>
read_command_line(pid_t _pid)
{
    auto _cmdline = std::vector<std::string>{};
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
