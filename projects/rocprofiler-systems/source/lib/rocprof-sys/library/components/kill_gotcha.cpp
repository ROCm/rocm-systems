// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "kill_gotcha.hpp"
#include "core/config.hpp"

#include "logger/debug.hpp"

#include <cstdlib>
#include <unistd.h>

namespace rocprofsys
{
namespace component
{
void
kill_gotcha::configure()
{
    kill_gotcha_t::get_initializer() = []() {
        kill_gotcha_t::configure<0, int, pid_t, int>("kill");
    };
}

int
kill_gotcha::operator()(const gotcha_data& _data, kill_func_t _func, pid_t _pid,
                        int _sig) const
{
    // When profiling multi-process applications where a parent process sends SIGKILL to
    // child processes, the termination can occur before the profiler has a chance to
    // flush collected data. This introduces a configurable delay before SIGKILL
    // signals are forwarded, allowing profiling data to be captured before process
    // termination.
    // NOTE: This is a workaround.

    auto kill_delay = get_kill_delay();
    if(kill_delay > 0)
    {
        auto _self_pid = getpid();

        if(_sig == SIGKILL && _pid != _self_pid && _pid > 0)
        {
            LOG_DEBUG("[kill_gotcha] Intercepted '{}({}, SIGKILL)' triggered from "
                      "process with id: {}. Sleeping for {} seconds...",
                      _data.tool_id, _pid, _self_pid, kill_delay);

            ::sleep(kill_delay);
        }
    }

    return (*_func)(_pid, _sig);
}
}  // namespace component
}  // namespace rocprofsys
