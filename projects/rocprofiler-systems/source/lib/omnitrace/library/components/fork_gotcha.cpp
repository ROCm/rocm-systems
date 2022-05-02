// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/components/fork_gotcha.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"

namespace omnitrace
{
void
fork_gotcha::configure()
{
    fork_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(fork_gotcha_t, 0, fork);
    };
}

void
fork_gotcha::audit(const gotcha_data_t&, audit::incoming)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT(
        get_debug_env(),
        "Warning! Calling fork() within an OpenMPI application using libfabric "
        "may result is segmentation fault\n");
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(get_debug_env(), 16);
}

void
fork_gotcha::audit(const gotcha_data_t& _data, audit::outgoing, pid_t _pid)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug_env(), "%s() return PID %i\n",
                                      _data.tool_id.c_str(), (int) _pid);
}
}  // namespace omnitrace
