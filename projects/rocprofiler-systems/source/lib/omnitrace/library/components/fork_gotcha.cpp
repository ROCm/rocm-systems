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
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"

#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>

#include <unistd.h>

namespace omnitrace
{
namespace component
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
    tim::set_env("OMNITRACE_PRELOAD", "0", 1);
    OMNITRACE_VERBOSE(1, "fork() called on PID %i (rank: %i), TID %li\n",
                      process::get_id(), dmp::rank(), threading::get_id());
    OMNITRACE_BASIC_DEBUG(
        "Warning! Calling fork() within an OpenMPI application using libfabric "
        "may result is segmentation fault\n");
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(get_debug_env(), 16);
}

void
fork_gotcha::audit(const gotcha_data_t&, audit::outgoing, pid_t _pid)
{
    if(_pid != 0)
    {
        OMNITRACE_VERBOSE(1, "fork() called on PID %i created PID %i\n", getppid(), _pid);
        settings::use_output_suffix()      = true;
        settings::default_process_suffix() = process::get_id();
    }
}
}  // namespace component
}  // namespace omnitrace
