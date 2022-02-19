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

#pragma once

#include "library/defines.hpp"

#include <timemory/compat/macros.h>

// forward decl of the API
extern "C"
{
    /// handles configuration logic
    void omnitrace_init_library(void) TIMEMORY_VISIBILITY("default");

    /// starts gotcha wrappers
    void omnitrace_init(const char*, bool, const char*) TIMEMORY_VISIBILITY("default");

    /// shuts down all tooling and generates output
    void omnitrace_finalize(void) TIMEMORY_VISIBILITY("default");

    /// sets an environment variable
    void omnitrace_set_env(const char* env_name, const char* env_val)
        TIMEMORY_VISIBILITY("default");

    /// sets whether MPI should be used
    void omnitrace_set_mpi(bool use, bool attached) TIMEMORY_VISIBILITY("default");

    /// starts an instrumentation region
    void omnitrace_push_trace(const char* name) TIMEMORY_VISIBILITY("default");

    /// stops an instrumentation region
    void omnitrace_pop_trace(const char* name) TIMEMORY_VISIBILITY("default");

    /// used by omnitrace-critical-trace
    bool omnitrace_init_tooling() TIMEMORY_VISIBILITY("hidden");
}
