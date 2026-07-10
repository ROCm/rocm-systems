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

// Models a rocprofiler-sdk client tool library: it exports rocprofiler_configure, so it is
// discovered by rocprofiler-sdk's link-map symbol walk during OMPT tool bring-up. The library
// that actually calls into OpenMP from its DT_INIT constructor is a separate consumer library
// (see openmp_consumer.cpp); this file only plays the role of the profiler tool.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <cstdio>
#include <cstdlib>

namespace
{
// Detects whether this library's static initialization (DT_INIT_ARRAY) ran *after*
// rocprofiler_configure/tool_init were invoked. If discovery bypassed DT_INIT_ARRAY to call
// rocprofiler_configure directly, the later real initialization would clobber this global back
// to 0 -- so tool_fini seeing anything other than 1 means the tool's static state was corrupted.
int var_check = 0;

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    var_check = 1;
    return 0;
}

void
tool_fini(void*)
{
    if(var_check != 1)
    {
        fprintf(stderr,
                "[client] ERROR: var_check == %d (expected 1); static initialization ran after "
                "tool_init -- DT_INIT_ARRAY bypass corrupted tool state\n",
                var_check);
        fflush(stderr);
        std::abort();
    }
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
