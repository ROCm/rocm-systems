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

#include "library/api.hpp"
#include "library/debug.hpp"

#include <exception>
#include <stdexcept>

extern "C" void
omnitrace_push_trace(const char* _name)
{
    omnitrace_push_trace_hidden(_name);
}

extern "C" void
omnitrace_pop_trace(const char* _name)
{
    omnitrace_pop_trace_hidden(_name);
}

extern "C" int
omnitrace_push_region(const char* _name)
{
    try
    {
        omnitrace_push_region_hidden(_name);
    } catch(std::exception& _e)
    {
        OMNITRACE_VERBOSE_F(1, "Exception caught: %s\n", _e.what());
        return -1;
    }
    return 0;
}

extern "C" int
omnitrace_pop_region(const char* _name)
{
    try
    {
        omnitrace_pop_region_hidden(_name);
    } catch(std::exception& _e)
    {
        OMNITRACE_VERBOSE_F(1, "Exception caught: %s\n", _e.what());
        return -1;
    }
    return 0;
}

extern "C" void
omnitrace_init_library(void)
{
    omnitrace_init_library_hidden();
}

extern "C" void
omnitrace_init(const char* _mode, bool _rewrite, const char* _arg0)
{
    omnitrace_init_hidden(_mode, _rewrite, _arg0);
}

extern "C" void
omnitrace_finalize(void)
{
    omnitrace_finalize_hidden();
}

extern "C" void
omnitrace_set_env(const char* env_name, const char* env_val)
{
    omnitrace_set_env_hidden(env_name, env_val);
}

extern "C" void
omnitrace_set_mpi(bool use, bool attached)
{
    omnitrace_set_mpi_hidden(use, attached);
}
