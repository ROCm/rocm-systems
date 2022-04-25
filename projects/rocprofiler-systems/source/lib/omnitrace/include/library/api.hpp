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

#include <cstddef>

// forward decl of the API
extern "C"
{
    /// handles configuration logic
    void omnitrace_init_library(void) OMNITRACE_PUBLIC_API;

    /// starts gotcha wrappers
    void omnitrace_init(const char*, bool, const char*) OMNITRACE_PUBLIC_API;

    /// shuts down all tooling and generates output
    void omnitrace_finalize(void) OMNITRACE_PUBLIC_API;

    /// sets an environment variable
    void omnitrace_set_env(const char* env_name,
                           const char* env_val) OMNITRACE_PUBLIC_API;

    /// sets whether MPI should be used
    void omnitrace_set_mpi(bool use, bool attached) OMNITRACE_PUBLIC_API;

    /// starts an instrumentation region
    void omnitrace_push_trace(const char* name) OMNITRACE_PUBLIC_API;

    /// stops an instrumentation region
    void omnitrace_pop_trace(const char* name) OMNITRACE_PUBLIC_API;

    /// starts an instrumentation region (user-defined)
    int omnitrace_push_region(const char* name) OMNITRACE_PUBLIC_API;

    /// stops an instrumentation region (user-defined)
    int omnitrace_pop_region(const char* name) OMNITRACE_PUBLIC_API;

    /// stores source code information
    void omnitrace_register_source(const char* file, const char* func, size_t line,
                                   size_t      address,
                                   const char* source) OMNITRACE_PUBLIC_API;

    /// increments coverage values
    void omnitrace_register_coverage(const char* file, const char* func,
                                     size_t address) OMNITRACE_PUBLIC_API;

    // these are the real implementations for internal calling convention
    void omnitrace_init_library_hidden(void) OMNITRACE_HIDDEN_API;
    void omnitrace_init_hidden(const char*, bool, const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_finalize_hidden(void) OMNITRACE_HIDDEN_API;
    void omnitrace_set_env_hidden(const char* env_name,
                                  const char* env_val) OMNITRACE_HIDDEN_API;
    void omnitrace_set_mpi_hidden(bool use, bool attached) OMNITRACE_HIDDEN_API;
    void omnitrace_push_trace_hidden(const char* name) OMNITRACE_HIDDEN_API;
    void omnitrace_pop_trace_hidden(const char* name) OMNITRACE_HIDDEN_API;
    void omnitrace_push_region_hidden(const char* name) OMNITRACE_HIDDEN_API;
    void omnitrace_pop_region_hidden(const char* name) OMNITRACE_HIDDEN_API;
    void omnitrace_register_source_hidden(const char* file, const char* func, size_t line,
                                          size_t      address,
                                          const char* source) OMNITRACE_HIDDEN_API;
    void omnitrace_register_coverage_hidden(const char* file, const char* func,
                                            size_t address) OMNITRACE_HIDDEN_API;
}
