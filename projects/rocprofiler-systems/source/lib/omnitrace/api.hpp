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
#include "omnitrace/categories.h"  // in omnitrace-user

#include <timemory/compat/macros.h>

#include <cstddef>

// forward decl of the API
extern "C"
{
    /// handles configuration logic
    void omnitrace_init_library(void) OMNITRACE_PUBLIC_API;

    /// handles configuration logic
    void omnitrace_init_tooling(void) OMNITRACE_PUBLIC_API;

    /// starts gotcha wrappers
    void omnitrace_init(const char*, bool, const char*) OMNITRACE_PUBLIC_API;

    /// shuts down all tooling and generates output
    void omnitrace_finalize(void) OMNITRACE_PUBLIC_API;

    /// remove libomnitrace from LD_PRELOAD
    void omnitrace_reset_preload(void) OMNITRACE_PUBLIC_API;

    /// sets an environment variable
    void omnitrace_set_env(const char*, const char*) OMNITRACE_PUBLIC_API;

    /// sets whether MPI should be used
    void omnitrace_set_mpi(bool, bool) OMNITRACE_PUBLIC_API;

    /// starts an instrumentation region
    void omnitrace_push_trace(const char*) OMNITRACE_PUBLIC_API;

    /// stops an instrumentation region
    void omnitrace_pop_trace(const char*) OMNITRACE_PUBLIC_API;

    /// starts an instrumentation region (user-defined)
    int omnitrace_push_region(const char*) OMNITRACE_PUBLIC_API;

    /// stops an instrumentation region (user-defined)
    int omnitrace_pop_region(const char*) OMNITRACE_PUBLIC_API;

    /// starts an instrumentation region in a user-defined category and (optionally)
    /// adds annotations to the perfetto trace.
    int omnitrace_push_category_region(omnitrace_category_t, const char*,
                                       omnitrace_annotation_t*,
                                       size_t) OMNITRACE_PUBLIC_API;

    /// stops an instrumentation region in a user-defined category and (optionally)
    /// adds annotations to the perfetto trace.
    int omnitrace_pop_category_region(omnitrace_category_t, const char*,
                                      omnitrace_annotation_t*,
                                      size_t) OMNITRACE_PUBLIC_API;

    /// stores source code information
    void omnitrace_register_source(const char* file, const char* func, size_t line,
                                   size_t      address,
                                   const char* source) OMNITRACE_PUBLIC_API;

    /// increments coverage values
    void omnitrace_register_coverage(const char* file, const char* func,
                                     size_t address) OMNITRACE_PUBLIC_API;

    // these are the real implementations for internal calling convention
    void omnitrace_init_library_hidden(void) OMNITRACE_HIDDEN_API;
    bool omnitrace_init_tooling_hidden(void) OMNITRACE_HIDDEN_API;
    void omnitrace_init_hidden(const char*, bool, const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_finalize_hidden(void) OMNITRACE_HIDDEN_API;
    void omnitrace_reset_preload_hidden(void) OMNITRACE_HIDDEN_API;
    void omnitrace_set_env_hidden(const char*, const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_set_mpi_hidden(bool, bool) OMNITRACE_HIDDEN_API;
    void omnitrace_push_trace_hidden(const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_pop_trace_hidden(const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_push_region_hidden(const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_pop_region_hidden(const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_push_category_region_hidden(omnitrace_category_t, const char*,
                                               omnitrace_annotation_t*,
                                               size_t) OMNITRACE_HIDDEN_API;
    void omnitrace_pop_category_region_hidden(omnitrace_category_t, const char*,
                                              omnitrace_annotation_t*,
                                              size_t) OMNITRACE_HIDDEN_API;
    void omnitrace_register_source_hidden(const char*, const char*, size_t, size_t,
                                          const char*) OMNITRACE_HIDDEN_API;
    void omnitrace_register_coverage_hidden(const char*, const char*,
                                            size_t) OMNITRACE_HIDDEN_API;
}
