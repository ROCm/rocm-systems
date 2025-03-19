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

#pragma once

#include "core/defines.hpp"
#include "rocprofiler-systems/categories.h"  // in rocprof-sys-user

#include <timemory/compat/macros.h>

#include <cstddef>

// forward decl of the API
extern "C"
{
    /// handles configuration logic
    void rocprofsys_init_library(void) ROCPROFSYS_PUBLIC_API;

    /// handles configuration logic
    void rocprofsys_init_tooling(void) ROCPROFSYS_PUBLIC_API;

    /// starts gotcha wrappers
    void rocprofsys_init(const char*, bool, const char*) ROCPROFSYS_PUBLIC_API;

    /// shuts down all tooling and generates output
    void rocprofsys_finalize(void) ROCPROFSYS_PUBLIC_API;

    /// remove librocprof-sys from LD_PRELOAD
    void rocprofsys_reset_preload(void) ROCPROFSYS_PUBLIC_API;

    /// sets an environment variable
    void rocprofsys_set_env(const char*, const char*) ROCPROFSYS_PUBLIC_API;

    /// sets whether MPI should be used
    void rocprofsys_set_mpi(bool, bool) ROCPROFSYS_PUBLIC_API;

    /// starts an instrumentation region
    void rocprofsys_push_trace(const char*) ROCPROFSYS_PUBLIC_API;

    /// stops an instrumentation region
    void rocprofsys_pop_trace(const char*) ROCPROFSYS_PUBLIC_API;

    /// starts an instrumentation region (user-defined)
    int rocprofsys_push_region(const char*) ROCPROFSYS_PUBLIC_API;

    /// stops an instrumentation region (user-defined)
    int rocprofsys_pop_region(const char*) ROCPROFSYS_PUBLIC_API;

    /// starts an instrumentation region in a user-defined category and (optionally)
    /// adds annotations to the perfetto trace.
    int rocprofsys_push_category_region(rocprofsys_category_t, const char*,
                                        rocprofsys_annotation_t*,
                                        size_t) ROCPROFSYS_PUBLIC_API;

    /// stops an instrumentation region in a user-defined category and (optionally)
    /// adds annotations to the perfetto trace.
    int rocprofsys_pop_category_region(rocprofsys_category_t, const char*,
                                       rocprofsys_annotation_t*,
                                       size_t) ROCPROFSYS_PUBLIC_API;

    /// stores source code information
    void rocprofsys_register_source(const char* file, const char* func, size_t line,
                                    size_t      address,
                                    const char* source) ROCPROFSYS_PUBLIC_API;

    /// increments coverage values
    void rocprofsys_register_coverage(const char* file, const char* func,
                                      size_t address) ROCPROFSYS_PUBLIC_API;

    /// mark causal progress
    void rocprofsys_progress(const char*) ROCPROFSYS_PUBLIC_API;

    /// mark causal progress with annotations
    void rocprofsys_annotated_progress(const char*, rocprofsys_annotation_t*,
                                       size_t) ROCPROFSYS_PUBLIC_API;

    // these are the real implementations for internal calling convention
    void rocprofsys_init_library_hidden(void) ROCPROFSYS_HIDDEN_API;
    bool rocprofsys_init_tooling_hidden(bool postinit = false) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_init_hidden(const char*, bool, const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_finalize_hidden(void) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_reset_preload_hidden(void) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_set_env_hidden(const char*, const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_set_mpi_hidden(bool, bool) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_push_trace_hidden(const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_pop_trace_hidden(const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_push_region_hidden(const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_pop_region_hidden(const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_push_category_region_hidden(rocprofsys_category_t, const char*,
                                                rocprofsys_annotation_t*,
                                                size_t) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_pop_category_region_hidden(rocprofsys_category_t, const char*,
                                               rocprofsys_annotation_t*,
                                               size_t) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_register_source_hidden(const char*, const char*, size_t, size_t,
                                           const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_register_coverage_hidden(const char*, const char*,
                                             size_t) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_progress_hidden(const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_annotated_progress_hidden(const char*, rocprofsys_annotation_t*,
                                              size_t) ROCPROFSYS_HIDDEN_API;
}
