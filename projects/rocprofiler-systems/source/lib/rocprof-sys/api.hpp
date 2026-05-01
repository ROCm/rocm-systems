// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
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
    bool rocprofsys_init_tooling_hidden(void) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_init_hidden(const char*, bool, const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_finalize_hidden(void) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_set_finalization_done_hidden(void) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_reset_for_reattach_hidden(void) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_reset_preload_hidden(void) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_set_env_hidden(const char*, const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_set_mpi_hidden(bool, bool) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_push_trace_hidden(const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_pop_trace_hidden(const char*) ROCPROFSYS_HIDDEN_API;
    void rocprofsys_flush_pending_region_cache_hidden() ROCPROFSYS_HIDDEN_API;
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

    /// registers external pause/resume callbacks (e.g. from the Python profiler).
    void rocprofsys_external_register_pause_callbacks(void (*)(),
                                                      void (*)()) ROCPROFSYS_PUBLIC_API;
}
