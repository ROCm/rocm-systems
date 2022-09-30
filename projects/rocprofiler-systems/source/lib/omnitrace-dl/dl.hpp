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

#ifndef OMNITRACE_DL_HPP_
#define OMNITRACE_DL_HPP_ 1

#if defined(OMNITRACE_DL_SOURCE) && (OMNITRACE_DL_SOURCE > 0)
#    include "common/defines.h"
#else
#    if !defined(OMNITRACE_PUBLIC_API)
#        define OMNITRACE_PUBLIC_API
#    endif
#endif

#include "omnitrace/user.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(OMNITRACE_USE_OMPT)
#    define OMNITRACE_USE_OMPT 0
#endif

#if !defined(OMNITRACE_USE_ROCTRACER)
#    define OMNITRACE_USE_ROCTRACER 0
#endif

#if !defined(OMNITRACE_USE_ROCPROFILER)
#    define OMNITRACE_USE_ROCPROFILER 0
#endif

//--------------------------------------------------------------------------------------//
//
//      omnitrace symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    void omnitrace_init_library(void) OMNITRACE_PUBLIC_API;
    void omnitrace_init_tooling(void) OMNITRACE_PUBLIC_API;
    void omnitrace_init(const char*, bool, const char*) OMNITRACE_PUBLIC_API;
    void omnitrace_finalize(void) OMNITRACE_PUBLIC_API;
    void omnitrace_set_env(const char* env_name,
                           const char* env_val) OMNITRACE_PUBLIC_API;
    void omnitrace_set_mpi(bool use, bool attached) OMNITRACE_PUBLIC_API;
    void omnitrace_push_trace(const char* name) OMNITRACE_PUBLIC_API;
    void omnitrace_pop_trace(const char* name) OMNITRACE_PUBLIC_API;
    void omnitrace_push_region(const char*) OMNITRACE_PUBLIC_API;
    void omnitrace_pop_region(const char*) OMNITRACE_PUBLIC_API;
    void omnitrace_register_source(const char* file, const char* func, size_t line,
                                   size_t      address,
                                   const char* source) OMNITRACE_PUBLIC_API;
    void omnitrace_register_coverage(const char* file, const char* func,
                                     size_t address) OMNITRACE_PUBLIC_API;

#if defined(OMNITRACE_DL_SOURCE) && (OMNITRACE_DL_SOURCE > 0)
    void omnitrace_preinit_library(void) OMNITRACE_HIDDEN_API;
    int  omnitrace_preload_library(void) OMNITRACE_HIDDEN_API;

    int omnitrace_user_start_trace_dl(void) OMNITRACE_HIDDEN_API;
    int omnitrace_user_stop_trace_dl(void) OMNITRACE_HIDDEN_API;

    int omnitrace_user_start_thread_trace_dl(void) OMNITRACE_HIDDEN_API;
    int omnitrace_user_stop_thread_trace_dl(void) OMNITRACE_HIDDEN_API;

    int omnitrace_user_push_region_dl(const char*) OMNITRACE_HIDDEN_API;
    int omnitrace_user_pop_region_dl(const char*) OMNITRACE_HIDDEN_API;

    // KokkosP
    struct OMNITRACE_HIDDEN_API SpaceHandle
    {
        char name[64];
    };

    struct OMNITRACE_HIDDEN_API Kokkos_Tools_ToolSettings
    {
        bool requires_global_fencing;
        bool padding[255];
    };

    void kokkosp_print_help(char*) OMNITRACE_PUBLIC_API;
    void kokkosp_parse_args(int, char**) OMNITRACE_PUBLIC_API;
    void kokkosp_declare_metadata(const char*, const char*) OMNITRACE_PUBLIC_API;
    void kokkosp_request_tool_settings(const uint32_t,
                                       Kokkos_Tools_ToolSettings*) OMNITRACE_PUBLIC_API;
    void kokkosp_init_library(const int, const uint64_t, const uint32_t,
                              void*) OMNITRACE_PUBLIC_API;
    void kokkosp_finalize_library() OMNITRACE_PUBLIC_API;
    void kokkosp_begin_parallel_for(const char*, uint32_t,
                                    uint64_t*) OMNITRACE_PUBLIC_API;
    void kokkosp_end_parallel_for(uint64_t) OMNITRACE_PUBLIC_API;
    void kokkosp_begin_parallel_reduce(const char*, uint32_t,
                                       uint64_t*) OMNITRACE_PUBLIC_API;
    void kokkosp_end_parallel_reduce(uint64_t) OMNITRACE_PUBLIC_API;
    void kokkosp_begin_parallel_scan(const char*, uint32_t,
                                     uint64_t*) OMNITRACE_PUBLIC_API;
    void kokkosp_end_parallel_scan(uint64_t) OMNITRACE_PUBLIC_API;
    void kokkosp_begin_fence(const char*, uint32_t, uint64_t*) OMNITRACE_PUBLIC_API;
    void kokkosp_end_fence(uint64_t) OMNITRACE_PUBLIC_API;
    void kokkosp_push_profile_region(const char*) OMNITRACE_PUBLIC_API;
    void kokkosp_pop_profile_region() OMNITRACE_PUBLIC_API;
    void kokkosp_create_profile_section(const char*, uint32_t*) OMNITRACE_PUBLIC_API;
    void kokkosp_destroy_profile_section(uint32_t) OMNITRACE_PUBLIC_API;
    void kokkosp_start_profile_section(uint32_t) OMNITRACE_PUBLIC_API;
    void kokkosp_stop_profile_section(uint32_t) OMNITRACE_PUBLIC_API;
    void kokkosp_allocate_data(const SpaceHandle, const char*, const void* const,
                               const uint64_t) OMNITRACE_PUBLIC_API;
    void kokkosp_deallocate_data(const SpaceHandle, const char*, const void* const,
                                 const uint64_t) OMNITRACE_PUBLIC_API;
    void kokkosp_begin_deep_copy(SpaceHandle, const char*, const void*, SpaceHandle,
                                 const char*, const void*, uint64_t) OMNITRACE_PUBLIC_API;
    void kokkosp_end_deep_copy() OMNITRACE_PUBLIC_API;
    void kokkosp_profile_event(const char*) OMNITRACE_PUBLIC_API;
    void kokkosp_dual_view_sync(const char*, const void* const,
                                bool) OMNITRACE_PUBLIC_API;
    void kokkosp_dual_view_modify(const char*, const void* const,
                                  bool) OMNITRACE_PUBLIC_API;

    // OpenMP Tools (OMPT)
#    if OMNITRACE_USE_OMPT > 0
    struct ompt_start_tool_result_t;

    ompt_start_tool_result_t* ompt_start_tool(unsigned int,
                                              const char*) OMNITRACE_PUBLIC_API;
#    endif

#    if OMNITRACE_USE_ROCTRACER > 0
    // HSA
    struct HsaApiTable;
    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names) OMNITRACE_PUBLIC_API;
    void OnUnload() OMNITRACE_PUBLIC_API;
#    endif

#    if OMNITRACE_USE_ROCPROFILER > 0
    // ROCP
    void OnLoadToolProp(void* settings) OMNITRACE_PUBLIC_API;
    void OnUnloadTool() OMNITRACE_PUBLIC_API;
#    endif
#endif
}

#endif  // OMNITRACE_DL_HPP_ 1
