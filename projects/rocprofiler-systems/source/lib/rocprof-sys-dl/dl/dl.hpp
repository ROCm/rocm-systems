// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCPROFSYS_DL_HPP_
#define ROCPROFSYS_DL_HPP_

#if defined(ROCPROFSYS_DL_SOURCE) && (ROCPROFSYS_DL_SOURCE > 0)
#    include "common/defines.h"
#else
#    if !defined(ROCPROFSYS_PUBLIC_API)
#        define ROCPROFSYS_PUBLIC_API
#    endif
#endif

#include "rocprofiler-systems/user.h"

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

#if !defined(ROCPROFSYS_USE_OMPT)
#    define ROCPROFSYS_USE_OMPT 0
#endif

//--------------------------------------------------------------------------------------//
//
//      rocprof-sys symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    void rocprofsys_init_library(void) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_init_tooling(void) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_init(const char*, bool, const char*) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_finalize(void) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_set_env(const char* env_name,
                            const char* env_val) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_set_mpi(bool use) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_set_instrumented(int) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_push_trace(const char* name) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_push_trace_with_args(const char* name,
                                         const char* args) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_pop_trace(const char* name) ROCPROFSYS_PUBLIC_API;
    int  rocprofsys_push_region(const char*) ROCPROFSYS_PUBLIC_API;
    int  rocprofsys_pop_region(const char*) ROCPROFSYS_PUBLIC_API;
    int  rocprofsys_push_category_region(rocprofsys_category_t, const char*,
                                         rocprofsys_annotation_t*,
                                         size_t) ROCPROFSYS_PUBLIC_API;
    int  rocprofsys_pop_category_region(rocprofsys_category_t, const char*,
                                        rocprofsys_annotation_t*,
                                        size_t) ROCPROFSYS_PUBLIC_API;

    void rocprofsys_register_source(const char* file, const char* func, size_t line,
                                    size_t      address,
                                    const char* source) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_register_coverage(const char* file, const char* func,
                                      size_t address) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_progress(const char*) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_annotated_progress(const char*, rocprofsys_annotation_t*,
                                       size_t) ROCPROFSYS_PUBLIC_API;
    void rocprofsys_external_register_pause_callbacks(void (*)(),
                                                      void (*)()) ROCPROFSYS_PUBLIC_API;

#if defined(ROCPROFSYS_DL_SOURCE) && (ROCPROFSYS_DL_SOURCE > 0)
    void rocprofsys_preinit_library(void) ROCPROFSYS_HIDDEN_API;
    int  rocprofsys_preload_library(void) ROCPROFSYS_HIDDEN_API;

    int rocprofsys_user_start_trace_dl(void) ROCPROFSYS_HIDDEN_API;
    int rocprofsys_user_stop_trace_dl(void) ROCPROFSYS_HIDDEN_API;

    int rocprofsys_user_start_thread_trace_dl(void) ROCPROFSYS_HIDDEN_API;
    int rocprofsys_user_stop_thread_trace_dl(void) ROCPROFSYS_HIDDEN_API;

    int rocprofsys_user_push_region_dl(const char*) ROCPROFSYS_HIDDEN_API;
    int rocprofsys_user_pop_region_dl(const char*) ROCPROFSYS_HIDDEN_API;

    int rocprofsys_user_push_annotated_region_dl(const char*, rocprofsys_annotation_t*,
                                                 size_t) ROCPROFSYS_HIDDEN_API;
    int rocprofsys_user_pop_annotated_region_dl(const char*, rocprofsys_annotation_t*,
                                                size_t) ROCPROFSYS_HIDDEN_API;

    int rocprofsys_user_progress_dl(const char* name) ROCPROFSYS_HIDDEN_API;
    int rocprofsys_user_annotated_progress_dl(const char*, rocprofsys_annotation_t*,
                                              size_t) ROCPROFSYS_HIDDEN_API;
    // KokkosP
    struct ROCPROFSYS_HIDDEN_API SpaceHandle
    {
        char name[64];
    };

    struct ROCPROFSYS_HIDDEN_API Kokkos_Tools_ToolSettings
    {
        bool requires_global_fencing;
        bool padding[255];
    };

    void kokkosp_print_help(char*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_parse_args(int, char**) ROCPROFSYS_PUBLIC_API;
    void kokkosp_declare_metadata(const char*, const char*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_request_tool_settings(const std::uint32_t,
                                       Kokkos_Tools_ToolSettings*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_init_library(const int, const std::uint64_t, const std::uint32_t,
                              void*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_finalize_library() ROCPROFSYS_PUBLIC_API;
    void kokkosp_begin_parallel_for(const char*, std::uint32_t,
                                    std::uint64_t*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_end_parallel_for(std::uint64_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_begin_parallel_reduce(const char*, std::uint32_t,
                                       std::uint64_t*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_end_parallel_reduce(std::uint64_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_begin_parallel_scan(const char*, std::uint32_t,
                                     std::uint64_t*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_end_parallel_scan(std::uint64_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_begin_fence(const char*, std::uint32_t,
                             std::uint64_t*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_end_fence(std::uint64_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_push_profile_region(const char*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_pop_profile_region() ROCPROFSYS_PUBLIC_API;
    void kokkosp_create_profile_section(const char*,
                                        std::uint32_t*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_destroy_profile_section(std::uint32_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_start_profile_section(std::uint32_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_stop_profile_section(std::uint32_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_allocate_data(const SpaceHandle, const char*, const void* const,
                               const std::uint64_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_deallocate_data(const SpaceHandle, const char*, const void* const,
                                 const std::uint64_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_begin_deep_copy(SpaceHandle, const char*, const void*, SpaceHandle,
                                 const char*, const void*,
                                 std::uint64_t) ROCPROFSYS_PUBLIC_API;
    void kokkosp_end_deep_copy() ROCPROFSYS_PUBLIC_API;
    void kokkosp_profile_event(const char*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_dual_view_sync(const char*, const void* const,
                                bool) ROCPROFSYS_PUBLIC_API;
    void kokkosp_dual_view_modify(const char*, const void* const,
                                  bool) ROCPROFSYS_PUBLIC_API;

    // OpenMP Tools (OMPT)
#    if ROCPROFSYS_USE_OMPT > 0
    struct ompt_start_tool_result_t;

    ompt_start_tool_result_t* ompt_start_tool(unsigned int,
                                              const char*) ROCPROFSYS_PUBLIC_API;
#    endif

    struct rocprofiler_tool_configure_result_t;
    struct rocprofiler_client_id_t;

#endif  // ROCPROFSYS_DL_SOURCE
}

namespace rocprofsys
{
namespace dl
{
enum class InstrumentMode : int
{
    None          = -1,
    BinaryRewrite = 0,
    ProcessCreate = 1,  // runtime instrumentation at start of process
    PythonProfile = 2,  // python setprofile
    Last,
};
}
}  // namespace rocprofsys

#endif  // ROCPROFSYS_DL_HPP_ 1
