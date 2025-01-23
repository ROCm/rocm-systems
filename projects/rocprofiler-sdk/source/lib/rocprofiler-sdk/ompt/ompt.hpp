// mit license
//
// copyright (c) 2023-2025 Advanced Micro Devices, Inc. all rights reserved.
//
// permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "software"), to deal
// in the software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the software, and to permit persons to whom the software is
// furnished to do so, subject to the following conditions:
//
// the above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the software.
//
// the software is provided "as is", without warranty of any kind, express or
// implied, including but not limited to the warranties of merchantability,
// fitness for a particular purpose and noninfringement.  in no event shall the
// authors or copyright holders be liable for any claim, damages or other
// liability, whether in an action of contract, tort or otherwise, arising from,
// out of or in connection with the software or the use or other dealings in
// the software.

#pragma once

#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/ompt.h>
#include <rocprofiler-sdk/ompt/api_id.h>
#include <rocprofiler-sdk/ompt/omp-tools.h>

#include <cstdint>
#include <deque>
#include <vector>

namespace rocprofiler
{
namespace ompt
{
struct ompt_table
{
    ompt_callback_thread_begin_t      ompt_thread_begin_fn      = nullptr;
    ompt_callback_thread_end_t        ompt_thread_end_fn        = nullptr;
    ompt_callback_parallel_begin_t    ompt_parallel_begin_fn    = nullptr;
    ompt_callback_parallel_end_t      ompt_parallel_end_fn      = nullptr;
    ompt_callback_task_create_t       ompt_task_create_fn       = nullptr;
    ompt_callback_task_schedule_t     ompt_task_schedule_fn     = nullptr;
    ompt_callback_implicit_task_t     ompt_implicit_task_fn     = nullptr;
    ompt_callback_device_initialize_t ompt_device_initialize_fn = nullptr;
    ompt_callback_device_finalize_t   ompt_device_finalize_fn   = nullptr;
    ompt_callback_device_load_t       ompt_device_load_fn       = nullptr;
    // ompt_callback_device_unload_t   ompt_device_unload_fn      = nullptr;
    ompt_callback_sync_region_t        ompt_sync_region_wait_fn   = nullptr;
    ompt_callback_mutex_t              ompt_mutex_released_fn     = nullptr;
    ompt_callback_dependences_t        ompt_dependences_fn        = nullptr;
    ompt_callback_task_dependence_t    ompt_task_dependence_fn    = nullptr;
    ompt_callback_work_t               ompt_work_fn               = nullptr;
    ompt_callback_masked_t             ompt_masked_fn             = nullptr;
    ompt_callback_target_map_t         ompt_target_map_fn         = nullptr;
    ompt_callback_sync_region_t        ompt_sync_region_fn        = nullptr;
    ompt_callback_mutex_acquire_t      ompt_lock_init_fn          = nullptr;
    ompt_callback_mutex_t              ompt_lock_destroy_fn       = nullptr;
    ompt_callback_mutex_acquire_t      ompt_mutex_acquire_fn      = nullptr;
    ompt_callback_mutex_t              ompt_mutex_acquired_fn     = nullptr;
    ompt_callback_nest_lock_t          ompt_nest_lock_fn          = nullptr;
    ompt_callback_flush_t              ompt_flush_fn              = nullptr;
    ompt_callback_cancel_t             ompt_cancel_fn             = nullptr;
    ompt_callback_sync_region_t        ompt_reduction_fn          = nullptr;
    ompt_callback_dispatch_t           ompt_dispatch_fn           = nullptr;
    ompt_callback_target_emi_t         ompt_target_emi_fn         = nullptr;
    ompt_callback_target_data_op_emi_t ompt_target_data_op_emi_fn = nullptr;
    ompt_callback_target_submit_emi_t  ompt_target_submit_emi_fn  = nullptr;
    // ompt_callback_target_map_emi_t  ompt_target_map_emi_fn     = nullptr;
    ompt_callback_error_t ompt_error_fn = nullptr;
};

struct ompt_domain_info
{
    using args_type                           = rocprofiler_ompt_args_t;
    using retval_type                         = void;
    using callback_data_type                  = rocprofiler_callback_tracing_ompt_data_t;
    using buffer_data_type                    = rocprofiler_buffer_tracing_ompt_record_t;
    using enum_type                           = rocprofiler_ompt_operation_t;
    static constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_OMPT;
    static constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_OMPT;
    static constexpr auto none                = ROCPROFILER_OMPT_ID_NONE;
    static constexpr auto last                = ROCPROFILER_OMPT_ID_LAST;
    static constexpr auto ompt_last           = ROCPROFILER_OMPT_ID_callback_functions;
    static constexpr auto external_correlation_id_domain_idx =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_OMPT;
};

using buffer_ompt_record_t = rocprofiler_buffer_tracing_ompt_record_t;
using callback_ompt_data_t = rocprofiler_callback_tracing_ompt_data_t;

// save state for ompt between callbacks
struct ompt_save_state
{
    uint64_t                               thr_id;           // thread this was created on
    uint64_t                               start_timestamp;  // timestamp when it was created
    rocprofiler_ompt_operation_t           operation_idx;    // for error checking
    context::correlation_id*               corr_id;          // correlation id
    tracing::external_correlation_id_map_t external_corr_ids;
    tracing::callback_context_data_vec_t   callback_contexts;
    tracing::buffered_context_data_vec_t   buffered_contexts;
};

// proxy for ompt_data_t pointers received from OMPT callbacks
// SDK gets index 0, client get index 1
class ompt_data_proxy
{
public:
    ompt_data_t* get_client_ptr(ompt_data_t* ompt_ptr) { return get<1>(ompt_ptr); }
    ompt_data_t* get_internal_ptr(ompt_data_t* ompt_ptr) { return get<0>(ompt_ptr); }

private:
    struct proxy_ptrs
    {
        std::array<ompt_data_t, 2> v;
    };

    // get the proxy pointer for idx. If ompt_ptr->ptr is 0,
    // allocate a new proxy struct and assign it to ompt_ptr->ptr
    // return the address of the requested proxy element,
    // or null if ompt_ptr is null
    template <int idx>
    ompt_data_t* get(ompt_data_t* ompt_ptr)
    {
        static constexpr proxy_ptrs nulval{};
        if(ompt_ptr == nullptr) return nullptr;
        if(ompt_ptr->ptr == nullptr)
        {
            std::lock_guard<std::mutex> lk(m);
            ompt_ptr->ptr = &proxies.emplace_back(nulval);
        }
        auto* ptr = static_cast<proxy_ptrs*>(ompt_ptr->ptr);
        return &(ptr->v[idx]);
    }

    std::deque<proxy_ptrs> proxies;
    std::mutex             m;
};

// function to return client pointer for use outside
ompt_data_t*
proxy_data_ptr(ompt_data_t* real_ptr);

struct ompt_task_save_state
{
    context::correlation_id* corr_id;
    int                      task_flags;
};

template <size_t opidx>
struct ompt_impl
{
    template <typename DataArgst, typename... Args>
    static void set_data_args(DataArgst&, Args... args);

    template <typename... Args>
    static void begin(ompt_data_t* data, Args... args);

    template <typename... Args>
    static void end(ompt_data_t* data, Args... args);

    template <typename... Args>
    static context::correlation_id* event_common(Args... args);

    template <typename... Args>
    static void event(Args&&... args);
};

template <size_t OpIdx>
struct ompt_info;

const char*
name_by_id(uint32_t id);

std::vector<uint32_t>
get_ids();

void
iterate_args(uint32_t                                         id,
             const rocprofiler_callback_tracing_ompt_data_t&  data,
             rocprofiler_callback_tracing_operation_args_cb_t callback,
             int32_t                                          max_deref,
             void*                                            user_data);

using ompt_update_func = void (*)(const char* cbname, ompt_callback_t* cbf, int cbnum);
void update_table(ompt_update_func);

void
update_callback(rocprofiler_ompt_callback_functions_t& cb_functions);

}  // namespace ompt
}  // namespace rocprofiler
