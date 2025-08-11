// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/ompt/ompt.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/ompt.h>
#include <rocprofiler-sdk/ompt/api_args.h>
#include <rocprofiler-sdk/ompt/omp-tools.h>

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace ompt
{
namespace
{
ompt_table&
get_table();

struct ompt_table_lookup
{
    using type = ompt_table;
    auto& operator()(type& _v) const { return _v; }
    auto& operator()(type* _v) const { return *_v; }
    auto& operator()() const { return (*this)(get_table()); }
};
}  // namespace
}  // namespace ompt
}  // namespace rocprofiler

#define ROCPROFILER_LIB_ROCPROFILER_OMPT_OMPT_CPP_IMPL 1
#include "ompt.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_OMPT_OMPT_CPP_IMPL

namespace rocprofiler
{
namespace ompt
{
namespace
{
auto&
get_ompt_state_stack()
{
    // for callbacks that don't have a place to stash context, we assume
    // a per-thread stack. otherwise we stash the saved state in the ompt_data_t field.
    static thread_local auto _v = tracing::small_vector_t<ompt_save_state*, 8>{};
    return _v;
}

auto*
get_ompt_data_proxy()
{
    static auto*& _v = common::static_object<ompt_data_proxy>::construct();
    return _v;
}

// Macros for access to appropriate ompt_data_t* proxy
#define CLIENT(name)   (CHECK_NOTNULL(get_ompt_data_proxy())->get_client_ptr(name))
#define INTERNAL(name) (CHECK_NOTNULL(get_ompt_data_proxy())->get_internal_ptr(name))

void
ompt_thread_begin_callback(ompt_thread_t thread_type, ompt_data_t* thread_data)
{
    ompt_impl<ROCPROFILER_OMPT_ID_thread_begin>::event(thread_type, CLIENT(thread_data));
}

void
ompt_thread_end_callback(ompt_data_t* thread_data)
{
    ompt_impl<ROCPROFILER_OMPT_ID_thread_end>::event(CLIENT(thread_data));
}

void
ompt_parallel_begin_callback(ompt_data_t*        encountering_task_data,
                             const ompt_frame_t* encountering_task_frame,
                             ompt_data_t*        parallel_data,
                             unsigned int        requested_parallelism,
                             int                 flags,
                             const void*         codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_parallel_begin>::event(CLIENT(encountering_task_data),
                                                         encountering_task_frame,
                                                         CLIENT(parallel_data),
                                                         requested_parallelism,
                                                         flags,
                                                         codeptr_ra);
}

void
ompt_parallel_end_callback(ompt_data_t* parallel_data,
                           ompt_data_t* encountering_task_data,
                           int          flags,
                           const void*  codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_parallel_end>::event(
        CLIENT(parallel_data), CLIENT(encountering_task_data), flags, codeptr_ra);
}

void
ompt_task_create_callback(ompt_data_t*        encountering_task_data,
                          const ompt_frame_t* encountering_task_frame,
                          ompt_data_t*        new_task_data,
                          int                 flags,
                          int                 has_dependences,
                          const void*         codeptr_ra)
{
    auto* corr_id =
        ompt_impl<ROCPROFILER_OMPT_ID_task_create>::event_common(CLIENT(encountering_task_data),
                                                                 encountering_task_frame,
                                                                 CLIENT(new_task_data),
                                                                 flags,
                                                                 has_dependences,
                                                                 codeptr_ra);

    auto* state                  = new ompt_task_save_state{corr_id, flags};
    INTERNAL(new_task_data)->ptr = state;

    context::pop_latest_correlation_id(corr_id);
}

void
ompt_task_schedule_callback(ompt_data_t*       prior_task_data,
                            ompt_task_status_t prior_task_status,
                            ompt_data_t*       next_task_data)
{
    auto* corr_id = ompt_impl<ROCPROFILER_OMPT_ID_task_schedule>::event_common(
        CLIENT(prior_task_data), prior_task_status, CLIENT(next_task_data));
    context::pop_latest_correlation_id(corr_id);
    corr_id->sub_ref_count();

    auto* pprior = INTERNAL(prior_task_data);
    auto* pnext  = INTERNAL(next_task_data);
    assert(pprior != nullptr);
    auto* state_prior  = reinterpret_cast<ompt_task_save_state*>(pprior->ptr);
    auto* state_next   = pnext ? reinterpret_cast<ompt_task_save_state*>(pnext->ptr) : nullptr;
    auto* prior_corrid = context::get_latest_correlation_id();
    if(state_prior->corr_id == prior_corrid && state_prior->task_flags != 0)
    {
        // pop the current correlation ID (for the prior_task)
        assert((state_prior->task_flags & 0xFF) == ompt_task_explicit);
        context::pop_latest_correlation_id(prior_corrid);
    }
    if(state_next && (state_next->task_flags & 0xFF) == ompt_task_explicit)
    {
        // push the next correlation ID (for the next_task)
        context::push_correlation_id(state_next->corr_id);
    }
    if(prior_task_status == ompt_task_yield || prior_task_status == ompt_task_detach ||
       prior_task_status == ompt_task_switch)
        return;
    // the prior task is done
    assert(state_prior->task_flags != 0);
    if(prior_task_status == ompt_task_complete)
    {
        // FIXME? do we need to decrement the ref count
        // state_prior->corr_id->sub_ref_count();
        delete state_prior;
        pprior->ptr = nullptr;
    }
}

void
ompt_implicit_task_callback(ompt_scope_endpoint_t endpoint,
                            ompt_data_t*          parallel_data,
                            ompt_data_t*          task_data,
                            unsigned int          actual_parallelism,
                            unsigned int          index,
                            int                   flags)
{
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_implicit_task>::begin(INTERNAL(task_data),
                                                            endpoint,
                                                            CLIENT(parallel_data),
                                                            CLIENT(task_data),
                                                            actual_parallelism,
                                                            index,
                                                            flags);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_implicit_task>::end(INTERNAL(task_data),
                                                          endpoint,
                                                          CLIENT(parallel_data),
                                                          CLIENT(task_data),
                                                          actual_parallelism,
                                                          index,
                                                          flags);
    }
    else
    {
        ROCP_FATAL << "endpoint in implicit_task is not begin or end: " << endpoint;
    }
}

void
ompt_device_initialize_callback(int                    device_num,
                                const char*            type,
                                ompt_device_t*         device,
                                ompt_function_lookup_t lookup,
                                const char*            documentation)
{
    ompt_impl<ROCPROFILER_OMPT_ID_device_initialize>::event(
        device_num, type, device, lookup, documentation);
}

void
ompt_device_finalize_callback(int device_num)
{
    ompt_impl<ROCPROFILER_OMPT_ID_device_finalize>::event(device_num);
}

void
ompt_device_load_callback(int         device_num,
                          const char* filename,
                          int64_t     offset_in_file,
                          void*       vma_in_file,
                          size_t      bytes,
                          void*       host_addr,
                          void*       device_addr,
                          uint64_t    module_id)
{
    ompt_impl<ROCPROFILER_OMPT_ID_device_load>::event(device_num,
                                                      filename,
                                                      offset_in_file,
                                                      vma_in_file,
                                                      bytes,
                                                      host_addr,
                                                      device_addr,
                                                      module_id);
}

// void
// ompt_device_unload_callback(int device_num, uint64_t module_id)
// {
//     ompt_impl<ROCPROFILER_OMPT_ID_device_unload>::event(device_num, module_id);
// }

void
ompt_sync_region_wait_callback(ompt_sync_region_t    kind,
                               ompt_scope_endpoint_t endpoint,
                               ompt_data_t*          parallel_data,
                               ompt_data_t*          task_data,
                               const void*           codeptr_ra)
{
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_sync_region_wait>::begin(
            nullptr, kind, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_sync_region_wait>::end(
            nullptr, kind, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else
    {
        ROCP_FATAL << "endpoint in sync_region_wait is not begin or end: " << endpoint;
    }
}

void
ompt_mutex_released_callback(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void* codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_mutex_released>::event(kind, wait_id, codeptr_ra);
}

void
ompt_dependences_callback(ompt_data_t* task_data, const ompt_dependence_t* deps, int ndeps)
{
    ompt_impl<ROCPROFILER_OMPT_ID_dependences>::event(CLIENT(task_data), deps, ndeps);
}

void
ompt_task_dependence_callback(ompt_data_t* src_task_data, ompt_data_t* sink_task_data)
{
    ompt_impl<ROCPROFILER_OMPT_ID_task_dependence>::event(CLIENT(src_task_data),
                                                          CLIENT(sink_task_data));
}

void
ompt_work_callback(ompt_work_t           work_type,
                   ompt_scope_endpoint_t endpoint,
                   ompt_data_t*          parallel_data,
                   ompt_data_t*          task_data,
                   uint64_t              count,
                   const void*           codeptr_ra)
{
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_work>::begin(nullptr,
                                                   work_type,
                                                   endpoint,
                                                   CLIENT(parallel_data),
                                                   CLIENT(task_data),
                                                   count,
                                                   codeptr_ra);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_work>::end(nullptr,
                                                 work_type,
                                                 endpoint,
                                                 CLIENT(parallel_data),
                                                 CLIENT(task_data),
                                                 count,
                                                 codeptr_ra);
    }
    else
    {
        ROCP_FATAL << "endpoint in work is not begin or end: " << endpoint;
    }
}

void
ompt_masked_callback(ompt_scope_endpoint_t endpoint,
                     ompt_data_t*          parallel_data,
                     ompt_data_t*          task_data,
                     const void*           codeptr_ra)
{
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_masked>::begin(
            nullptr, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_masked>::end(
            nullptr, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else
    {
        ROCP_FATAL << "endpoint in masked is not begin or end: " << endpoint;
    }
}

void
ompt_target_map_callback(ompt_id_t     target_id,
                         unsigned int  nitems,
                         void**        host_addr,
                         void**        device_addr,
                         size_t*       bytes,
                         unsigned int* mapping_flags,
                         const void*   codeptr_ra)
{
    common::consume_args(
        target_id, nitems, host_addr, device_addr, bytes, mapping_flags, codeptr_ra);
}

void
ompt_sync_region_callback(ompt_sync_region_t    kind,
                          ompt_scope_endpoint_t endpoint,
                          ompt_data_t*          parallel_data,
                          ompt_data_t*          task_data,
                          const void*           codeptr_ra)
{
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_sync_region>::begin(
            nullptr, kind, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_sync_region>::end(
            nullptr, kind, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else
    {
        ROCP_FATAL << "endpoint in sync_region is not begin or end: " << endpoint;
    }
}

void
ompt_lock_init_callback(ompt_mutex_t   kind,
                        unsigned int   hint,
                        unsigned int   impl,
                        ompt_wait_id_t wait_id,
                        const void*    codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_lock_init>::event(kind, hint, impl, wait_id, codeptr_ra);
}

void
ompt_lock_destroy_callback(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void* codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_lock_destroy>::event(kind, wait_id, codeptr_ra);
}

void
ompt_mutex_acquire_callback(ompt_mutex_t   kind,
                            unsigned int   hint,
                            unsigned int   impl,
                            ompt_wait_id_t wait_id,
                            const void*    codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_mutex_acquire>::event(kind, hint, impl, wait_id, codeptr_ra);
}

void
ompt_mutex_acquired_callback(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void* codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_mutex_acquired>::event(kind, wait_id, codeptr_ra);
}

void
ompt_nest_lock_callback(ompt_scope_endpoint_t endpoint,
                        ompt_wait_id_t        wait_id,
                        const void*           codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_nest_lock>::event(endpoint, wait_id, codeptr_ra);
}

void
ompt_flush_callback(ompt_data_t* thread_data, const void* codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_flush>::event(CLIENT(thread_data), codeptr_ra);
}

void
ompt_cancel_callback(ompt_data_t* task_data, int flags, const void* codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_cancel>::event(CLIENT(task_data), flags, codeptr_ra);
}

void
ompt_reduction_callback(ompt_sync_region_t    kind,
                        ompt_scope_endpoint_t endpoint,
                        ompt_data_t*          parallel_data,
                        ompt_data_t*          task_data,
                        const void*           codeptr_ra)
{
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_reduction>::begin(
            nullptr, kind, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_reduction>::end(
            nullptr, kind, endpoint, CLIENT(parallel_data), CLIENT(task_data), codeptr_ra);
    }
    else
    {
        ROCP_FATAL << "endpoint in reduction is not begin or end: " << endpoint;
    }
}

void
ompt_dispatch_callback(ompt_data_t*    parallel_data,
                       ompt_data_t*    task_data,
                       ompt_dispatch_t kind,
                       ompt_data_t     instance)
{
    ompt_impl<ROCPROFILER_OMPT_ID_dispatch>::event(
        CLIENT(parallel_data), CLIENT(task_data), kind, instance);
}

void
ompt_target_emi_callback(ompt_target_t         kind,
                         ompt_scope_endpoint_t endpoint,
                         int                   device_num,
                         ompt_data_t*          task_data,
                         ompt_data_t*          target_task_data,
                         ompt_data_t*          target_data,
                         const void*           codeptr_ra)
{
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_target_emi>::begin(INTERNAL(target_data),
                                                         kind,
                                                         endpoint,
                                                         device_num,
                                                         CLIENT(task_data),
                                                         CLIENT(target_task_data),
                                                         CLIENT(target_data),
                                                         codeptr_ra);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_target_emi>::end(INTERNAL(target_data),
                                                       kind,
                                                       endpoint,
                                                       device_num,
                                                       CLIENT(task_data),
                                                       CLIENT(target_task_data),
                                                       CLIENT(target_data),
                                                       codeptr_ra);
    }
    else
    {
        ROCP_FATAL << "endpoint in target_emi is not begin or end: " << endpoint;
    }
}

void
ompt_target_data_op_emi_callback(ompt_scope_endpoint_t endpoint,
                                 ompt_data_t*          target_task_data,
                                 ompt_data_t*          target_data,
                                 ompt_id_t*            host_op_id,
                                 ompt_target_data_op_t optype,
                                 void*                 src_address,
                                 int                   src_device_num,
                                 void*                 dst_address,
                                 int                   dst_device_num,
                                 size_t                bytes,
                                 const void*           codeptr_ra)
{
    auto* _host_op_data = reinterpret_cast<ompt_data_t*>(host_op_id);
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_target_data_op_emi>::begin(INTERNAL(_host_op_data),
                                                                 endpoint,
                                                                 CLIENT(target_task_data),
                                                                 CLIENT(target_data),
                                                                 CLIENT(_host_op_data),
                                                                 optype,
                                                                 src_address,
                                                                 src_device_num,
                                                                 dst_address,
                                                                 dst_device_num,
                                                                 bytes,
                                                                 codeptr_ra);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_target_data_op_emi>::end(INTERNAL(_host_op_data),
                                                               endpoint,
                                                               CLIENT(target_task_data),
                                                               CLIENT(target_data),
                                                               CLIENT(_host_op_data),
                                                               optype,
                                                               src_address,
                                                               src_device_num,
                                                               dst_address,
                                                               dst_device_num,
                                                               bytes,
                                                               codeptr_ra);
    }
    else
    {
        ROCP_FATAL << "endpoint in target_data_op_emi is not begin or end: " << endpoint;
    }
}

void
ompt_target_submit_emi_callback(ompt_scope_endpoint_t endpoint,
                                ompt_data_t*          target_data,
                                ompt_id_t*            host_op_id,
                                unsigned int          requested_num_teams)
{
    auto* _host_op_data = reinterpret_cast<ompt_data_t*>(host_op_id);
    if(endpoint == ompt_scope_begin)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_target_submit_emi>::begin(INTERNAL(_host_op_data),
                                                                endpoint,
                                                                CLIENT(target_data),
                                                                CLIENT(_host_op_data),
                                                                requested_num_teams);
    }
    else if(endpoint == ompt_scope_end)
    {
        ompt_impl<ROCPROFILER_OMPT_ID_target_submit_emi>::end(INTERNAL(_host_op_data),
                                                              endpoint,
                                                              CLIENT(target_data),
                                                              CLIENT(_host_op_data),
                                                              requested_num_teams);
    }
    else
    {
        ROCP_FATAL << "endpoint in target_submit_emi is not begin or end: " << endpoint;
    }
    (void) target_data;
}

// void
// ompt_target_map_emi_callback(ompt_data_t*  target_data,
//                              unsigned int  nitems,
//                              void**        host_addr,
//                              void**        device_addr,
//                              size_t*       bytes,
//                              unsigned int* mapping_flags,
//                              const void*   codeptr_ra)
// {
//     common::consume_args(
//         target_data, nitems, host_addr, device_addr, bytes, mapping_flags, codeptr_ra);
// }

void
ompt_error_callback(ompt_severity_t severity,
                    const char*     message,
                    size_t          length,
                    const void*     codeptr_ra)
{
    ompt_impl<ROCPROFILER_OMPT_ID_error>::event(severity, message, length, codeptr_ra);
}
#undef CLIENT
#undef INTERNAL

// The ompt callback table
ompt_table ompt_callback_table = {
    ompt_thread_begin_callback,
    ompt_thread_end_callback,
    ompt_parallel_begin_callback,
    ompt_parallel_end_callback,
    ompt_task_create_callback,
    ompt_task_schedule_callback,
    ompt_implicit_task_callback,
    ompt_device_initialize_callback,
    ompt_device_finalize_callback,
    ompt_device_load_callback,
    // ompt_device_unload_callback,
    ompt_sync_region_wait_callback,
    ompt_mutex_released_callback,
    ompt_dependences_callback,
    ompt_task_dependence_callback,
    ompt_work_callback,
    ompt_masked_callback,
    ompt_target_map_callback,
    ompt_sync_region_callback,
    ompt_lock_init_callback,
    ompt_lock_destroy_callback,
    ompt_mutex_acquire_callback,
    ompt_mutex_acquired_callback,
    ompt_nest_lock_callback,
    ompt_flush_callback,
    ompt_cancel_callback,
    ompt_reduction_callback,
    ompt_dispatch_callback,
    ompt_target_emi_callback,
    ompt_target_data_op_emi_callback,
    ompt_target_submit_emi_callback,
    // ompt_target_map_emi_callback,
    ompt_error_callback,
};

ompt_table&
get_table()
{
    return ompt_callback_table;
}

void
rocprof_ompt_cb_interface(rocprofiler_ompt_callback_functions_t& cb_functions)
{
    ompt_impl<ROCPROFILER_OMPT_ID_callback_functions>::event(cb_functions);
}
}  // namespace

ompt_data_t*
proxy_data_ptr(ompt_data_t* realptr)
{
    return (get_ompt_data_proxy())->get_client_ptr(realptr);
}

// special case fake callback to send the ompt cb function pointers
template <>
struct ompt_info<ROCPROFILER_OMPT_ID_callback_functions>
{
    static constexpr auto callback_domain_idx = ompt_domain_info::callback_domain_idx;
    static constexpr auto buffered_domain_idx = ompt_domain_info::buffered_domain_idx;
    static constexpr auto operation_idx       = ROCPROFILER_OMPT_ID_callback_functions;
    static constexpr auto name                = "omp_callback_functions";
    static constexpr bool unsupported         = false;
    static constexpr auto begin               = -1;

    using this_type = ompt_info<ROCPROFILER_OMPT_ID_callback_functions>;
    using base_type = ompt_impl<ROCPROFILER_OMPT_ID_callback_functions>;

    static constexpr auto offset() { return -1; }

    template <typename DataT>
    static auto& get_api_data_args(DataT& _data)
    {
        return _data.callback_functions;
    }
};

// These implement the callbacks for OMPT
template <size_t OpIdx>
template <typename... Args>
void
ompt_impl<OpIdx>::begin(ompt_data_t* data, Args... args)
{
    using info_type = ompt_info<OpIdx>;

    ROCP_TRACE << __FUNCTION__ << " :: " << info_type::name;

    constexpr auto external_corr_id_domain_idx =
        ompt_domain_info::external_correlation_id_domain_idx;

    constexpr auto ref_count         = 2;
    auto           thr_id            = common::get_tid();
    auto           callback_contexts = tracing::callback_context_data_vec_t{};
    auto           buffered_contexts = tracing::buffered_context_data_vec_t{};
    auto           external_corr_ids = tracing::external_correlation_id_map_t{};

    tracing::populate_contexts(info_type::callback_domain_idx,
                               info_type::buffered_domain_idx,
                               info_type::operation_idx,
                               callback_contexts,
                               buffered_contexts,
                               external_corr_ids);

    auto* corr_id          = tracing::correlation_service::construct(ref_count);
    auto  internal_corr_id = corr_id->internal;
    auto  ancestor_corr_id = corr_id->ancestor;

    tracing::populate_external_correlation_ids(external_corr_ids,
                                               thr_id,
                                               external_corr_id_domain_idx,
                                               info_type::operation_idx,
                                               internal_corr_id);

    // invoke the callbacks
    if(!callback_contexts.empty())
    {
        auto tracer_data = common::init_public_api_struct(callback_ompt_data_t{});
        set_data_args(info_type::get_api_data_args(tracer_data.args), std::forward<Args>(args)...);

        tracing::execute_phase_enter_callbacks(callback_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::callback_domain_idx,
                                               info_type::operation_idx,
                                               tracer_data);
    }

    // enter callback may update the external correlation id field
    tracing::update_external_correlation_ids(
        external_corr_ids, thr_id, external_corr_id_domain_idx);

    // stash the state
    ompt_save_state* state = new ompt_save_state{.thr_id            = thr_id,
                                                 .start_timestamp   = 0,
                                                 .operation_idx     = info_type::operation_idx,
                                                 .corr_id           = corr_id,
                                                 .external_corr_ids = external_corr_ids,
                                                 .callback_contexts = callback_contexts,
                                                 .buffered_contexts = buffered_contexts};

    if(data)
        data->ptr = state;
    else
        get_ompt_state_stack().emplace_back(state);

    // decrement the reference count before returning
    corr_id->sub_ref_count();
    state->start_timestamp = common::timestamp_ns();
}

template <size_t OpIdx>
template <typename... Args>
void
ompt_impl<OpIdx>::end(ompt_data_t* data, Args... args)
{
    using info_type = ompt_info<OpIdx>;

    ROCP_TRACE << __FUNCTION__ << " :: " << info_type::name;

    // END PART OF OMPT CALLBACK
    auto end_timestamp = common::timestamp_ns();

    ompt_save_state* state = nullptr;
    if(data != nullptr)
        state = static_cast<ompt_save_state*>(data->ptr);
    else
        state = get_ompt_state_stack().pop_back_val();
    assert(state != nullptr);

    ROCP_FATAL_IF(state->operation_idx != info_type::operation_idx)
        << "Mismatch of OMPT operation: begin=" << state->operation_idx
        << ", end=" << info_type::operation_idx;

    auto& callback_contexts = state->callback_contexts;
    auto& buffered_contexts = state->buffered_contexts;
    auto  external_corr_ids = state->external_corr_ids;

    auto* corr_id          = state->corr_id;
    auto  internal_corr_id = corr_id->internal;
    auto  ancestor_corr_id = corr_id->ancestor;

    ROCP_FATAL_IF(common::get_tid() != state->thr_id)
        << "MIsmatch of OMPT begin/end thread id: "
        << " current=" << common::get_tid() << ", expected= " << state->thr_id;

    if(!callback_contexts.empty())
    {
        auto tracer_data = common::init_public_api_struct(callback_ompt_data_t{});
        set_data_args(info_type::get_api_data_args(tracer_data.args), std::forward<Args>(args)...);

        tracing::execute_phase_exit_callbacks(callback_contexts,
                                              external_corr_ids,
                                              info_type::callback_domain_idx,
                                              info_type::operation_idx,
                                              tracer_data);
    }

    if(!buffered_contexts.empty())
    {
        auto buffer_record = common::init_public_api_struct(buffer_ompt_record_t{});
        if constexpr(OpIdx == ROCPROFILER_OMPT_ID_target_emi ||
                     OpIdx == ROCPROFILER_OMPT_ID_target_data_op_emi ||
                     OpIdx == ROCPROFILER_OMPT_ID_target_submit_emi)
        {
            auto tracer_data = common::init_public_api_struct(callback_ompt_data_t{});
            set_data_args(info_type::get_api_data_args(tracer_data.args),
                          std::forward<Args>(args)...);
            if constexpr(OpIdx == ROCPROFILER_OMPT_ID_target_emi)
            {
                buffer_record.target.kind       = tracer_data.args.target_emi.kind;
                buffer_record.target.device_num = tracer_data.args.target_emi.device_num;
                buffer_record.target.task_id    = tracer_data.args.target_emi.task_data->value;
                buffer_record.target.target_id  = tracer_data.args.target_emi.target_data->value;
                buffer_record.target.codeptr_ra = tracer_data.args.target_emi.codeptr_ra;
            }
            else if constexpr(OpIdx == ROCPROFILER_OMPT_ID_target_data_op_emi)
            {
                buffer_record.target_data_op.host_op_id =
                    tracer_data.args.target_data_op_emi.host_op_id->value;
                buffer_record.target_data_op.optype = tracer_data.args.target_data_op_emi.optype;
                buffer_record.target_data_op.src_device_num =
                    tracer_data.args.target_data_op_emi.src_device_num;
                buffer_record.target_data_op.dst_device_num =
                    tracer_data.args.target_data_op_emi.dst_device_num;
                buffer_record.target_data_op.reserved = 0;
                buffer_record.target_data_op.bytes    = tracer_data.args.target_data_op_emi.bytes;
                buffer_record.target_data_op.codeptr_ra =
                    tracer_data.args.target_data_op_emi.codeptr_ra;
            }
            else if constexpr(OpIdx == ROCPROFILER_OMPT_ID_target_submit_emi)
            {
                buffer_record.target_kernel.device_num = 0;  // FIXME
                buffer_record.target_kernel.requested_num_teams =
                    tracer_data.args.target_submit_emi.requested_num_teams;
                buffer_record.target_kernel.host_op_id =
                    tracer_data.args.target_submit_emi.host_op_id->value;
            }
        }

        buffer_record.start_timestamp = state->start_timestamp;
        buffer_record.end_timestamp   = end_timestamp;
        tracing::execute_buffer_record_emplace(buffered_contexts,
                                               state->thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::buffered_domain_idx,
                                               info_type::operation_idx,
                                               buffer_record);
    }

    // decrement the reference count after usage in the callback/buffers
    state->corr_id->sub_ref_count();
    context::pop_latest_correlation_id(state->corr_id);
    delete state;
    if(data) data->ptr = nullptr;
}

template <size_t OpIdx>
template <typename... Args>
context::correlation_id*
ompt_impl<OpIdx>::event_common(Args... args)
{
    using info_type = ompt_info<OpIdx>;

    ROCP_TRACE << __FUNCTION__ << " :: " << info_type::name;

    constexpr auto external_corr_id_domain_idx =
        ompt_domain_info::external_correlation_id_domain_idx;

    constexpr auto ref_count         = 1;
    auto           thr_id            = common::get_tid();
    auto           callback_contexts = tracing::callback_context_data_vec_t{};
    auto           buffered_contexts = tracing::buffered_context_data_vec_t{};
    auto           external_corr_ids = tracing::external_correlation_id_map_t{};

    tracing::populate_contexts(info_type::callback_domain_idx,
                               info_type::buffered_domain_idx,
                               info_type::operation_idx,
                               callback_contexts,
                               buffered_contexts,
                               external_corr_ids);

    auto     buffer_record    = common::init_public_api_struct(buffer_ompt_record_t{});
    auto     tracer_data      = common::init_public_api_struct(callback_ompt_data_t{});
    auto*    corr_id          = tracing::correlation_service::construct(ref_count);
    uint64_t internal_corr_id = corr_id->internal;
    uint64_t ancestor_corr_id = corr_id->ancestor;

    tracing::populate_external_correlation_ids(external_corr_ids,
                                               thr_id,
                                               external_corr_id_domain_idx,
                                               info_type::operation_idx,
                                               internal_corr_id);

    // invoke the callbacks
    if(!callback_contexts.empty())
    {
        set_data_args(info_type::get_api_data_args(tracer_data.args), std::forward<Args>(args)...);

        tracing::execute_phase_none_callbacks(callback_contexts,
                                              thr_id,
                                              internal_corr_id,
                                              external_corr_ids,
                                              ancestor_corr_id,
                                              info_type::callback_domain_idx,
                                              info_type::operation_idx,
                                              tracer_data);
    }

    tracing::update_external_correlation_ids(
        external_corr_ids, thr_id, external_corr_id_domain_idx);

    if(!buffered_contexts.empty())
    {
        buffer_record.start_timestamp = common::timestamp_ns();
        buffer_record.end_timestamp   = buffer_record.start_timestamp;
        tracing::execute_buffer_record_emplace(buffered_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::buffered_domain_idx,
                                               info_type::operation_idx,
                                               buffer_record);
    }

    return corr_id;
}

template <size_t OpIdx>
template <typename... Args>
void
ompt_impl<OpIdx>::event(Args&&... args)
{
    auto corr_id = ompt_impl<OpIdx>::event_common(std::forward<Args>(args)...);
    context::pop_latest_correlation_id(corr_id);
    corr_id->sub_ref_count();
}

namespace
{
template <typename Tp>
decltype(auto)
convert_arg(Tp&& _arg)
{
    using type = common::mpl::unqualified_type_t<Tp>;
    if constexpr(common::mpl::is_string_type<type>::value)
    {
        if(!_arg) return std::remove_reference_t<Tp>(_arg);
        return common::get_string_entry(std::string_view{_arg})->c_str();
    }
    else
        return std::forward<Tp>(_arg);
}

template <size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto _idx = ompt_info<OpIdx>::operation_idx;
    if(_idx < ompt_domain_info::last) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0) get_ids(_id_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return ompt_info<OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

bool
should_enable_callback(rocprofiler_callback_tracing_kind_t _callback_domain,
                       rocprofiler_buffer_tracing_kind_t   _buffered_domain,
                       int                                 _operation)
{
    // we loop over all the *registered* contexts and see if any of them, at any point in time,
    // might require callback or buffered API tracing
    for(const auto& itr : context::get_registered_contexts())
    {
        if(!itr) continue;

        // if there is a callback tracer enabled for the given domain and op, we need to wrap
        if(itr->callback_tracer && itr->callback_tracer->domains(_callback_domain) &&
           itr->callback_tracer->domains(_callback_domain, _operation))
            return true;

        // if there is a buffered tracer enabled for the given domain and op, we need to wrap
        if(itr->buffered_tracer && itr->buffered_tracer->domains(_buffered_domain) &&
           itr->buffered_tracer->domains(_buffered_domain, _operation))
            return true;
    }
    return false;
}

template <size_t OpIdx>
void
update_table(ompt_update_func f, std::integral_constant<size_t, OpIdx>)
{
    auto _info = ompt_info<OpIdx>();

    if(_info.unsupported)
    {
        ROCP_INFO << "OMPT operation not supported: " << _info.name;
        return;
    }
    // check to see if there are any contexts which enable this operation in the OMPT API domain
    if(!should_enable_callback(
           _info.callback_domain_idx, _info.buffered_domain_idx, _info.operation_idx))
        return;

    ROCP_TRACE << "updating table entry for " << _info.name;

    // Register this callback for OMPT at init time.
    auto& _func    = _info.get_table_func();
    auto* _ompt_cb = reinterpret_cast<ompt_callback_t*>(&_func);
    f(_info.name, _ompt_cb, _info.ompt_idx);
}

template <size_t OpIdx, size_t... OpIdxTail>
void
update_table(ompt_update_func f, std::index_sequence<OpIdx, OpIdxTail...>)
{
    update_table(f, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0) update_table(f, std::index_sequence<OpIdxTail...>{});
}
}  // namespace

template <size_t OpIdx>
template <typename DataArgsT, typename... Args>
void
ompt_impl<OpIdx>::set_data_args(DataArgsT& _data_args, Args... args)
{
    if constexpr(sizeof...(Args) == 0)
        _data_args.no_args.empty = '\0';
    else
        _data_args = DataArgsT{convert_arg(args)...};
}

// check out the assembly here... this compiles to a switch statement
const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ompt_domain_info::last>{});
}

std::vector<uint32_t>
get_ids()
{
    constexpr auto last_api_id = ompt_domain_info::last;
    auto           _data       = std::vector<uint32_t>{};
    _data.reserve(last_api_id);
    get_ids(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <typename DataT, size_t OpIdx, size_t... OpIdxTail>
void
iterate_args(const uint32_t                                   id,
             const DataT&                                     data,
             rocprofiler_callback_tracing_operation_args_cb_t func,
             int32_t                                          max_deref,
             void*                                            user_data,
             std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id)
    {
        using info_type = ompt_info<OpIdx>;
        auto&& arg_list = info_type::as_arg_list(data, max_deref);
        auto&& arg_addr = info_type::as_arg_addr(data);
        for(size_t i = 0; i < std::min(arg_list.size(), arg_addr.size()); ++i)
        {
            auto ret = func(info_type::callback_domain_idx,    // kind
                            id,                                // operation
                            i,                                 // arg_number
                            arg_addr.at(i),                    // arg_value_addr
                            arg_list.at(i).indirection_level,  // indirection
                            arg_list.at(i).type,               // arg_type
                            arg_list.at(i).name,               // arg_name
                            arg_list.at(i).value.c_str(),      // arg_value_str
                            arg_list.at(i).dereference_count,  // num deref in str
                            user_data);
            if(ret != 0) break;
        }
        return;
    }
    if constexpr(sizeof...(OpIdxTail) > 0)
        iterate_args(id, data, func, max_deref, user_data, std::index_sequence<OpIdxTail...>{});
}

void
update_callback(rocprofiler_ompt_callback_functions_t& cb_functions)
{
    auto _info = ompt_info<ROCPROFILER_OMPT_ID_callback_functions>();

    if(should_enable_callback(
           _info.callback_domain_idx, _info.buffered_domain_idx, _info.operation_idx))
        rocprof_ompt_cb_interface(cb_functions);
}

void
iterate_args(uint32_t                                         id,
             const rocprofiler_callback_tracing_ompt_data_t&  data,
             rocprofiler_callback_tracing_operation_args_cb_t callback,
             int32_t                                          max_deref,
             void*                                            user_data)
{
    if(callback)
        iterate_args(id,
                     data,
                     callback,
                     max_deref,
                     user_data,
                     std::make_index_sequence<ompt::ompt_domain_info::ompt_last>{});
}

void
update_table(ompt_update_func f)
{
    update_table(f, std::make_index_sequence<ompt::ompt_domain_info::ompt_last>{});
}

}  // namespace ompt
}  // namespace rocprofiler
