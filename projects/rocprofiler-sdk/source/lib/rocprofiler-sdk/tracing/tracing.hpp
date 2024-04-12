// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "lib/common/mpl.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <functional>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tracing
{
// template <typename DomainT, typename... Args>
// bool
// context_filter(const context::context* ctx, DomainT domain, Args... args);

// template <typename ClearContainersT = std::false_type>
// void
// populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
//                   rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
//                   rocprofiler_tracing_operation_t     operation_idx,
//                   callback_context_data_vec_t&        callback_contexts,
//                   buffered_context_data_vec_t&        buffered_contexts,
//                   external_correlation_id_map_t&      extern_corr_ids,
//                   ClearContainersT = ClearContainersT{});

// template <typename ClearContainersT = std::false_type>
// void
// populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
//                   rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
//                   callback_context_data_vec_t&        callback_contexts,
//                   buffered_context_data_vec_t&        buffered_contexts,
//                   external_correlation_id_map_t&      extern_corr_ids,
//                   ClearContainersT = ClearContainersT{});

// template <typename ClearContainersT = std::false_type>
// void
// populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
//                   rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
//                   rocprofiler_tracing_operation_t     operation_idx,
//                   tracing_data&                       data,
//                   ClearContainersT = ClearContainersT{});

// template <typename ClearContainersT = std::false_type>
// void
// populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
//                   rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
//                   tracing_data&                       data,
//                   ClearContainersT = ClearContainersT{});

// void
// populate_external_correlation_ids(external_correlation_id_map_t& external_corr_ids,
//                                   rocprofiler_thread_id_t        thr_id,
//                                   rocprofiler_external_correlation_id_request_kind_t kind,
//                                   rocprofiler_tracing_operation_t                    operation,
//                                   uint64_t internal_corr_id);

// void
// update_external_correlation_ids(external_correlation_id_map_t& external_corr_ids,
//                                 rocprofiler_thread_id_t        thr_id,
//                                 rocprofiler_external_correlation_id_request_kind_t kind);

// template <typename TracerDataT>
// void
// execute_phase_none_callbacks(callback_context_data_vec_t&        callback_contexts,
//                              rocprofiler_thread_id_t             thr_id,
//                              uint64_t                            internal_corr_id,
//                              external_correlation_id_map_t&      external_corr_ids,
//                              rocprofiler_callback_tracing_kind_t domain,
//                              rocprofiler_tracing_operation_t     operation,
//                              TracerDataT&                        tracer_data);

// template <typename TracerDataT>
// void
// execute_phase_enter_callbacks(callback_context_data_vec_t&        callback_contexts,
//                               rocprofiler_thread_id_t             thr_id,
//                               uint64_t                            internal_corr_id,
//                               external_correlation_id_map_t&      external_corr_ids,
//                               rocprofiler_callback_tracing_kind_t domain,
//                               rocprofiler_tracing_operation_t     operation,
//                               TracerDataT&                        tracer_data);

// template <typename TracerDataT>
// void
// execute_phase_exit_callbacks(callback_context_data_vec_t&        callback_contexts,
//                              external_correlation_id_map_t&      external_corr_ids,
//                              rocprofiler_callback_tracing_kind_t domain,
//                              rocprofiler_tracing_operation_t     operation,
//                              TracerDataT&                        tracer_data);

// template <typename BufferRecordT, typename OperationT = rocprofiler_tracing_operation_t>
// void
// execute_buffer_record_emplace(buffered_context_data_vec_t&      buffered_contexts,
//                               rocprofiler_thread_id_t           thr_id,
//                               uint64_t                          internal_corr_id,
//                               external_correlation_id_map_t&    external_corr_ids,
//                               rocprofiler_buffer_tracing_kind_t domain,
//                               OperationT                        operation,
//                               BufferRecordT&&                   base_record);

template <typename DomainT, typename... Args>
inline bool
context_filter(const context::context* ctx, DomainT domain, Args... args)
{
    if constexpr(std::is_same<DomainT, rocprofiler_buffer_tracing_kind_t>::value)
    {
        return (ctx->buffered_tracer && ctx->buffered_tracer->domains(domain, args...));
    }
    else if constexpr(std::is_same<DomainT, rocprofiler_callback_tracing_kind_t>::value)
    {
        return (ctx->callback_tracer && ctx->callback_tracer->domains(domain, args...));
    }
    else
    {
        static_assert(common::mpl::assert_false<DomainT>::value, "unsupported domain type");
        return false;
    }
}

template <typename ClearContainersT = std::false_type>
inline void
populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
                  rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
                  rocprofiler_tracing_operation_t     operation_idx,
                  callback_context_data_vec_t&        callback_contexts,
                  buffered_context_data_vec_t&        buffered_contexts,
                  external_correlation_id_map_t&      extern_corr_ids,
                  ClearContainersT = ClearContainersT{})
{
    if constexpr(ClearContainersT::value)
    {
        callback_contexts.clear();
        buffered_contexts.clear();
        extern_corr_ids.clear();
    }

    const auto minimal_context_filter = [](const context_t* ctx) {
        return (ctx->callback_tracer || ctx->buffered_tracer);
    };

    for(const auto* itr : context::get_active_contexts(minimal_context_filter))
    {
        if(!itr) continue;

        // if the given domain + op is not enabled, skip this context
        if(context_filter(itr, callback_domain_idx, operation_idx))
        {
            callback_contexts.emplace_back(
                callback_context_data{itr, rocprofiler_callback_tracing_record_t{}});
            extern_corr_ids.emplace(itr, empty_user_data);
        }

        // if the given domain + op is not enabled, skip this context
        if(context_filter(itr, buffered_domain_idx, operation_idx))
        {
            buffered_contexts.emplace_back(buffered_context_data{itr});
            extern_corr_ids.emplace(itr, empty_user_data);
        }
    }
}

template <typename ClearContainersT = std::false_type>
inline void
populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
                  rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
                  callback_context_data_vec_t&        callback_contexts,
                  buffered_context_data_vec_t&        buffered_contexts,
                  external_correlation_id_map_t&      extern_corr_ids,
                  ClearContainersT = ClearContainersT{})
{
    if constexpr(ClearContainersT::value)
    {
        callback_contexts.clear();
        buffered_contexts.clear();
        extern_corr_ids.clear();
    }

    const auto minimal_context_filter = [](const context_t* ctx) {
        return (ctx->callback_tracer || ctx->buffered_tracer);
    };

    for(const auto* itr : context::get_active_contexts(minimal_context_filter))
    {
        if(!itr) continue;

        // if the given domain + op is not enabled, skip this context
        if(context_filter(itr, callback_domain_idx))
        {
            callback_contexts.emplace_back(
                callback_context_data{itr, rocprofiler_callback_tracing_record_t{}});
            extern_corr_ids.emplace(itr, empty_user_data);
        }

        // if the given domain + op is not enabled, skip this context
        if(context_filter(itr, buffered_domain_idx))
        {
            buffered_contexts.emplace_back(buffered_context_data{itr});
            extern_corr_ids.emplace(itr, empty_user_data);
        }
    }
}

template <typename ClearContainersT = std::false_type>
inline void
populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
                  rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
                  rocprofiler_tracing_operation_t     operation_idx,
                  tracing_data&                       data,
                  ClearContainersT = ClearContainersT{})
{
    populate_contexts<ClearContainersT>(callback_domain_idx,
                                        buffered_domain_idx,
                                        operation_idx,
                                        data.callback_contexts,
                                        data.buffered_contexts,
                                        data.external_correlation_ids);
}

template <typename ClearContainersT = std::false_type>
inline void
populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
                  rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
                  tracing_data&                       data,
                  ClearContainersT = ClearContainersT{})
{
    populate_contexts<ClearContainersT>(callback_domain_idx,
                                        buffered_domain_idx,
                                        data.callback_contexts,
                                        data.buffered_contexts,
                                        data.external_correlation_ids);
}

inline void
populate_external_correlation_ids(external_correlation_id_map_t& external_corr_ids,
                                  rocprofiler_thread_id_t        thr_id,
                                  rocprofiler_external_correlation_id_request_kind_t kind,
                                  rocprofiler_tracing_operation_t                    operation,
                                  uint64_t internal_corr_id)
{
    for(auto& itr : external_corr_ids)
    {
        itr.second = itr.first->correlation_tracer.external_correlator.get(
            thr_id, itr.first, kind, operation, internal_corr_id);
    }
}

inline void
update_external_correlation_ids(external_correlation_id_map_t& external_corr_ids,
                                rocprofiler_thread_id_t        thr_id,
                                rocprofiler_external_correlation_id_request_kind_t kind)
{
    // enter callback may update the external correlation id field
    for(auto& itr : external_corr_ids)
    {
        itr.second =
            itr.first->correlation_tracer.external_correlator.update(itr.second, thr_id, kind);
    }
}

template <typename TracerDataT>
inline void
execute_phase_none_callbacks(callback_context_data_vec_t&         callback_contexts,
                             rocprofiler_thread_id_t              thr_id,
                             uint64_t                             internal_corr_id,
                             const external_correlation_id_map_t& external_corr_ids,
                             rocprofiler_callback_tracing_kind_t  domain,
                             rocprofiler_tracing_operation_t      operation,
                             TracerDataT&                         tracer_data)
{
    for(auto& itr : callback_contexts)
    {
        if(!context_filter(itr.ctx, domain, operation)) continue;

        auto&       ctx              = itr.ctx;
        auto&       record           = itr.record;
        auto&       user_data        = itr.user_data;
        const auto& extern_corr_id_v = external_corr_ids.at(ctx);

        auto corr_id_v = rocprofiler_correlation_id_t{internal_corr_id, extern_corr_id_v};
        record = rocprofiler_callback_tracing_record_t{rocprofiler_context_id_t{ctx->context_idx},
                                                       thr_id,
                                                       corr_id_v,
                                                       domain,
                                                       operation,
                                                       ROCPROFILER_CALLBACK_PHASE_NONE,
                                                       static_cast<void*>(&tracer_data)};

        auto& callback_info = ctx->callback_tracer->callback_data.at(domain);
        callback_info.callback(record, &user_data, callback_info.data);
    }
}

template <typename TracerDataT>
inline void
execute_phase_enter_callbacks(callback_context_data_vec_t&         callback_contexts,
                              rocprofiler_thread_id_t              thr_id,
                              uint64_t                             internal_corr_id,
                              const external_correlation_id_map_t& external_corr_ids,
                              rocprofiler_callback_tracing_kind_t  domain,
                              rocprofiler_tracing_operation_t      operation,
                              TracerDataT&                         tracer_data)
{
    for(auto& itr : callback_contexts)
    {
        if(!context_filter(itr.ctx, domain, operation)) continue;

        auto&       ctx              = itr.ctx;
        auto&       record           = itr.record;
        auto&       user_data        = itr.user_data;
        const auto& extern_corr_id_v = external_corr_ids.at(ctx);

        auto corr_id_v = rocprofiler_correlation_id_t{internal_corr_id, extern_corr_id_v};
        record = rocprofiler_callback_tracing_record_t{rocprofiler_context_id_t{ctx->context_idx},
                                                       thr_id,
                                                       corr_id_v,
                                                       domain,
                                                       operation,
                                                       ROCPROFILER_CALLBACK_PHASE_ENTER,
                                                       static_cast<void*>(&tracer_data)};

        auto& callback_info = ctx->callback_tracer->callback_data.at(domain);
        callback_info.callback(record, &user_data, callback_info.data);
    }
}

template <typename TracerDataT>
inline void
execute_phase_exit_callbacks(callback_context_data_vec_t&         callback_contexts,
                             const external_correlation_id_map_t& external_corr_ids,
                             rocprofiler_callback_tracing_kind_t  domain,
                             rocprofiler_tracing_operation_t      operation,
                             TracerDataT&                         tracer_data)
{
    for(auto& itr : callback_contexts)
    {
        if(!context_filter(itr.ctx, domain, operation)) continue;

        auto&       ctx              = itr.ctx;
        auto&       record           = itr.record;
        auto&       user_data        = itr.user_data;
        const auto& extern_corr_id_v = external_corr_ids.at(ctx);

        auto corr_id_v =
            rocprofiler_correlation_id_t{record.correlation_id.internal, extern_corr_id_v};
        record = rocprofiler_callback_tracing_record_t{rocprofiler_context_id_t{ctx->context_idx},
                                                       record.thread_id,
                                                       corr_id_v,
                                                       domain,
                                                       record.operation,
                                                       ROCPROFILER_CALLBACK_PHASE_EXIT,
                                                       static_cast<void*>(&tracer_data)};

        auto& callback_info = ctx->callback_tracer->callback_data.at(domain);
        callback_info.callback(record, &user_data, callback_info.data);
    }
}

template <typename BufferRecordT, typename OperationT = rocprofiler_tracing_operation_t>
inline void
execute_buffer_record_emplace(const buffered_context_data_vec_t&   buffered_contexts,
                              rocprofiler_thread_id_t              thr_id,
                              uint64_t                             internal_corr_id,
                              const external_correlation_id_map_t& external_corr_ids,
                              rocprofiler_buffer_tracing_kind_t    domain,
                              OperationT                           operation,
                              BufferRecordT&&                      base_record)
{
    base_record.thread_id = thr_id;
    base_record.kind      = domain;
    base_record.operation = operation;
    // external correlation will be updated right before record is placed in buffer
    base_record.correlation_id = rocprofiler_correlation_id_t{internal_corr_id, empty_user_data};

    for(const auto& itr : buffered_contexts)
    {
        if(!context_filter(itr.ctx, domain, operation)) continue;

        auto  buffer_id = itr.ctx->buffered_tracer->buffer_data.at(domain);
        auto* buffer_v  = buffer::get_buffer(buffer_id);
        if(buffer_v && buffer_v->context_id == itr.ctx->context_idx &&
           buffer_v->buffer_id == buffer_id.handle)
        {
            // make copy of record
            auto record_v = base_record;
            // update the record with the correlation
            record_v.correlation_id.external = external_corr_ids.at(itr.ctx);

            buffer_v->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING, domain, record_v);
        }
    }
}
}  // namespace tracing
}  // namespace rocprofiler
