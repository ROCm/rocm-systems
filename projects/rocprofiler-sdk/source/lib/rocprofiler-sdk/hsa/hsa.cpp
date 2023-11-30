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

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/details/ostream.hpp"
#include "lib/rocprofiler-sdk/hsa/types.hpp"
#include "lib/rocprofiler-sdk/hsa/utils.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <glog/logging.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace hsa
{
namespace
{
std::atomic<activity_functor_t> report_activity = {};

struct null_type
{};

template <typename DataT, typename Tp>
void
set_data_retval(DataT& _data, Tp _val)
{
    if constexpr(std::is_same<Tp, null_type>::value)
    {
        (void) _data;
        (void) _val;
    }
    else if constexpr(std::is_same<Tp, hsa_signal_value_t>::value)
    {
        _data.hsa_signal_value_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, uint64_t>::value)
    {
        _data.uint64_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, uint32_t>::value)
    {
        _data.uint32_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, hsa_status_t>::value)
    {
        _data.hsa_status_t_retval = _val;
    }
    else
    {
        static_assert(std::is_void<Tp>::value, "Error! unsupported return type");
    }
}
}  // namespace

hsa_api_table_t&
get_table()
{
    static auto _core     = CoreApiTable{};
    static auto _amd_ext  = AmdExtTable{};
    static auto _img_ext  = ImageExtTable{};
    static auto _fini_ext = FinalizerExtTable{};
    static auto _v        = []() {
        _core.version = {
            HSA_CORE_API_TABLE_MAJOR_VERSION, sizeof(_core), HSA_CORE_API_TABLE_STEP_VERSION, 0};
        _amd_ext.version  = {HSA_AMD_EXT_API_TABLE_MAJOR_VERSION,
                            sizeof(_amd_ext),
                            HSA_AMD_EXT_API_TABLE_STEP_VERSION,
                            0};
        _img_ext.version  = {HSA_IMAGE_API_TABLE_MAJOR_VERSION,
                            sizeof(_img_ext),
                            HSA_IMAGE_API_TABLE_STEP_VERSION,
                            0};
        _fini_ext.version = {HSA_FINALIZER_API_TABLE_MAJOR_VERSION,
                             sizeof(_fini_ext),
                             HSA_FINALIZER_API_TABLE_STEP_VERSION,
                             0};
        auto _version     = ApiTableVersion{
            HSA_API_TABLE_MAJOR_VERSION, sizeof(HsaApiTable), HSA_API_TABLE_STEP_VERSION, 0};
        auto _val = hsa_api_table_t{_version, &_core, &_amd_ext, &_fini_ext, &_img_ext};
        return _val;
    }();
    return _v;
}

template <size_t Idx>
template <typename DataArgsT, typename... Args>
auto
hsa_api_impl<Idx>::set_data_args(DataArgsT& _data_args, Args... args)
{
    if constexpr(Idx == ROCPROFILER_HSA_API_ID_hsa_amd_memory_async_copy_rect)
    {
        auto _tuple                  = std::make_tuple(args...);
        _data_args.dst               = std::get<0>(_tuple);
        _data_args.dst_offset        = std::get<1>(_tuple);
        _data_args.src               = std::get<2>(_tuple);
        _data_args.src_offset        = std::get<3>(_tuple);
        _data_args.range             = std::get<4>(_tuple);
        _data_args.range__val        = *(std::get<4>(_tuple));
        _data_args.copy_agent        = std::get<5>(_tuple);
        _data_args.dir               = std::get<6>(_tuple);
        _data_args.num_dep_signals   = std::get<7>(_tuple);
        _data_args.dep_signals       = std::get<8>(_tuple);
        _data_args.completion_signal = std::get<9>(_tuple);
    }
    else
    {
        _data_args = DataArgsT{args...};
    }
}

template <size_t Idx>
template <typename FuncT, typename... Args>
auto
hsa_api_impl<Idx>::exec(FuncT&& _func, Args&&... args)
{
    using return_type = std::decay_t<std::invoke_result_t<FuncT, Args...>>;

    if(_func)
    {
        static_assert(std::is_void<return_type>::value || std::is_enum<return_type>::value ||
                          std::is_integral<return_type>::value,
                      "Error! unsupported return type");

        if constexpr(std::is_void<return_type>::value)
        {
            _func(std::forward<Args>(args)...);
            return null_type{};
        }
        else
        {
            return _func(std::forward<Args>(args)...);
        }
    }

    if constexpr(std::is_void<return_type>::value)
        return null_type{};
    else
        return return_type{HSA_STATUS_ERROR};
}

template <size_t Idx>
template <typename... Args>
auto
hsa_api_impl<Idx>::functor(Args&&... args)
{
    using info_type = hsa_api_info<Idx>;

    struct callback_context_data
    {
        const context::context*               ctx       = nullptr;
        rocprofiler_callback_tracing_record_t record    = {};
        rocprofiler_user_data_t               user_data = {.value = 0};
    };

    struct buffered_context_data
    {
        const context::context* ctx                  = nullptr;
        rocprofiler_user_data_t external_correlation = {};
    };

    static thread_local auto active_contexts   = context::context_array_t{};
    auto                     thr_id            = common::get_tid();
    auto                     callback_contexts = std::vector<callback_context_data>{};
    auto                     buffered_contexts = std::vector<buffered_context_data>{};
    auto                     has_pc_sampling   = false;
    for(const auto* itr : context::get_active_contexts(active_contexts))
    {
        if(!itr) continue;

        // if(itr->pc_sampler) has_pc_sampling = true;

        if(itr->callback_tracer)
        {
            // if the given domain + op is not enabled, skip this context
            if(itr->callback_tracer->domains(info_type::callback_domain_idx,
                                             info_type::operation_idx))
                callback_contexts.emplace_back(
                    callback_context_data{itr, rocprofiler_callback_tracing_record_t{}});
        }

        if(itr->buffered_tracer)
        {
            // if the given domain + op is not enabled, skip this context
            if(itr->buffered_tracer->domains(info_type::buffered_domain_idx,
                                             info_type::operation_idx))
                buffered_contexts.emplace_back(buffered_context_data{
                    itr, itr->correlation_tracer.external_correlator.get(thr_id)});
        }
    }

    if(callback_contexts.empty() && buffered_contexts.empty())
    {
        auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);
        if constexpr(!std::is_same<decltype(_ret), null_type>::value)
            return _ret;
        else
            return HSA_STATUS_SUCCESS;
    }

    using correlation_service     = context::correlation_tracing_service;
    using buffer_hsa_api_record_t = rocprofiler_buffer_tracing_hsa_api_record_t;
    using callback_hsa_api_data_t = rocprofiler_callback_tracing_hsa_api_data_t;

    constexpr auto empty_user_data  = rocprofiler_user_data_t{.value = 0};
    auto           ref_count        = (has_pc_sampling) ? 4 : 2;
    auto           buffer_record    = common::init_public_api_struct(buffer_hsa_api_record_t{});
    auto           tracer_data      = common::init_public_api_struct(callback_hsa_api_data_t{});
    auto*          corr_id          = correlation_service::construct(ref_count);
    auto           internal_corr_id = corr_id->internal;

    // construct the buffered info before the callback so the callbacks are as closely wrapped
    // around the function call as possible
    if(!buffered_contexts.empty())
    {
        buffer_record.kind = info_type::buffered_domain_idx;
        // external correlation will be updated right before record is placed in buffer
        buffer_record.correlation_id =
            rocprofiler_correlation_id_t{internal_corr_id, empty_user_data};
        buffer_record.operation = info_type::operation_idx;
        buffer_record.thread_id = thr_id;
    }

    tracer_data.size = sizeof(rocprofiler_callback_tracing_hsa_api_data_t);
    set_data_args(info_type::get_api_data_args(tracer_data.args), std::forward<Args>(args)...);

    // invoke the callbacks
    if(!callback_contexts.empty())
    {
        set_data_args(info_type::get_api_data_args(tracer_data.args), std::forward<Args>(args)...);

        for(auto& itr : callback_contexts)
        {
            auto& ctx       = itr.ctx;
            auto& record    = itr.record;
            auto& user_data = itr.user_data;

            auto extern_corr_id_v = ctx->correlation_tracer.external_correlator.get(thr_id);

            auto corr_id_v = rocprofiler_correlation_id_t{internal_corr_id, extern_corr_id_v};
            record =
                rocprofiler_callback_tracing_record_t{rocprofiler_context_id_t{ctx->context_idx},
                                                      thr_id,
                                                      corr_id_v,
                                                      info_type::callback_domain_idx,
                                                      info_type::operation_idx,
                                                      ROCPROFILER_CALLBACK_PHASE_ENTER,
                                                      static_cast<void*>(&tracer_data)};

            auto& callback_info =
                ctx->callback_tracer->callback_data.at(info_type::callback_domain_idx);
            callback_info.callback(record, &user_data, callback_info.data);

            // enter callback may update the external correlation id field
            record.correlation_id.external =
                ctx->correlation_tracer.external_correlator.get(thr_id);
        }
    }

    // record the start timestamp as close to the function call as possible
    if(!buffered_contexts.empty())
    {
        for(auto& itr : buffered_contexts)
        {
            itr.external_correlation = itr.ctx->correlation_tracer.external_correlator.get(thr_id);
        }

        buffer_record.start_timestamp = common::timestamp_ns();
    }

    // decrement the reference count before invoking
    corr_id->ref_count.fetch_sub(1);

    auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);

    // record the end timestamp as close to the function call as possible
    if(!buffered_contexts.empty())
    {
        buffer_record.end_timestamp = common::timestamp_ns();
    }

    if(!callback_contexts.empty())
    {
        set_data_retval(tracer_data.retval, _ret);

        for(auto& itr : callback_contexts)
        {
            auto& ctx       = itr.ctx;
            auto& record    = itr.record;
            auto& user_data = itr.user_data;

            record.phase   = ROCPROFILER_CALLBACK_PHASE_EXIT;
            record.payload = static_cast<void*>(&tracer_data);

            auto& callback_info =
                ctx->callback_tracer->callback_data.at(info_type::callback_domain_idx);
            callback_info.callback(record, &user_data, callback_info.data);
        }
    }

    if(!buffered_contexts.empty())
    {
        for(auto& itr : buffered_contexts)
        {
            assert(itr.ctx->buffered_tracer);
            auto buffer_id =
                itr.ctx->buffered_tracer->buffer_data.at(info_type::buffered_domain_idx);
            auto buffer_v = buffer::get_buffer(buffer_id);
            if(buffer_v && buffer_v->context_id == itr.ctx->context_idx &&
               buffer_v->buffer_id == buffer_id.handle)
            {
                // make copy of record
                auto record_v = buffer_record;
                // update the record with the correlation
                record_v.correlation_id.external = itr.external_correlation;

                buffer_v->emplace(
                    ROCPROFILER_BUFFER_CATEGORY_TRACING, info_type::buffered_domain_idx, record_v);
            }
        }
    }

    // decrement the reference count after usage in the callback/buffers
    corr_id->ref_count.fetch_sub(1);

    context::pop_latest_correlation_id(corr_id);

    if constexpr(!std::is_same<decltype(_ret), null_type>::value)
        return _ret;
    else
        return HSA_STATUS_SUCCESS;
}
}  // namespace hsa
}  // namespace rocprofiler

#define ROCPROFILER_LIB_ROCPROFILER_HSA_HSA_CPP_IMPL 1

// template specializations
#include "hsa.def.cpp"

namespace rocprofiler
{
namespace hsa
{
namespace
{
template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return hsa_api_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Idx, size_t... IdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<Idx, IdxTail...>)
{
    if(std::string_view{hsa_api_info<Idx>::name} == std::string_view{name})
        return hsa_api_info<Idx>::operation_idx;
    if constexpr(sizeof...(IdxTail) > 0)
        return id_by_name(name, std::index_sequence<IdxTail...>{});
    else
        return ROCPROFILER_HSA_API_ID_NONE;
}

template <size_t Idx, size_t... IdxTail>
void
iterate_args(const uint32_t                                     id,
             const rocprofiler_callback_tracing_hsa_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t   func,
             void*                                              user_data,
             std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id)
    {
        using info_type = hsa_api_info<Idx>;
        auto&& arg_list = info_type::as_arg_list(data);
        auto&& arg_addr = info_type::as_arg_addr(data);
        for(size_t i = 0; i < std::min(arg_list.size(), arg_addr.size()); ++i)
        {
            auto ret = func(info_type::callback_domain_idx,  // kind
                            id,                              // operation
                            i,                               // arg_number
                            arg_list.at(i).first.c_str(),    // arg_name
                            arg_list.at(i).second.c_str(),   // arg_value_str
                            arg_addr.at(i),                  // arg_value_addr
                            user_data);
            if(ret != 0) break;
        }
    }
    if constexpr(sizeof...(IdxTail) > 0)
        iterate_args(id, data, func, user_data, std::index_sequence<IdxTail...>{});
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < ROCPROFILER_HSA_API_ID_LAST) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, hsa_api_info<Idx>::operation_idx), ...);
}

template <size_t... Idx>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, const char* _v) {
        if(_v != nullptr && strnlen(_v, 1) > 0) _vec.emplace_back(_v);
    };

    (_emplace(_name_list, hsa_api_info<Idx>::name), ...);
}

bool
should_wrap_functor(rocprofiler_callback_tracing_kind_t _callback_domain,
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

template <size_t... Idx>
void
update_table(hsa_api_table_t* _orig, std::index_sequence<Idx...>)
{
    auto _update = [](hsa_api_table_t* _orig_v, auto _info) {
        // check to see if there are any contexts which enable this operation in the HSA API domain
        if(!should_wrap_functor(
               _info.callback_domain_idx, _info.buffered_domain_idx, _info.operation_idx))
            return;

        // 1. get the sub-table containing the function pointer
        // 2. get reference to function pointer in sub-table
        // 3. update function pointer with functor
        auto& _table = _info.get_table(_orig_v);
        auto& _func  = _info.get_table_func(_table);
        _func        = _info.get_functor(_func);
    };

    (_update(_orig, hsa_api_info<Idx>{}), ...);
}
}  // namespace

// check out the assembly here... this compiles to a switch statement
const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

uint32_t
id_by_name(const char* name)
{
    return id_by_name(name, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

void
iterate_args(uint32_t                                           id,
             const rocprofiler_callback_tracing_hsa_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t   callback,
             void*                                              user_data)
{
    if(callback)
        iterate_args(
            id, data, callback, user_data, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_HSA_API_ID_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
    return _data;
}

std::vector<const char*>
get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_HSA_API_ID_LAST);
    get_names(_data, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
    return _data;
}

void
set_callback(activity_functor_t _func)
{
    auto&& _v = report_activity.load();
    report_activity.compare_exchange_strong(_v, _func);
}

void
update_table(hsa_api_table_t* _orig)
{
    if(_orig) update_table(_orig, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}
}  // namespace hsa
}  // namespace rocprofiler
