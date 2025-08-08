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

#include "lib/common/defines.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/marker/marker.hpp"
#include "lib/rocprofiler-sdk/marker/utils.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker.h>

#include <rocprofiler-sdk-roctx/roctx.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace marker
{
namespace
{
struct null_type
{};

template <typename Tp>
auto
get_default_retval()
{
    if constexpr(std::is_integral<Tp>::value)
        return Tp{0};
    else
        static_assert(std::is_empty<Tp>::value, "Error! unsupported return type");
}

template <typename DataT, typename Tp>
void
set_data_retval(DataT& _data, [[maybe_unused]] Tp _val)
{
    if constexpr(std::is_same<int32_t, Tp>::value)
        _data.int32_t_retval = _val;
    else if constexpr(std::is_same<int64_t, Tp>::value)
        _data.int64_t_retval = _val;
    else if constexpr(std::is_same<roctx_range_id_t, Tp>::value)
        _data.roctx_range_id_t_retval = _val;
    else
        static_assert(std::is_empty<Tp>::value, "Error! unsupported return type");
}

template <typename Tp>
Tp*
get_table_impl()
{
    static auto*& _v = common::static_object<Tp>::construct(common::init_public_api_struct(Tp{}));
    return _v;
}

template <size_t TableIdx>
auto*
get_table();

struct range_data_t : public tracing::tracing_data
{
    using callback_api_data_t = rocprofiler_callback_tracing_marker_api_data_t;
    using buffered_api_data_t = rocprofiler_buffer_tracing_marker_api_record_t;

    callback_api_data_t      callback_data = common::init_public_api_struct(callback_api_data_t{});
    buffered_api_data_t      buffer_record = common::init_public_api_struct(buffered_api_data_t{});
    context::correlation_id* corr_id       = nullptr;
    rocprofiler_thread_id_t  thread_id     = common::get_tid();
};

auto&
get_range_thread_stack()
{
    static thread_local auto push_op_stack = common::container::small_vector<range_data_t, 8>{};
    return push_op_stack;
}

auto&
get_range_process_stack()
{
    static auto push_op_stack =
        common::Synchronized<std::unordered_map<roctx_range_id_t, range_data_t>>{};
    return push_op_stack;
}
}  // namespace

template <size_t TableIdx, size_t OpIdx>
template <typename DataArgsT, typename... Args>
auto
roctx_api_impl<TableIdx, OpIdx>::set_data_args(DataArgsT& _data_args, Args... args)
{
    if constexpr(sizeof...(Args) == 0)
        _data_args.no_args.empty = '\0';
    else
        _data_args = DataArgsT{args...};
}

template <size_t TableIdx, size_t OpIdx>
template <typename FuncT, typename... Args>
auto
roctx_api_impl<TableIdx, OpIdx>::exec(FuncT&& _func, Args&&... args)
{
    using return_type = std::decay_t<std::invoke_result_t<FuncT, Args...>>;

    if(_func)
    {
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

    using info_type = roctx_api_info<TableIdx, OpIdx>;
    ROCP_ERROR << "nullptr to next roctx function for " << info_type::name << " ("
               << info_type::operation_idx << ")";

    if constexpr(std::is_void<return_type>::value)
        return null_type{};
    else
        return get_default_retval<return_type>();
}

template <size_t TableIdx, size_t OpIdx>
template <typename RetT, typename... Args>
RetT
roctx_api_impl<TableIdx, OpIdx>::functor(Args... args)
{
    using info_type           = roctx_api_info<TableIdx, OpIdx>;
    using callback_api_data_t = typename roctx_domain_info<TableIdx>::callback_data_type;
    using buffered_api_data_t = typename roctx_domain_info<TableIdx>::buffer_data_type;

    constexpr auto external_corr_id_domain_idx =
        roctx_domain_info<TableIdx>::external_correlation_id_domain_idx;

    ROCP_INFO_IF(registration::get_fini_status() != 0) << "Executing " << info_type::name;

    auto thr_id            = common::get_tid();
    auto callback_contexts = tracing::callback_context_data_vec_t{};
    auto buffered_contexts = tracing::buffered_context_data_vec_t{};
    auto external_corr_ids = tracing::external_correlation_id_map_t{};

    tracing::populate_contexts(info_type::callback_domain_idx,
                               info_type::buffered_domain_idx,
                               info_type::operation_idx,
                               callback_contexts,
                               buffered_contexts,
                               external_corr_ids);

    if(callback_contexts.empty() && buffered_contexts.empty())
    {
        [[maybe_unused]] auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);
        if constexpr(!std::is_void<RetT>::value)
            return _ret;
        else
            return;
    }

    auto  ref_count        = 2;
    auto  buffer_record    = common::init_public_api_struct(buffered_api_data_t{});
    auto  callback_data    = common::init_public_api_struct(callback_api_data_t{});
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
        set_data_args(info_type::get_api_data_args(callback_data.args),
                      std::forward<Args>(args)...);

        tracing::execute_phase_enter_callbacks(callback_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::callback_domain_idx,
                                               info_type::operation_idx,
                                               callback_data);
    }

    // enter callback may update the external correlation id field
    tracing::update_external_correlation_ids(
        external_corr_ids, thr_id, external_corr_id_domain_idx);

    // record the start timestamp as close to the function call as possible
    if(!buffered_contexts.empty())
    {
        buffer_record.start_timestamp = common::timestamp_ns();
    }

    // decrement the reference count before invoking
    corr_id->sub_ref_count();

    auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);

    // record the end timestamp as close to the function call as possible
    if(!buffered_contexts.empty())
    {
        buffer_record.end_timestamp = common::timestamp_ns();
    }

    if(!callback_contexts.empty())
    {
        set_data_retval(callback_data.retval, _ret);

        tracing::execute_phase_exit_callbacks(callback_contexts,
                                              external_corr_ids,
                                              info_type::callback_domain_idx,
                                              info_type::operation_idx,
                                              callback_data);
    }

    if(!buffered_contexts.empty())
    {
        tracing::execute_buffer_record_emplace(buffered_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::buffered_domain_idx,
                                               info_type::operation_idx,
                                               buffer_record);
    }

    // decrement the reference count after usage in the callback/buffers
    corr_id->sub_ref_count();

    context::pop_latest_correlation_id(corr_id);

    if constexpr(!std::is_void<RetT>::value) return _ret;
}

template <size_t TableIdx, size_t OpIdx>
template <typename RetT, typename... Args>
RetT
roctx_api_impl<TableIdx, OpIdx>::push_functor(Args... args)
{
    using info_type = roctx_api_info<TableIdx, OpIdx>;

    constexpr auto external_corr_id_domain_idx =
        roctx_domain_info<TableIdx>::external_correlation_id_domain_idx;

    ROCP_INFO_IF(registration::get_fini_status() != 0) << "Executing " << info_type::name;

    auto  thr_id            = common::get_tid();
    auto  range_data        = range_data_t{};
    auto& external_corr_ids = range_data.external_correlation_ids;

    tracing::populate_contexts(info_type::callback_domain_idx,
                               info_type::buffered_domain_idx,
                               info_type::operation_idx,
                               range_data);

    if(range_data.empty())
    {
        [[maybe_unused]] auto _ret =
            exec(info_type::get_push_table_func(), std::forward<Args>(args)...);
        if constexpr(!std::is_void<RetT>::value)
            return _ret;
        else
            return;
    }

    auto   ref_count     = 1;
    auto&  buffer_record = range_data.buffer_record;
    auto&  callback_data = range_data.callback_data;
    auto*& corr_id       = range_data.corr_id;

    corr_id               = tracing::correlation_service::construct(ref_count);
    auto internal_corr_id = corr_id->internal;
    auto ancestor_corr_id = corr_id->ancestor;

    tracing::populate_external_correlation_ids(external_corr_ids,
                                               thr_id,
                                               external_corr_id_domain_idx,
                                               info_type::operation_idx,
                                               internal_corr_id);

    // invoke the callbacks
    if(!range_data.callback_contexts.empty())
    {
        set_data_args(info_type::get_api_data_args(callback_data.args),
                      std::forward<Args>(args)...);

        tracing::execute_phase_enter_callbacks(range_data.callback_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::callback_domain_idx,
                                               info_type::operation_idx,
                                               callback_data);
    }

    // enter callback may update the external correlation id field
    tracing::update_external_correlation_ids(
        external_corr_ids, thr_id, external_corr_id_domain_idx);

    // record the start timestamp as close to the function call as possible
    if(!range_data.buffered_contexts.empty())
    {
        buffer_record.start_timestamp = common::timestamp_ns();
    }

    auto _ret = exec(info_type::get_push_table_func(), std::forward<Args>(args)...);

    if(!range_data.callback_contexts.empty())
    {
        set_data_retval(callback_data.retval, _ret);
    }

    if constexpr(OpIdx == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA)
    {
        get_range_thread_stack().emplace_back(std::move(range_data));
    }
    else if constexpr(OpIdx == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxProcessRangeA)
    {
        // push the range data to the process stack
        get_range_process_stack().wlock(
            [](auto& _stack, auto _key, auto&& _range_data) {
                _stack.emplace(_key, std::move(_range_data));
            },
            _ret,
            std::move(range_data));
    }

    if constexpr(!std::is_void<RetT>::value) return _ret;
}

template <size_t TableIdx, size_t OpIdx>
template <typename RetT, typename... Args>
RetT
roctx_api_impl<TableIdx, OpIdx>::pop_functor(Args... args)
{
    using info_type = roctx_api_info<TableIdx, OpIdx>;

    auto range_data = range_data_t{};

    if constexpr(OpIdx == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA)
    {
        if(auto& _range_stack = get_range_thread_stack(); !_range_stack.empty())
        {
            // if the range API is used, we need to use the range tracing data
            // for push/pop operations, otherwise we can use the main API tracing
            range_data = _range_stack.back();
            _range_stack.pop_back();
        }
    }
    else if constexpr(OpIdx == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxProcessRangeA)
    {
        auto range_id = std::get<0>(std::tie(args...));
        static_assert(sizeof...(Args) == 1,
                      "roctxRangeStopA requires a single argument of type roctx_range_id_t");

        // push the range data to the process stack
        get_range_process_stack().wlock(
            [](auto& _range_stack, auto _key, auto& _dst) {
                // find the data for the range id if it exists, copy it and delete it
                if(auto itr = _range_stack.find(_key); itr != _range_stack.end())
                {
                    _dst = _range_stack.at(_key);
                    _range_stack.erase(itr);
                }
            },
            range_id,
            range_data);
    }

    auto _ret = exec(info_type::get_pop_table_func(), std::forward<Args>(args)...);

    if(range_data.empty())
    {
        if constexpr(!std::is_void<RetT>::value)
            return _ret;
        else
            return;
    }

    auto&  external_corr_ids = range_data.external_correlation_ids;
    auto&  buffer_record     = range_data.buffer_record;
    auto&  callback_data     = range_data.callback_data;
    auto*& corr_id           = range_data.corr_id;

    ROCP_FATAL_IF(!corr_id) << fmt::format("No correlation id found for range pop operation :: {}",
                                           info_type::name);

    auto thr_id           = range_data.thread_id;
    auto internal_corr_id = corr_id->internal;
    auto ancestor_corr_id = corr_id->ancestor;

    // record the end timestamp as close to the function call as possible
    if(!range_data.buffered_contexts.empty())
    {
        buffer_record.end_timestamp = common::timestamp_ns();
    }

    if(!range_data.callback_contexts.empty())
    {
        tracing::execute_phase_exit_callbacks(range_data.callback_contexts,
                                              external_corr_ids,
                                              info_type::callback_domain_idx,
                                              info_type::operation_idx,
                                              callback_data);
    }

    if(!range_data.buffered_contexts.empty())
    {
        tracing::execute_buffer_record_emplace(range_data.buffered_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::buffered_domain_idx,
                                               info_type::operation_idx,
                                               buffer_record);
    }

    // decrement the reference count after usage in the callback/buffers
    corr_id->sub_ref_count();

    context::pop_latest_correlation_id(corr_id);

    if constexpr(!std::is_void<RetT>::value) return _ret;
}
}  // namespace marker
}  // namespace rocprofiler

#define ROCPROFILER_LIB_ROCPROFILER_SDK_MARKER_RANGE_MARKER_CPP_IMPL 1

// template specializations
#include "range_marker.def.cpp"

namespace rocprofiler
{
namespace marker
{
namespace
{
template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return roctx_api_info<TableIdx, OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id<TableIdx>(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(std::string_view{roctx_api_info<TableIdx, OpIdx>::name} == std::string_view{name})
        return roctx_api_info<TableIdx, OpIdx>::operation_idx;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return id_by_name<TableIdx>(name, std::index_sequence<OpIdxTail...>{});
    else
        return roctx_domain_info<TableIdx>::none;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto _idx = roctx_api_info<TableIdx, OpIdx>::operation_idx;
    if(_idx < roctx_domain_info<TableIdx>::last) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0)
        get_ids<TableIdx>(_id_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto&& _name = roctx_api_info<TableIdx, OpIdx>::name;
    if(_name != nullptr && strnlen(_name, 1) > 0) _name_list.emplace_back(_name);

    if constexpr(sizeof...(OpIdxTail) > 0)
        get_names<TableIdx>(_name_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
iterate_args(const uint32_t                                        id,
             const rocprofiler_callback_tracing_marker_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t      func,
             int32_t                                               max_deref,
             void*                                                 user_data,
             std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id)
    {
        using info_type = roctx_api_info<TableIdx, OpIdx>;
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
        iterate_args<TableIdx>(
            id, data, func, max_deref, user_data, std::index_sequence<OpIdxTail...>{});
}
}  // namespace

// check out the assembly here... this compiles to a switch statement
template <size_t TableIdx>
const char*
name_by_id(uint32_t id)
{
    return name_by_id<TableIdx>(id, std::make_index_sequence<roctx_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
uint32_t
id_by_name(const char* name)
{
    return id_by_name<TableIdx>(name,
                                std::make_index_sequence<roctx_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
std::vector<uint32_t>
get_ids()
{
    constexpr auto last_api_id = roctx_domain_info<TableIdx>::last;
    auto           _data       = std::vector<uint32_t>{};
    _data.reserve(last_api_id);
    get_ids<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
std::vector<const char*>
get_names()
{
    constexpr auto last_api_id = roctx_domain_info<TableIdx>::last;
    auto           _data       = std::vector<const char*>{};
    _data.reserve(last_api_id);
    get_names<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
void
iterate_args(uint32_t                                              id,
             const rocprofiler_callback_tracing_marker_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t      callback,
             int32_t                                               max_deref,
             void*                                                 user_data)
{
    if(callback)
        iterate_args<TableIdx>(id,
                               data,
                               callback,
                               max_deref,
                               user_data,
                               std::make_index_sequence<roctx_domain_info<TableIdx>::last>{});
}

namespace range
{
namespace
{
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

template <size_t TableIdx, typename Tp, size_t OpIdx>
void
copy_table(Tp* _orig, uint64_t _tbl_instance, std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename roctx_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = roctx_api_info<TableIdx, OpIdx>{};

        if constexpr(_info.is_range)
        {
            // make sure we don't access a field that doesn't exist in input table
            // NOLINTNEXTLINE(misc-redundant-expression)
            if(_info.push_offset() >= _orig->size || _info.pop_offset() >= _orig->size) return;

            // 1. get the sub-table containing the function pointer in original table
            // 2. get reference to function pointer in sub-table in original table
            auto& _orig_table     = _info.get_table(_orig);
            auto& _orig_push_func = _info.get_push_table_func(_orig_table);
            auto& _orig_pop_func  = _info.get_pop_table_func(_orig_table);
            // 3. get the sub-table containing the function pointer in saved table
            // 4. get reference to function pointer in sub-table in saved table
            // 5. save the original function in the saved table
            auto& _copy_table     = _info.get_table(*get_table<TableIdx>());
            auto& _push_copy_func = _info.get_push_table_func(_copy_table);
            auto& _pop_copy_func  = _info.get_pop_table_func(_copy_table);

            ROCP_FATAL_IF(_push_copy_func && _tbl_instance == 0)
                << _info.name << " has non-null function pointer " << _push_copy_func
                << " despite this being the first instance of the library being copies";

            ROCP_FATAL_IF(_pop_copy_func && _tbl_instance == 0)
                << _info.name << " has non-null function pointer " << _pop_copy_func
                << " despite this being the first instance of the library being copies";

            if(!_push_copy_func || !_pop_copy_func)
            {
                ROCP_TRACE << "copying table entry for " << _info.name;
                _push_copy_func = _orig_push_func;
                _pop_copy_func  = _orig_pop_func;
            }
            else
            {
                ROCP_TRACE << "skipping copying table entry for " << _info.name
                           << " from table instance " << _tbl_instance;
            }
        }
        else
        {
            // make sure we don't access a field that doesn't exist in input table
            if(_info.offset() >= _orig->size) return;

            // 1. get the sub-table containing the function pointer in original table
            // 2. get reference to function pointer in sub-table in original table
            auto& _orig_table = _info.get_table(_orig);
            auto& _orig_func  = _info.get_table_func(_orig_table);
            // 3. get the sub-table containing the function pointer in saved table
            // 4. get reference to function pointer in sub-table in saved table
            // 5. save the original function in the saved table
            auto& _copy_table = _info.get_table(*get_table<TableIdx>());
            auto& _copy_func  = _info.get_table_func(_copy_table);

            ROCP_FATAL_IF(_copy_func && _tbl_instance == 0)
                << _info.name << " has non-null function pointer " << _copy_func
                << " despite this being the first instance of the library being copies";

            if(!_copy_func)
            {
                ROCP_TRACE << "copying table entry for " << _info.name;
                _copy_func = _orig_func;
            }
            else
            {
                ROCP_TRACE << "skipping copying table entry for " << _info.name
                           << " from table instance " << _tbl_instance;
            }
        }
    }
}

template <size_t TableIdx, typename Tp, size_t OpIdx>
void
update_table(Tp* _orig, std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename roctx_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = roctx_api_info<TableIdx, OpIdx>{};

        if constexpr(_info.is_range)
        {
            // make sure we don't access a field that doesn't exist in input table
            // NOLINTNEXTLINE(misc-redundant-expression)
            if(_info.push_offset() >= _orig->size || _info.pop_offset() >= _orig->size) return;

            // check to see if there are any contexts which enable this operation in the ROCTX API
            // domain
            if(!should_wrap_functor(
                   _info.callback_domain_idx, _info.buffered_domain_idx, _info.operation_idx))
                return;

            ROCP_TRACE << "updating table entry for " << _info.name;

            // 1. get the sub-table containing the function pointer in original table
            // 2. get reference to function pointer in sub-table in original table
            // 3. update function pointer with wrapper
            auto& _table = _info.get_table(_orig);

            auto& _push_func = _info.get_push_table_func(_table);
            _push_func       = _info.get_push_functor(_push_func);

            auto& _pop_func = _info.get_pop_table_func(_table);
            _pop_func       = _info.get_pop_functor(_pop_func);
        }
        else
        {
            // make sure we don't access a field that doesn't exist in input table
            if(_info.offset() >= _orig->size) return;

            // check to see if there are any contexts which enable this operation in the ROCTX API
            // domain
            if(!should_wrap_functor(
                   _info.callback_domain_idx, _info.buffered_domain_idx, _info.operation_idx))
                return;

            ROCP_TRACE << "updating table entry for " << _info.name;

            // 1. get the sub-table containing the function pointer in original table
            // 2. get reference to function pointer in sub-table in original table
            // 3. update function pointer with wrapper
            auto& _table = _info.get_table(_orig);
            auto& _func  = _info.get_table_func(_table);
            _func        = _info.get_functor(_func);
        }
    }
}

template <size_t TableIdx, typename Tp, size_t OpIdx, size_t... OpIdxTail>
void
copy_table(Tp* _orig, uint64_t _tbl_instance, std::index_sequence<OpIdx, OpIdxTail...>)
{
    copy_table<TableIdx>(_orig, _tbl_instance, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0)
        copy_table<TableIdx>(_orig, _tbl_instance, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, typename Tp, size_t OpIdx, size_t... OpIdxTail>
void
update_table(Tp* _orig, std::index_sequence<OpIdx, OpIdxTail...>)
{
    update_table<TableIdx>(_orig, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0)
        update_table<TableIdx>(_orig, std::index_sequence<OpIdxTail...>{});
}
}  // namespace

template <typename TableT>
void
copy_table(TableT* _orig, uint64_t _tbl_instance)
{
    constexpr auto TableIdx = roctx_table_id_lookup<TableT>::value;
    if(_orig)
        copy_table<TableIdx>(
            _orig, _tbl_instance, std::make_index_sequence<roctx_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
update_table(TableT* _orig, uint64_t _instv)
{
    constexpr auto TableIdx = roctx_table_id_lookup<TableT>::value;
    if(_orig)
    {
        copy_table(_orig, _instv);
        update_table<TableIdx>(_orig,
                               std::make_index_sequence<roctx_domain_info<TableIdx>::last>{});
    }
}
}  // namespace range

using iterate_args_data_t = rocprofiler_callback_tracing_marker_api_data_t;
using iterate_args_cb_t   = rocprofiler_callback_tracing_operation_args_cb_t;

#define INSTANTIATE_MARKER_TABLE_FUNC(TABLE_TYPE, TABLE_IDX)                                       \
    template void        range::update_table<TABLE_TYPE>(TABLE_TYPE * _tbl, uint64_t _instv);      \
    template const char* name_by_id<TABLE_IDX>(uint32_t);                                          \
    template uint32_t    id_by_name<TABLE_IDX>(const char*);                                       \
    template std::vector<uint32_t>    get_ids<TABLE_IDX>();                                        \
    template std::vector<const char*> get_names<TABLE_IDX>();                                      \
    template void                     iterate_args<TABLE_IDX>(                                     \
        uint32_t, const iterate_args_data_t&, iterate_args_cb_t, int32_t, void*);

INSTANTIATE_MARKER_TABLE_FUNC(roctx_core_api_table_t, ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange)

#undef INSTANTIATE_MARKER_TABLE_FUNC
}  // namespace marker
}  // namespace rocprofiler
