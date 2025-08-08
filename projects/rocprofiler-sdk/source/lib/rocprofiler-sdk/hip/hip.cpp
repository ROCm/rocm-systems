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

#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hip/utils.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hip/driver_types.h>
#include <hip/hip_runtime_api.h>
// must be included after runtime api
#include <hip/hip_deprecated.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace hip
{
namespace
{
struct null_type
{};

template <typename Tp>
auto
get_default_retval()
{
    if constexpr(std::is_pointer<Tp>::value)
    {
        Tp v = nullptr;
        return v;
    }
    else if constexpr(std::is_same<Tp, hipError_t>::value)
        return hipErrorUnknown;
    else if constexpr(std::is_same<Tp, hipChannelFormatDesc>::value)
        return hipChannelFormatDesc{};
    else if constexpr(std::is_same<Tp, int>::value)
        return -1;
    else if constexpr(std::is_void<Tp>::value)
        return null_type{};
    else
        static_assert(std::is_empty<Tp>::value, "Error! unsupported return type");
}

template <typename DataT, typename Tp>
void
set_data_retval(DataT& _data, Tp _val)
{
    if constexpr(std::is_same<Tp, null_type>::value)
    {
        (void) _data;
        (void) _val;
    }
    else if constexpr(std::is_same<Tp, hipError_t>::value)
    {
        _data.hipError_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, hipChannelFormatDesc>::value)
    {
        _data.hipChannelFormatDesc_retval = _val;
    }
    else if constexpr(std::is_same<Tp, const char*>::value)
    {
        _data.const_charp_retval = _val;
    }
    else if constexpr(std::is_same<Tp, void**>::value)
    {
        _data.voidpp_retval = _val;
    }
    else if constexpr(std::is_same<Tp, int>::value)
    {
        _data.int_retval = _val;
    }
    else
    {
        static_assert(std::is_empty<Tp>::value, "Error! unsupported return type");
    }
}

template <typename Tp>
decltype(auto)
convert_arg_type(Tp&& val)
{
    using data_type = common::mpl::unqualified_type_t<Tp>;
    if constexpr(std::is_same<data_type, dim3>::value)
    {
        return rocprofiler_dim3_t{val.x, val.y, val.z};
    }
    else
    {
        return std::forward<Tp>(val);
    }
}
}  // namespace

hip_api_table_t&
get_table()
{
    static auto _compiler = hip_compiler_api_table_t{};
    static auto _runtime  = hip_runtime_api_table_t{};
    static auto _v        = []() {
        _compiler.size = sizeof(_compiler);
        _runtime.size  = sizeof(_runtime);
        auto _val      = hip_api_table_t{&_compiler, &_runtime};
        return _val;
    }();
    return _v;
}

template <size_t TableIdx, size_t OpIdx>
template <typename DataArgsT, typename... Args>
auto
hip_api_impl<TableIdx, OpIdx>::set_data_args(DataArgsT& _data_args, Args... args)
{
    if constexpr(sizeof...(Args) == 0)
        _data_args.no_args.empty = '\0';
    else
        _data_args = DataArgsT{args...};
}

template <size_t TableIdx, size_t OpIdx>
template <typename FuncT, typename... Args>
auto
hip_api_impl<TableIdx, OpIdx>::exec(FuncT&& _func, Args&&... args)
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

    using info_type = hip_api_info<TableIdx, OpIdx>;
    ROCP_ERROR << "nullptr to next hip function for " << info_type::name << " ("
               << info_type::operation_idx << ")";

    return get_default_retval<return_type>();
}

template <size_t TableIdx, size_t OpIdx>
template <typename RetT, typename... Args>
RetT
hip_api_impl<TableIdx, OpIdx>::functor(Args... args)
{
    using info_type           = hip_api_info<TableIdx, OpIdx>;
    using callback_api_data_t = typename hip_domain_info<TableIdx>::callback_data_type;
    using buffered_api_data_t = typename hip_domain_info<TableIdx>::buffered_data_type;
    using buffered_ext_data_t = typename hip_domain_info<TableIdx>::buffered_ext_data_type;

    constexpr auto external_corr_id_domain_idx =
        hip_domain_info<TableIdx>::external_correlation_id_domain_idx;

    if(registration::get_fini_status() != 0)
    {
        [[maybe_unused]] auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);
        if constexpr(!std::is_void<RetT>::value)
            return _ret;
        else
            return;
    }

    constexpr auto ref_count         = 2;
    auto           thr_id            = common::get_tid();
    auto           callback_contexts = tracing::callback_context_data_vec_t{};
    auto           buffered_contexts = tracing::buffered_context_data_vec_t{};
    auto           extended_contexts = tracing::buffered_context_data_vec_t{};
    auto           external_corr_ids = tracing::external_correlation_id_map_t{};

    tracing::populate_contexts(info_type::callback_domain_idx,
                               info_type::buffered_domain_idx,
                               info_type::operation_idx,
                               callback_contexts,
                               buffered_contexts,
                               external_corr_ids);

    tracing::populate_contexts(info_type::buffered_ext_domain_idx,
                               info_type::operation_idx,
                               extended_contexts,
                               external_corr_ids);

    if(callback_contexts.empty() && buffered_contexts.empty() && extended_contexts.empty())
    {
        [[maybe_unused]] auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);
        if constexpr(!std::is_void<RetT>::value)
            return _ret;
        else
            return;
    }

    auto  buffer_record    = common::init_public_api_struct(buffered_api_data_t{});
    auto  extended_record  = common::init_public_api_struct(buffered_ext_data_t{});
    auto  tracer_data      = common::init_public_api_struct(callback_api_data_t{});
    auto* corr_id          = tracing::correlation_service::construct(ref_count);
    auto  internal_corr_id = corr_id->internal;
    auto  ancestor_corr_id = corr_id->ancestor;

    tracing::populate_external_correlation_ids(external_corr_ids,
                                               thr_id,
                                               external_corr_id_domain_idx,
                                               info_type::operation_idx,
                                               internal_corr_id);

    // set the arguments
    if(!callback_contexts.empty() || !extended_contexts.empty())
    {
        set_data_args(info_type::get_api_data_args(tracer_data.args),
                      convert_arg_type(std::forward<Args>(args))...);
    }

    // invoke the callbacks
    if(!callback_contexts.empty())
    {
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

    // record the start timestamp as close to the function call as possible
    if(!buffered_contexts.empty() || !extended_contexts.empty())
    {
        buffer_record.start_timestamp = common::timestamp_ns();
    }

    // decrement the reference count before invoking
    corr_id->sub_ref_count();

    auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);

    // record the end timestamp as close to the function call as possible
    if(!buffered_contexts.empty() || !extended_contexts.empty())
    {
        buffer_record.end_timestamp = common::timestamp_ns();
    }

    if(!callback_contexts.empty() || !extended_contexts.empty())
    {
        set_data_retval(tracer_data.retval, _ret);
    }

    if(!callback_contexts.empty())
    {
        tracing::execute_phase_exit_callbacks(callback_contexts,
                                              external_corr_ids,
                                              info_type::callback_domain_idx,
                                              info_type::operation_idx,
                                              tracer_data);
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

    if(!extended_contexts.empty())
    {
        extended_record.start_timestamp = buffer_record.start_timestamp;
        extended_record.end_timestamp   = buffer_record.end_timestamp;
        extended_record.args            = tracer_data.args;
        extended_record.retval          = tracer_data.retval;

        tracing::execute_buffer_record_emplace(extended_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               external_corr_ids,
                                               ancestor_corr_id,
                                               info_type::buffered_ext_domain_idx,
                                               info_type::operation_idx,
                                               extended_record);
    }

    // decrement the reference count after usage in the callback/buffers
    corr_id->sub_ref_count();

    context::pop_latest_correlation_id(corr_id);

    if constexpr(!std::is_void<RetT>::value) return _ret;
}
}  // namespace hip
}  // namespace rocprofiler

#define ROCPROFILER_LIB_ROCPROFILER_HIP_HIP_CPP_IMPL 1

// template specializations
#include "hip.def.cpp"

namespace rocprofiler
{
namespace hip
{
namespace
{
template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return hip_api_info<TableIdx, OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id<TableIdx>(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(std::string_view{hip_api_info<TableIdx, OpIdx>::name} == std::string_view{name})
        return hip_api_info<TableIdx, OpIdx>::operation_idx;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return id_by_name<TableIdx>(name, std::index_sequence<OpIdxTail...>{});
    else
        return hip_domain_info<TableIdx>::none;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto _idx = hip_api_info<TableIdx, OpIdx>::operation_idx;
    if(_idx < hip_domain_info<TableIdx>::last) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0)
        get_ids<TableIdx>(_id_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto&& _name = hip_api_info<TableIdx, OpIdx>::name;
    if(_name != nullptr && strnlen(_name, 1) > 0) _name_list.emplace_back(_name);

    if constexpr(sizeof...(OpIdxTail) > 0)
        get_names<TableIdx>(_name_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, typename DataT, typename FuncT, size_t OpIdx, size_t... OpIdxTail>
void
iterate_args(const uint32_t id,
             const DataT&   data,
             FuncT          func,
             int32_t        max_deref,
             void*          user_data,
             std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id)
    {
        using info_type = hip_api_info<TableIdx, OpIdx>;
        auto&& arg_list = info_type::as_arg_list(data, max_deref);
        auto&& arg_addr = info_type::as_arg_addr(data);
        for(size_t i = 0; i < std::min(arg_list.size(), arg_addr.size()); ++i)
        {
            using return_type = typename common::mpl::function_traits<FuncT>::result_type;

            auto ret = return_type{};
            if constexpr(std::is_same<FuncT,
                                      rocprofiler_callback_tracing_operation_args_cb_t>::value)
            {
                ret = func(info_type::callback_domain_idx,    // kind
                           id,                                // operation
                           i,                                 // arg_number
                           arg_addr.at(i),                    // arg_value_addr
                           arg_list.at(i).indirection_level,  // indirection
                           arg_list.at(i).type,               // arg_type
                           arg_list.at(i).name,               // arg_name
                           arg_list.at(i).value.c_str(),      // arg_value_str
                           arg_list.at(i).dereference_count,  // num deref in str
                           user_data);
            }
            else if constexpr(std::is_same<FuncT,
                                           rocprofiler_buffer_tracing_operation_args_cb_t>::value)
            {
                ret = func(info_type::buffered_ext_domain_idx,  // kind
                           id,                                  // operation
                           i,                                   // arg_number
                           arg_addr.at(i),                      // arg_value_addr
                           arg_list.at(i).indirection_level,    // indirection
                           arg_list.at(i).type,                 // arg_type
                           arg_list.at(i).name,                 // arg_name
                           arg_list.at(i).value.c_str(),        // arg_value_str
                           user_data);
            }
            else
            {
                static_assert(common::mpl::assert_false<FuncT>::value,
                              "Error! unsupported callback type");
            }

            if(ret != 0) break;
        }
        return;
    }
    if constexpr(sizeof...(OpIdxTail) > 0)
        iterate_args<TableIdx>(
            id, data, func, max_deref, user_data, std::index_sequence<OpIdxTail...>{});
}

bool
should_wrap_functor(rocprofiler_callback_tracing_kind_t _callback_domain,
                    rocprofiler_buffer_tracing_kind_t   _buffered_domain,
                    rocprofiler_buffer_tracing_kind_t   _buffered_ext_domain,
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

        // if there is a buffered tracer enabled for the given domain and op, we need to wrap
        if(itr->buffered_tracer && itr->buffered_tracer->domains(_buffered_ext_domain) &&
           itr->buffered_tracer->domains(_buffered_ext_domain, _operation))
            return true;
    }
    return false;
}

template <size_t TableIdx, typename Tp, size_t OpIdx>
void
copy_table(Tp* _orig, uint64_t _tbl_instance, std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename hip_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = hip_api_info<TableIdx, OpIdx>{};

        // make sure we don't access a field that doesn't exist in input table
        if(_info.offset() >= _orig->size) return;

        // 1. get the sub-table containing the function pointer in original table
        // 2. get reference to function pointer in sub-table in original table
        auto& _orig_table = _info.get_table(_orig);
        auto& _orig_func  = _info.get_table_func(_orig_table);
        // 3. get the sub-table containing the function pointer in saved table
        // 4. get reference to function pointer in sub-table in saved table
        // 5. save the original function in the saved table
        auto& _copy_table = _info.get_table(get_table());
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

template <size_t TableIdx, typename Tp, size_t OpIdx>
void
update_table(Tp* _orig, std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename hip_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = hip_api_info<TableIdx, OpIdx>{};

        // make sure we don't access a field that doesn't exist in input table
        if(_info.offset() >= _orig->size) return;

        // check to see if there are any contexts which enable this operation in the HIP API domain
        if(!should_wrap_functor(_info.callback_domain_idx,
                                _info.buffered_domain_idx,
                                _info.buffered_ext_domain_idx,
                                _info.operation_idx))
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

// check out the assembly here... this compiles to a switch statement
template <size_t TableIdx>
const char*
name_by_id(uint32_t id)
{
    return name_by_id<TableIdx>(id, std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
uint32_t
id_by_name(const char* name)
{
    return id_by_name<TableIdx>(name, std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
std::vector<uint32_t>
get_ids()
{
    constexpr auto last_api_id = hip_domain_info<TableIdx>::last;
    auto           _data       = std::vector<uint32_t>{};
    _data.reserve(last_api_id);
    get_ids<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
std::vector<const char*>
get_names()
{
    constexpr auto last_api_id = hip_domain_info<TableIdx>::last;
    auto           _data       = std::vector<const char*>{};
    _data.reserve(last_api_id);
    get_names<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
void
iterate_args(uint32_t                                         id,
             const rocprofiler_hip_api_args_t&                data,
             rocprofiler_callback_tracing_operation_args_cb_t callback,
             int32_t                                          max_deref,
             void*                                            user_data)
{
    if(callback)
        iterate_args<TableIdx>(id,
                               data,
                               callback,
                               max_deref,
                               user_data,
                               std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
void
iterate_args(uint32_t                                       id,
             const rocprofiler_hip_api_args_t&              data,
             rocprofiler_buffer_tracing_operation_args_cb_t callback,
             void*                                          user_data)
{
    if(callback)
        iterate_args<TableIdx>(id,
                               data,
                               callback,
                               0,
                               user_data,
                               std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
copy_table(TableT* _orig, uint64_t _tbl_instance)
{
    constexpr auto TableIdx = hip_table_id_lookup<TableT>::value;
    if(_orig)
        copy_table<TableIdx>(
            _orig, _tbl_instance, std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
update_table(TableT* _orig)
{
    constexpr auto TableIdx = hip_table_id_lookup<TableT>::value;
    if(_orig)
        update_table<TableIdx>(_orig, std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
}

using hip_api_data_t   = rocprofiler_hip_api_args_t;
using hip_op_args_cb_t = rocprofiler_callback_tracing_operation_args_cb_t;
using hip_op_args_bf_t = rocprofiler_buffer_tracing_operation_args_cb_t;

#define INSTANTIATE_HIP_TABLE_FUNC(TABLE_TYPE, TABLE_IDX)                                          \
    template void                     copy_table<TABLE_TYPE>(TABLE_TYPE * _tbl, uint64_t _instv);  \
    template void                     update_table<TABLE_TYPE>(TABLE_TYPE * _tbl);                 \
    template const char*              name_by_id<TABLE_IDX>(uint32_t);                             \
    template uint32_t                 id_by_name<TABLE_IDX>(const char*);                          \
    template std::vector<uint32_t>    get_ids<TABLE_IDX>();                                        \
    template std::vector<const char*> get_names<TABLE_IDX>();                                      \
    template void                     iterate_args<TABLE_IDX>(                                     \
        uint32_t, const hip_api_data_t&, hip_op_args_cb_t, int32_t, void*);    \
    template void iterate_args<TABLE_IDX>(uint32_t, const hip_api_data_t&, hip_op_args_bf_t, void*);

INSTANTIATE_HIP_TABLE_FUNC(hip_runtime_api_table_t, ROCPROFILER_HIP_TABLE_ID_Runtime)
INSTANTIATE_HIP_TABLE_FUNC(hip_compiler_api_table_t, ROCPROFILER_HIP_TABLE_ID_Compiler)
}  // namespace hip
}  // namespace rocprofiler
