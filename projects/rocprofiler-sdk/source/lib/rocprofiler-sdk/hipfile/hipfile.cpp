// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hipfile/hipfile.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/hipfile/utils.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hipfile/table_id.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace hipfile
{
namespace
{
struct null_type
{};

template <typename Tp>
auto
get_default_retval()
{
    if constexpr(std::is_same<Tp, const char*>::value)
    {
        return "UnknownString";
    }
    else if constexpr(std::is_pointer<Tp>::value)
    {
        Tp v = nullptr;
        return v;
    }
    else if constexpr(std::is_same<Tp, hipFileError_t>::value)
    {
        return hipFileError_t{hipFileInternalError, hipErrorUnknown};
    }
    else if constexpr(std::is_same<Tp, ssize_t>::value)
    {
        return static_cast<ssize_t>(-1);
    }
    else if constexpr(std::is_same<Tp, int64_t>::value)
    {
        return int64_t{0};
    }
    else if constexpr(std::is_void<Tp>::value)
    {
        return null_type{};
    }
    else
    {
        static_assert(std::is_empty<Tp>::value, "Error! unsupported return type");
    }
}

template <typename DataT, typename Tp>
void
set_data_retval(DataT& _data, Tp _val)
{
    if constexpr(std::is_same<Tp, hipFileError_t>::value)
    {
        _data.hipFileError_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, ssize_t>::value)
    {
        _data.ssize_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, int64_t>::value)
    {
        _data.int64_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, const char*>::value)
    {
        _data.const_charp_retval = _val;
    }
    else if constexpr(std::is_same<Tp, null_type>::value)
    {
        (void) _data;
        (void) _val;
    }
    else
    {
        static_assert(std::is_empty<Tp>::value, "Error! unsupported return type");
    }
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

template <typename Tp>
decltype(auto)
convert_arg(Tp&& val)
{
    using data_type = common::mpl::unqualified_type_t<Tp>;
    if constexpr(std::is_same<data_type, const char*>::value)
    {
        return common::get_string_entry(val ? val : "")->c_str();
    }
    else if constexpr(std::is_same<data_type, char*>::value)
    {
        return std::forward<Tp>(val);
    }
    else
    {
        static_assert(!common::mpl::is_string_type<data_type>::value,
                      "argument type is a string type. preceding if constexpr is incorrect");
        return std::forward<Tp>(val);
    }
}

}  // namespace

template <size_t TableIdx, size_t OpIdx>
template <typename DataArgsT, typename... Args>
auto
hipfile_api_impl<TableIdx, OpIdx>::set_data_args(DataArgsT& _data_args, Args... args)
{
    if constexpr(sizeof...(Args) == 0)
        _data_args.empty = '\0';
    else
        _data_args = DataArgsT{args...};
}

template <size_t TableIdx, size_t OpIdx>
template <typename FuncT, typename... Args>
auto
hipfile_api_impl<TableIdx, OpIdx>::exec(FuncT&& _func, Args&&... args)
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

    using info_type = hipfile_api_info<TableIdx, OpIdx>;
    ROCP_ERROR << "nullptr to next hipfile function for " << info_type::name << " ("
               << info_type::operation_idx << ")";

    return get_default_retval<return_type>();
}

template <size_t TableIdx, size_t OpIdx>
template <typename RetT, typename... Args>
RetT
hipfile_api_impl<TableIdx, OpIdx>::functor(Args... args)
{
    using info_type           = hipfile_api_info<TableIdx, OpIdx>;
    using callback_api_data_t = typename hipfile_domain_info<TableIdx>::callback_data_type;
    using buffered_api_data_t = typename hipfile_domain_info<TableIdx>::buffer_data_type;
    using buffered_ext_data_t = typename hipfile_domain_info<TableIdx>::buffered_ext_data_type;

    constexpr auto external_corr_id_domain_idx =
        hipfile_domain_info<TableIdx>::external_correlation_id_domain_idx;

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

    if(!callback_contexts.empty() || !extended_contexts.empty())
    {
        set_data_args(info_type::get_api_data_args(tracer_data.args),
                      convert_arg(std::forward<Args>(args))...);
    }

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

    tracing::update_external_correlation_ids(
        external_corr_ids, thr_id, external_corr_id_domain_idx);

    if(!buffered_contexts.empty() || !extended_contexts.empty())
    {
        buffer_record.start_timestamp = common::timestamp_ns();
    }

    corr_id->sub_ref_count();

    auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);

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

    corr_id->sub_ref_count();

    context::pop_latest_correlation_id(corr_id);

    if constexpr(!std::is_void<RetT>::value) return _ret;
}
}  // namespace hipfile
}  // namespace rocprofiler

#define ROCPROFILER_LIB_ROCPROFILER_SDK_HIPFILE_HIPFILE_CPP_IMPL 1

#include "hipfile.def.cpp"

namespace rocprofiler
{
namespace hipfile
{
namespace
{
template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return hipfile_api_info<TableIdx, OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id<TableIdx>(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(std::string_view{hipfile_api_info<TableIdx, OpIdx>::name} == std::string_view{name})
        return hipfile_api_info<TableIdx, OpIdx>::operation_idx;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return id_by_name<TableIdx>(name, std::index_sequence<OpIdxTail...>{});
    else
        return hipfile_domain_info<TableIdx>::none;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto _idx = hipfile_api_info<TableIdx, OpIdx>::operation_idx;
    if(_idx < hipfile_domain_info<TableIdx>::last) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0)
        get_ids<TableIdx>(_id_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto&& _name = hipfile_api_info<TableIdx, OpIdx>::name;
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
        using info_type = hipfile_api_info<TableIdx, OpIdx>;
        auto&& arg_list = info_type::as_arg_list(data, max_deref);
        auto&& arg_addr = info_type::as_arg_addr(data);
        for(size_t i = 0; i < std::min(arg_list.size(), arg_addr.size()); ++i)
        {
            int ret = 0;
            if constexpr(std::is_same<FuncT,
                                      rocprofiler_callback_tracing_operation_args_cb_t>::value)
            {
                ret = func(info_type::callback_domain_idx,
                           id,
                           i,
                           arg_addr.at(i),
                           arg_list.at(i).indirection_level,
                           arg_list.at(i).type,
                           arg_list.at(i).name,
                           arg_list.at(i).value.c_str(),
                           arg_list.at(i).dereference_count,
                           user_data);
            }
            else if constexpr(std::is_same<FuncT,
                                           rocprofiler_buffer_tracing_operation_args_cb_t>::value)
            {
                ret = func(info_type::buffered_ext_domain_idx,
                           id,
                           i,
                           arg_addr.at(i),
                           arg_list.at(i).indirection_level,
                           arg_list.at(i).type,
                           arg_list.at(i).name,
                           arg_list.at(i).value.c_str(),
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
    for(const auto& itr : context::get_registered_contexts())
    {
        if(!itr) continue;

        if(itr->callback_tracer && itr->callback_tracer->domains(_callback_domain) &&
           itr->callback_tracer->domains(_callback_domain, _operation))
            return true;

        if(itr->buffered_tracer && itr->buffered_tracer->domains(_buffered_domain) &&
           itr->buffered_tracer->domains(_buffered_domain, _operation))
            return true;

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
    using table_type = typename hipfile_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = hipfile_api_info<TableIdx, OpIdx>{};

        if(_info.offset() >= _orig->size) return;

        auto& _orig_table = _info.get_table(_orig);
        auto& _orig_func  = _info.get_table_func(_orig_table);
        auto& _copy_table = _info.get_table(*get_table<TableIdx>());
        auto& _copy_func  = _info.get_table_func(_copy_table);

        ROCP_FATAL_IF(_copy_func && _tbl_instance == 0)
            << _info.name << " has non-null function pointer " << _copy_func
            << " despite this being the first instance of the library being copied";

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
    using table_type = typename hipfile_table_lookup<TableIdx>::type;

    static_assert(OpIdx < context::domain_ops_padding,
                  "operation index exceeds context domain ops padding");

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = hipfile_api_info<TableIdx, OpIdx>{};

        if(_info.offset() >= _orig->size) return;

        if(!should_wrap_functor(_info.callback_domain_idx,
                                _info.buffered_domain_idx,
                                _info.buffered_ext_domain_idx,
                                _info.operation_idx))
            return;

        ROCP_TRACE << "updating table entry for " << _info.name;

        auto& _table = _info.get_table(_orig);
        auto& _func  = _info.get_table_func(_table);
        if(_func) _func = _info.get_functor(_func);
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

template <size_t TableIdx>
const char*
name_by_id(uint32_t id)
{
    return name_by_id<TableIdx>(id,
                                std::make_index_sequence<hipfile_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
uint32_t
id_by_name(const char* name)
{
    return id_by_name<TableIdx>(name,
                                std::make_index_sequence<hipfile_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
std::vector<uint32_t>
get_ids()
{
    constexpr auto last_api_id = hipfile_domain_info<TableIdx>::last;
    auto           _data       = std::vector<uint32_t>{};
    _data.reserve(last_api_id);
    get_ids<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
std::vector<const char*>
get_names()
{
    constexpr auto last_api_id = hipfile_domain_info<TableIdx>::last;
    auto           _data       = std::vector<const char*>{};
    _data.reserve(last_api_id);
    get_names<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
void
iterate_args(uint32_t                                         id,
             const rocprofiler_hipfile_api_args_t&            data,
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
                               std::make_index_sequence<hipfile_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
void
iterate_args(uint32_t                                       id,
             const rocprofiler_hipfile_api_args_t&          data,
             rocprofiler_buffer_tracing_operation_args_cb_t callback,
             void*                                          user_data)
{
    if(callback)
        iterate_args<TableIdx>(id,
                               data,
                               callback,
                               0,
                               user_data,
                               std::make_index_sequence<hipfile_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
copy_table(TableT* _orig, uint64_t _tbl_instance)
{
    constexpr auto TableIdx = hipfile_table_id_lookup<TableT>::value;
    if(_orig)
        copy_table<TableIdx>(
            _orig, _tbl_instance, std::make_index_sequence<hipfile_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
update_table(TableT* _orig)
{
    constexpr auto TableIdx = hipfile_table_id_lookup<TableT>::value;
    if(_orig)
        update_table<TableIdx>(_orig,
                               std::make_index_sequence<hipfile_domain_info<TableIdx>::last>{});
}

using hipfile_api_args_t      = rocprofiler_hipfile_api_args_t;
using hipfile_cb_op_args_cb_t = rocprofiler_callback_tracing_operation_args_cb_t;
using hipfile_bf_op_args_cb_t = rocprofiler_buffer_tracing_operation_args_cb_t;

#define INSTANTIATE_HIPFILE_TABLE_FUNC(TABLE_TYPE, TABLE_IDX)                                              \
    template void                     copy_table<TABLE_TYPE>(TABLE_TYPE * _tbl, uint64_t _instv);          \
    template void                     update_table<TABLE_TYPE>(TABLE_TYPE * _tbl);                         \
    template const char*              name_by_id<TABLE_IDX>(uint32_t);                                     \
    template uint32_t                 id_by_name<TABLE_IDX>(const char*);                                  \
    template std::vector<uint32_t>    get_ids<TABLE_IDX>();                                                \
    template std::vector<const char*> get_names<TABLE_IDX>();                                              \
    template void                     iterate_args<TABLE_IDX>(                                             \
        uint32_t, const hipfile_api_args_t&, hipfile_cb_op_args_cb_t, int32_t, void*); \
    template void iterate_args<TABLE_IDX>(                                                                 \
        uint32_t, const hipfile_api_args_t&, hipfile_bf_op_args_cb_t, void*);

INSTANTIATE_HIPFILE_TABLE_FUNC(hipfile_api_func_table_t, ROCPROFILER_HIPFILE_TABLE_ID_CORE)
}  // namespace hipfile
}  // namespace rocprofiler
