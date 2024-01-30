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
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/details/ostream.hpp"
#include "lib/rocprofiler-sdk/hsa/types.hpp"
#include "lib/rocprofiler-sdk/hsa/utils.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa/api_id.h>

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

// helper to ensure that the version variable is initialized with the correct major version, minor
// version (sizeof), and step version
#define DEFINE_TABLE_VERSION_IMPL(VARIABLE, OBJECT, MAJOR_VERSION, STEP_VERSION)                   \
    constexpr auto VARIABLE = hsa_table_version_t{MAJOR_VERSION, sizeof(OBJECT), STEP_VERSION, 0};

// helper to ensure that the version variable is initialized with the correct major version, minor
// version (sizeof), and step version
#define DEFINE_TABLE_VERSION(ALIAS, NAME)                                                          \
    DEFINE_TABLE_VERSION_IMPL(hsa_##ALIAS##_table_version,                                         \
                              hsa_##ALIAS##_table_t,                                               \
                              HSA_##NAME##_TABLE_MAJOR_VERSION,                                    \
                              HSA_##NAME##_TABLE_STEP_VERSION)

DEFINE_TABLE_VERSION(api, API)
DEFINE_TABLE_VERSION(core, CORE_API)
DEFINE_TABLE_VERSION(amd_ext, AMD_EXT_API)
DEFINE_TABLE_VERSION(fini_ext, FINALIZER_API)
DEFINE_TABLE_VERSION(img_ext, IMAGE_API)

#undef DEFINE_TABLE_VERSION
#undef DEFINE_TABLE_VERSION_IMPL

template <typename Tp>
Tp*&
get_table_impl(hsa_table_version_t _version)
{
    static_assert(common::static_object<Tp>::is_trivial_standard_layout(),
                  "This HSA API table is not a trivial, standard layout type as it should be");

    auto*& val   = common::static_object<Tp>::construct();
    val->version = _version;
    return val;
}
}  // namespace

hsa_table_version_t
get_table_version()
{
    return hsa_api_table_version;
}

// helper to ensure that table type is paired with the correct table version
#define GET_TABLE_IMPL(ALIAS) get_table_impl<hsa_##ALIAS##_table_t>(hsa_##ALIAS##_table_version);

hsa_core_table_t*
get_core_table()
{
    static auto*& val = GET_TABLE_IMPL(core);
    return val;
}

hsa_amd_ext_table_t*
get_amd_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(amd_ext);
    return val;
}

hsa_fini_ext_table_t*
get_fini_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(fini_ext);
    return val;
}

hsa_img_ext_table_t*
get_img_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(img_ext);
    return val;
}

#undef GET_TABLE_IMPL

hsa_api_table_t&
get_table()
{
    static auto tbl = hsa_api_table_t{.version        = hsa_api_table_version,
                                      .core_          = get_core_table(),
                                      .amd_ext_       = get_amd_ext_table(),
                                      .finalizer_ext_ = get_fini_ext_table(),
                                      .image_ext_     = get_img_ext_table()};
    return tbl;
}

template <size_t TableIdx, size_t OpIdx>
template <typename DataArgsT, typename... Args>
auto
hsa_api_impl<TableIdx, OpIdx>::set_data_args(DataArgsT& _data_args, Args... args)
{
    _data_args = DataArgsT{args...};
}

template <size_t TableIdx, size_t OpIdx>
template <typename FuncT, typename... Args>
auto
hsa_api_impl<TableIdx, OpIdx>::exec(FuncT&& _func, Args&&... args)
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

namespace
{
using correlation_service     = context::correlation_tracing_service;
using buffer_hsa_api_record_t = rocprofiler_buffer_tracing_hsa_api_record_t;
using callback_hsa_api_data_t = rocprofiler_callback_tracing_hsa_api_data_t;

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

constexpr auto empty_user_data = rocprofiler_user_data_t{.value = 0};

void
populate_contexts(rocprofiler_callback_tracing_kind_t callback_domain_idx,
                  rocprofiler_buffer_tracing_kind_t   buffered_domain_idx,
                  int                                 operation_idx,
                  std::vector<callback_context_data>& callback_contexts,
                  std::vector<buffered_context_data>& buffered_contexts)
{
    auto active_contexts = context::context_array_t{};
    auto thr_id          = common::get_tid();
    for(const auto* itr : context::get_active_contexts(active_contexts))
    {
        if(!itr) continue;

        // if(itr->pc_sampler) has_pc_sampling = true;

        if(itr->callback_tracer)
        {
            // if the given domain + op is not enabled, skip this context
            if(itr->callback_tracer->domains(callback_domain_idx, operation_idx))
                callback_contexts.emplace_back(
                    callback_context_data{itr, rocprofiler_callback_tracing_record_t{}});
        }

        if(itr->buffered_tracer)
        {
            // if the given domain + op is not enabled, skip this context
            if(itr->buffered_tracer->domains(buffered_domain_idx, operation_idx))
                buffered_contexts.emplace_back(buffered_context_data{
                    itr, itr->correlation_tracer.external_correlator.get(thr_id)});
        }
    }
}
}  // namespace

template <size_t TableIdx, size_t OpIdx>
template <typename... Args>
auto
hsa_api_impl<TableIdx, OpIdx>::functor(Args&&... args)
{
    using info_type = hsa_api_info<TableIdx, OpIdx>;

    if(registration::get_fini_status() != 0)
    {
        auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);
        if constexpr(!std::is_same<decltype(_ret), null_type>::value)
            return _ret;
        else
            return HSA_STATUS_SUCCESS;
    }

    auto thr_id            = common::get_tid();
    auto callback_contexts = std::vector<callback_context_data>{};
    auto buffered_contexts = std::vector<buffered_context_data>{};
    auto has_pc_sampling   = false;

    populate_contexts(info_type::callback_domain_idx,
                      info_type::buffered_domain_idx,
                      info_type::operation_idx,
                      callback_contexts,
                      buffered_contexts);

    if(callback_contexts.empty() && buffered_contexts.empty())
    {
        auto _ret = exec(info_type::get_table_func(), std::forward<Args>(args)...);
        if constexpr(!std::is_same<decltype(_ret), null_type>::value)
            return _ret;
        else
            return HSA_STATUS_SUCCESS;
    }

    auto  ref_count        = (has_pc_sampling) ? 4 : 2;
    auto  buffer_record    = common::init_public_api_struct(buffer_hsa_api_record_t{});
    auto  tracer_data      = common::init_public_api_struct(callback_hsa_api_data_t{});
    auto* corr_id          = correlation_service::construct(ref_count);
    auto  internal_corr_id = corr_id->internal;

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
template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return hsa_api_info<TableIdx, OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id<TableIdx>(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(std::string_view{hsa_api_info<TableIdx, OpIdx>::name} == std::string_view{name})
        return hsa_api_info<TableIdx, OpIdx>::operation_idx;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return id_by_name<TableIdx>(name, std::index_sequence<OpIdxTail...>{});
    else
        return hsa_domain_info<TableIdx>::none;
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    uint32_t _idx = hsa_api_info<TableIdx, OpIdx>::operation_idx;
    if(_idx < hsa_domain_info<TableIdx>::last) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0)
        get_ids<TableIdx>(_id_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, size_t OpIdx, size_t... OpIdxTail>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto&& _name = hsa_api_info<TableIdx, OpIdx>::name;
    if(_name != nullptr && strnlen(_name, 1) > 0) _name_list.emplace_back(_name);

    if constexpr(sizeof...(OpIdxTail) > 0)
        get_names<TableIdx>(_name_list, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, size_t OpIdx, size_t... IdxTail>
void
iterate_args(const uint32_t                                     id,
             const rocprofiler_callback_tracing_hsa_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t   func,
             void*                                              user_data,
             std::index_sequence<OpIdx, IdxTail...>)
{
    if(OpIdx == id)
    {
        using info_type = hsa_api_info<TableIdx, OpIdx>;
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
        iterate_args<TableIdx>(id, data, func, user_data, std::index_sequence<IdxTail...>{});
}

bool
should_wrap_functor(const context::context_array_t&     _contexts,
                    rocprofiler_callback_tracing_kind_t _callback_domain,
                    rocprofiler_buffer_tracing_kind_t   _buffered_domain,
                    int                                 _operation)
{
    // we loop over all the *registered* contexts and see if any of them, at any point in time,
    // might require callback or buffered API tracing
    for(const auto& itr : _contexts)
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
copy_table(Tp* _orig, std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename hsa_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = hsa_api_info<TableIdx, OpIdx>{};

        LOG(INFO) << "copying table entry for " << _info.name;

        // make sure we don't access a field that doesn't exist in input table
        if(_info.offset() >= _orig->version.minor_id) return;

        // 1. get the sub-table containing the function pointer in original table
        // 2. get reference to function pointer in sub-table in original table
        auto& _table = _info.get_table(_orig);
        auto& _func  = _info.get_table_func(_table);
        // 3. get the sub-table containing the function pointer in saved table
        // 4. get reference to function pointer in sub-table in saved table
        // 5. save the original function in the saved table
        auto& _saved = _info.get_table(hsa_table_lookup<TableIdx>{}());
        auto& _ofunc = _info.get_table_func(_saved);
        _ofunc       = _func;
    }

    (void) _orig;
}

template <size_t TableIdx, typename Tp, size_t OpIdx>
void
update_table(const context::context_array_t& _contexts,
             Tp*                             _orig,
             std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename hsa_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = hsa_api_info<TableIdx, OpIdx>{};

        LOG(INFO) << "updating table entry for " << _info.name;

        // make sure we don't access a field that doesn't exist in input table
        if(_info.offset() >= _orig->version.minor_id) return;

        // check to see if there are any contexts which enable this operation in the ROCTX API
        // domain
        if(!should_wrap_functor(_contexts,
                                _info.callback_domain_idx,
                                _info.buffered_domain_idx,
                                _info.operation_idx))
            return;

        // 1. get the sub-table containing the function pointer in original table
        // 2. get reference to function pointer in sub-table in original table
        // 3. update function pointer with wrapper
        auto& _table = _info.get_table(_orig);
        auto& _func  = _info.get_table_func(_table);
        _func        = _info.get_functor(_func);
    }

    (void) _orig;
}

template <size_t TableIdx, typename Tp, size_t OpIdx, size_t... OpIdxTail>
void
copy_table(Tp* _orig, std::index_sequence<OpIdx, OpIdxTail...>)
{
    copy_table<TableIdx>(_orig, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0)
        copy_table<TableIdx>(_orig, std::index_sequence<OpIdxTail...>{});
}

template <size_t TableIdx, typename Tp, size_t OpIdx, size_t... OpIdxTail>
void
update_table(const context::context_array_t& _contexts,
             Tp*                             _orig,
             std::index_sequence<OpIdx, OpIdxTail...>)
{
    update_table<TableIdx>(_contexts, _orig, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0)
        update_table<TableIdx>(_contexts, _orig, std::index_sequence<OpIdxTail...>{});
}
}  // namespace

// check out the assembly here... this compiles to a switch statement
template <size_t TableIdx>
const char*
name_by_id(uint32_t id)
{
    return name_by_id<TableIdx>(id, std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
uint32_t
id_by_name(const char* name)
{
    return id_by_name<TableIdx>(name, std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
}

template <size_t TableIdx>
std::vector<uint32_t>
get_ids()
{
    constexpr auto last_api_id = hsa_domain_info<TableIdx>::last;
    auto           _data       = std::vector<uint32_t>{};
    _data.reserve(last_api_id);
    get_ids<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
std::vector<const char*>
get_names()
{
    constexpr auto last_api_id = hsa_domain_info<TableIdx>::last;
    auto           _data       = std::vector<const char*>{};
    _data.reserve(last_api_id);
    get_names<TableIdx>(_data, std::make_index_sequence<last_api_id>{});
    return _data;
}

template <size_t TableIdx>
void
iterate_args(uint32_t                                           id,
             const rocprofiler_callback_tracing_hsa_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t   callback,
             void*                                              user_data)
{
    if(callback)
        iterate_args<TableIdx>(id,
                               data,
                               callback,
                               user_data,
                               std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
copy_table(TableT* _orig)
{
    constexpr auto TableIdx = hsa_table_id_lookup<TableT>::value;
    if(_orig)
        copy_table<TableIdx>(_orig, std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
update_table(TableT* _orig)
{
    constexpr auto TableIdx = hsa_table_id_lookup<TableT>::value;
    if(_orig)
    {
        auto _contexts = context::get_registered_contexts();
        update_table<TableIdx>(
            _contexts, _orig, std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
    }
}

using iterate_args_data_t = rocprofiler_callback_tracing_hsa_api_data_t;
using iterate_args_cb_t   = rocprofiler_callback_tracing_operation_args_cb_t;

#define INSTANTIATE_MARKER_TABLE_FUNC(TABLE_TYPE, TABLE_IDX)                                       \
    template void                     copy_table<TABLE_TYPE>(TABLE_TYPE * _tbl);                   \
    template void                     update_table<TABLE_TYPE>(TABLE_TYPE * _tbl);                 \
    template const char*              name_by_id<TABLE_IDX>(uint32_t);                             \
    template uint32_t                 id_by_name<TABLE_IDX>(const char*);                          \
    template std::vector<uint32_t>    get_ids<TABLE_IDX>();                                        \
    template std::vector<const char*> get_names<TABLE_IDX>();                                      \
    template void                     iterate_args<TABLE_IDX>(                                     \
        uint32_t, const iterate_args_data_t&, iterate_args_cb_t, void*);

INSTANTIATE_MARKER_TABLE_FUNC(hsa_core_table_t, ROCPROFILER_HSA_TABLE_ID_Core)
INSTANTIATE_MARKER_TABLE_FUNC(hsa_amd_ext_table_t, ROCPROFILER_HSA_TABLE_ID_AmdExt)
INSTANTIATE_MARKER_TABLE_FUNC(hsa_img_ext_table_t, ROCPROFILER_HSA_TABLE_ID_ImageExt)
INSTANTIATE_MARKER_TABLE_FUNC(hsa_fini_ext_table_t, ROCPROFILER_HSA_TABLE_ID_FinalizeExt)

#undef INSTANTIATE_MARKER_TABLE_FUNC
}  // namespace hsa
}  // namespace rocprofiler
