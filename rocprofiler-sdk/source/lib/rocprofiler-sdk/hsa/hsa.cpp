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

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/details/ostream.hpp"
#include "lib/rocprofiler-sdk/hsa/pc_sampling.hpp"
#include "lib/rocprofiler-sdk/hsa/scratch_memory.hpp"
#include "lib/rocprofiler-sdk/hsa/utils.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa/api_id.h>
#include <rocprofiler-sdk/hsa/core_api_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>

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
DEFINE_TABLE_VERSION(amd_tool, TOOLS_API)

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
DEFINE_TABLE_VERSION(pc_sampling_ext, PC_SAMPLING_API)
#endif

#undef DEFINE_TABLE_VERSION
#undef DEFINE_TABLE_VERSION_IMPL

// helper to ensure that table type is paired with the correct table version
#define GET_TABLE_IMPL(ALIAS, TYPE)                                                                \
    get_table_impl<hsa_##ALIAS##_table_t, TYPE>(hsa_##ALIAS##_table_version);

template <typename Tp, typename TableT>
Tp*&
get_table_impl(hsa_table_version_t _version)
{
    static_assert(common::static_object<Tp, TableT>::is_trivial_standard_layout(),
                  "This HSA API table is not a trivial, standard layout type as it should be");

    auto*& val   = common::static_object<Tp, TableT>::construct();
    val->version = _version;
    return val;
}
}  // namespace

hsa_core_table_t*
get_tracing_core_table()
{
    static auto*& val = GET_TABLE_IMPL(core, tracing_table);
    return val;
}

hsa_amd_ext_table_t*
get_tracing_amd_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(amd_ext, tracing_table);
    return val;
}

hsa_fini_ext_table_t*
get_tracing_fini_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(fini_ext, tracing_table);
    return val;
}

hsa_img_ext_table_t*
get_tracing_img_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(img_ext, tracing_table);
    return val;
}

hsa_amd_tool_table_t*
get_tracing_amd_tool_table()  // table is never traced
{
    static auto*& val = GET_TABLE_IMPL(amd_tool, tracing_table);
    return val;
}

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

hsa_pc_sampling_ext_table_t*
get_tracing_pc_sampling_ext_table()  // table is never traced
{
    static auto*& val = GET_TABLE_IMPL(pc_sampling_ext, tracing_table);
    return val;
}

#endif

hsa_table_version_t
get_table_version()
{
    return hsa_api_table_version;
}

hsa_core_table_t*
get_core_table()
{
    static auto*& val = GET_TABLE_IMPL(core, internal_table);
    return val;
}

hsa_amd_ext_table_t*
get_amd_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(amd_ext, internal_table);
    return val;
}

hsa_fini_ext_table_t*
get_fini_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(fini_ext, internal_table);
    return val;
}

hsa_img_ext_table_t*
get_img_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(img_ext, internal_table);
    return val;
}

hsa_amd_tool_table_t*
get_amd_tool_table()
{
    static auto*& val = GET_TABLE_IMPL(amd_tool, internal_table);
    return val;
}

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

hsa_pc_sampling_ext_table_t*
get_pc_sampling_ext_table()
{
    static auto*& val = GET_TABLE_IMPL(pc_sampling_ext, internal_table);
    return val;
}

#endif

#undef GET_TABLE_IMPL

hsa_api_table_t&
get_table()
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    static auto tbl = hsa_api_table_t{.version          = hsa_api_table_version,
                                      .core_            = get_core_table(),
                                      .amd_ext_         = get_amd_ext_table(),
                                      .finalizer_ext_   = get_fini_ext_table(),
                                      .image_ext_       = get_img_ext_table(),
                                      .tools_           = get_amd_tool_table(),
                                      .pc_sampling_ext_ = get_pc_sampling_ext_table()};
#else
    static auto tbl = hsa_api_table_t{.version        = hsa_api_table_version,
                                      .core_          = get_core_table(),
                                      .amd_ext_       = get_amd_ext_table(),
                                      .finalizer_ext_ = get_fini_ext_table(),
                                      .image_ext_     = get_img_ext_table(),
                                      .tools_         = get_amd_tool_table()};
#endif
    return tbl;
}

template <size_t TableIdx, size_t OpIdx>
template <typename DataArgsT, typename... Args>
auto
hsa_api_impl<TableIdx, OpIdx>::set_data_args(DataArgsT& _data_args, Args... args)
{
    if constexpr(sizeof...(Args) == 0)
        _data_args.no_args.empty = '\0';
    else
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

template <size_t TableIdx, size_t OpIdx>
template <typename RetT, typename... Args>
RetT
hsa_api_impl<TableIdx, OpIdx>::functor(Args... args)
{
    using buffer_hsa_api_record_t = rocprofiler_buffer_tracing_hsa_api_record_t;
    using callback_hsa_api_data_t = rocprofiler_callback_tracing_hsa_api_data_t;
    using info_type               = hsa_api_info<TableIdx, OpIdx>;

    constexpr auto external_corr_id_domain_idx =
        hsa_domain_info<TableIdx>::external_correlation_id_domain_idx;

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
    auto           external_corr_ids = tracing::external_correlation_id_map_t{};

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

    auto  buffer_record    = common::init_public_api_struct(buffer_hsa_api_record_t{});
    auto  tracer_data      = common::init_public_api_struct(callback_hsa_api_data_t{});
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
        set_data_retval(tracer_data.retval, _ret);

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

    // decrement the reference count after usage in the callback/buffers
    corr_id->sub_ref_count();

    context::pop_latest_correlation_id(corr_id);

    if constexpr(!std::is_void<RetT>::value) return _ret;
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
             int32_t                                            max_deref,
             void*                                              user_data,
             std::index_sequence<OpIdx, IdxTail...>)
{
    if(OpIdx == id)
    {
        using info_type = hsa_api_info<TableIdx, OpIdx>;
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
    if constexpr(sizeof...(IdxTail) > 0)
        iterate_args<TableIdx>(
            id, data, func, max_deref, user_data, std::index_sequence<IdxTail...>{});
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

auto hsa_reference_count_value = std::atomic<int>{0};

hsa_status_t
hsa_init_refcnt_impl()
{
    struct scoped_dtor
    {
        scoped_dtor() = default;
        ~scoped_dtor() { ++hsa_reference_count_value; }
    };
    auto _dtor = scoped_dtor{};
    return get_core_table()->hsa_init_fn();
}

hsa_status_t
hsa_shut_down_refcnt_impl()
{
    if(hsa_reference_count_value > 0)
    {
        --hsa_reference_count_value;
        return get_core_table()->hsa_shut_down_fn();
    }
    return HSA_STATUS_SUCCESS;
}

template <size_t TableIdx, typename LookupT = internal_table, typename Tp, size_t OpIdx>
void
copy_table(Tp* _orig, uint64_t _tbl_instance, std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename hsa_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _info = hsa_api_info<TableIdx, OpIdx>{};

        // make sure we don't access a field that doesn't exist in input table
        if(_info.offset() >= _orig->version.minor_id) return;

        // 1. get the sub-table containing the function pointer in original table
        // 2. get reference to function pointer in sub-table in original table
        auto& _orig_table = _info.get_table(_orig);
        auto& _orig_func  = _info.get_table_func(_orig_table);
        // 3. get the sub-table containing the function pointer in saved table
        // 4. get reference to function pointer in sub-table in saved table
        // 5. save the original function in the saved table
        auto& _copy_table = _info.get_table(hsa_table_lookup<TableIdx>{}(LookupT{}));
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

        if constexpr(TableIdx == ROCPROFILER_HSA_TABLE_ID_Core &&
                     OpIdx == ROCPROFILER_HSA_CORE_API_ID_hsa_init)
        {
            auto& _func = _info.get_table_func(_info.get_table(_orig));
            _func       = hsa_init_refcnt_impl;
            if(get_hsa_ref_count() == 0) ++hsa_reference_count_value;
        }
        else if constexpr(TableIdx == ROCPROFILER_HSA_TABLE_ID_Core &&
                          OpIdx == ROCPROFILER_HSA_CORE_API_ID_hsa_shut_down)
        {
            auto& _func = _info.get_table_func(_info.get_table(_orig));
            _func       = hsa_shut_down_refcnt_impl;
        }
    }
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

        // make sure we don't access a field that doesn't exist in input table
        if(_info.offset() >= _orig->version.minor_id) return;

        // check to see if there are any contexts which enable this operation in the ROCTx API
        // domain
        if(!should_wrap_functor(_contexts,
                                _info.callback_domain_idx,
                                _info.buffered_domain_idx,
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

template <size_t TableIdx,
          typename LookupT = internal_table,
          typename Tp,
          size_t OpIdx,
          size_t... OpIdxTail>
void
copy_table(Tp* _orig, uint64_t _tbl_instance, std::index_sequence<OpIdx, OpIdxTail...>)
{
    copy_table<TableIdx, LookupT>(_orig, _tbl_instance, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0)
        copy_table<TableIdx, LookupT>(_orig, _tbl_instance, std::index_sequence<OpIdxTail...>{});
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

void
check_hsa_timing_functions_impl()
{
    CHECK(hsa::get_core_table() != nullptr);
    CHECK(hsa::get_core_table()->hsa_system_get_info_fn != nullptr)
        << "missing non-null function pointer to hsa_system_get_info_fn";
    CHECK(hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn != nullptr)
        << "missing non-null function pointer to hsa_amd_profiling_get_dispatch_time";
    CHECK(hsa::get_amd_ext_table()->hsa_amd_profiling_get_async_copy_time_fn != nullptr)
        << "missing non-null function pointer to hsa_amd_profiling_get_async_copy_time";
}

void
check_hsa_timing_functions()
{
    static auto _once = std::once_flag{};
    std::call_once(_once, check_hsa_timing_functions_impl);
}
}  // namespace

std::string_view
get_hsa_status_string(hsa_status_t _status)
{
    const char* _status_msg = nullptr;
    return (hsa::get_core_table()->hsa_status_string_fn(_status, &_status_msg) ==
                HSA_STATUS_SUCCESS &&
            _status_msg)
               ? std::string_view{_status_msg}
               : std::string_view{"(unknown HSA error)"};
}

uint64_t
get_hsa_timestamp_period()
{
    check_hsa_timing_functions();

    constexpr auto nanosec     = 1000000000UL;
    uint64_t       sysclock_hz = 0;
    ROCP_HSA_TABLE_CALL(ERROR,
                        hsa::get_core_table()->hsa_system_get_info_fn(
                            HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz));
    return (nanosec / sysclock_hz);
}

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
             int32_t                                            max_deref,
             void*                                              user_data)
{
    if(callback)
        iterate_args<TableIdx>(id,
                               data,
                               callback,
                               max_deref,
                               user_data,
                               std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
copy_table(TableT* _orig, uint64_t _tbl_instance)
{
    constexpr auto TableIdx = hsa_table_id_lookup<TableT>::value;
    if(_orig)
        copy_table<TableIdx, internal_table>(
            _orig, _tbl_instance, std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
}

template <typename TableT>
void
update_table(TableT* _orig, uint64_t _tbl_instance)
{
    constexpr auto TableIdx = hsa_table_id_lookup<TableT>::value;
    if(_orig)
    {
        copy_table<TableIdx, tracing_table>(
            _orig, _tbl_instance, std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});

        auto _contexts = context::get_registered_contexts();
        update_table<TableIdx>(
            _contexts, _orig, std::make_index_sequence<hsa_domain_info<TableIdx>::last>{});
    }
}

using iterate_args_data_t = rocprofiler_callback_tracing_hsa_api_data_t;
using iterate_args_cb_t   = rocprofiler_callback_tracing_operation_args_cb_t;

#define INSTANTIATE_HSA_TABLE_FUNC(TABLE_TYPE, TABLE_IDX)                                           \
    template void                     copy_table<TABLE_TYPE>(TABLE_TYPE * _tbl, uint64_t _instv);   \
    template void                     update_table<TABLE_TYPE>(TABLE_TYPE * _tbl, uint64_t _instv); \
    template const char*              name_by_id<TABLE_IDX>(uint32_t);                              \
    template uint32_t                 id_by_name<TABLE_IDX>(const char*);                           \
    template std::vector<uint32_t>    get_ids<TABLE_IDX>();                                         \
    template std::vector<const char*> get_names<TABLE_IDX>();                                       \
    template void                     iterate_args<TABLE_IDX>(                                      \
        uint32_t, const iterate_args_data_t&, iterate_args_cb_t, int32_t, void*);

INSTANTIATE_HSA_TABLE_FUNC(hsa_core_table_t, ROCPROFILER_HSA_TABLE_ID_Core)
INSTANTIATE_HSA_TABLE_FUNC(hsa_amd_ext_table_t, ROCPROFILER_HSA_TABLE_ID_AmdExt)
INSTANTIATE_HSA_TABLE_FUNC(hsa_img_ext_table_t, ROCPROFILER_HSA_TABLE_ID_ImageExt)
INSTANTIATE_HSA_TABLE_FUNC(hsa_fini_ext_table_t, ROCPROFILER_HSA_TABLE_ID_FinalizeExt)

template <>
void
copy_table<hsa_amd_tool_table_t>(hsa_amd_tool_table_t* _tbl, uint64_t _instv)
{
    scratch_memory::copy_table(_tbl, _instv);
}

template <>
void
update_table<hsa_amd_tool_table_t>(hsa_amd_tool_table_t* _tbl, uint64_t _instv)
{
    scratch_memory::update_table(_tbl, _instv);
}

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

template <>
void
copy_table<hsa_pc_sampling_ext_table_t>(hsa_pc_sampling_ext_table_t* _tbl, uint64_t _instv)
{
    pc_sampling::copy_table(_tbl, _instv);
}

#endif
#undef INSTANTIATE_HSA_TABLE_FUNC

int
get_hsa_ref_count()
{
    auto _val = hsa_reference_count_value.load();
    ROCP_TRACE << "hsa reference count: " << _val;
    return _val;
}
}  // namespace hsa
}  // namespace rocprofiler
