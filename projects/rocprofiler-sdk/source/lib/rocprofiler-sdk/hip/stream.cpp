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

#include "lib/rocprofiler-sdk/hip/stream.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/static_tl_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hip/utils.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>
#include <rocprofiler-sdk/hip/table_id.h>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <hip/driver_types.h>
#include <hip/hip_runtime_api.h>
// must be included after runtime api
#include <hip/hip_deprecated.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#define ROCPROFILER_LIB_ROCPROFILER_HIP_HIP_CPP_IMPL 1

// template specializations
#include "hip.def.cpp"

namespace rocprofiler
{
namespace hip
{
namespace stream
{
using stream_map_t = std::unordered_map<hipStream_t, rocprofiler_stream_id_t>;

namespace
{
auto*
get_stream_map()
{
    static auto*& _v = common::static_object<common::Synchronized<stream_map_t>>::construct();
    return _v;
}

auto
add_stream(hipStream_t stream)
{
    return get_stream_map()->wlock(
        [](stream_map_t& _data, hipStream_t _stream) {
            static uint64_t idx_offset = 0;

            auto idx = _data.size() + idx_offset;
            ROCP_INFO << fmt::format(
                "hipStream_t={} :: id={}.handle={}{}", static_cast<void*>(_stream), '{', idx, '}');

            ROCP_CI_LOG_IF(WARNING, idx == 0 && _stream != nullptr)
                << "null hip stream does not have index 0";

            if(!_data.emplace(_stream, rocprofiler_stream_id_t{.handle = idx}).second)
            {
                idx_offset += 1;
                auto _existing = _data.at(_stream);
                ROCP_INFO << "existing hipStream_t ("
                          << sdk::utility::as_hex(static_cast<void*>(_stream))
                          << ") reallocated. rocprofiler_stream_id_t{.handle = " << _existing.handle
                          << "} -> rocprofiler_stream_id_t{.handle = " << idx << "}";
                _data.at(_stream) = rocprofiler_stream_id_t{.handle = idx};
            }

            return _data.at(_stream);
        },
        stream);
}

auto
get_stream_id(hipStream_t stream)
{
    return get_stream_map()->rlock(
        [](const stream_map_t& _data, hipStream_t _stream) {
            ROCP_ERROR_IF(_data.count(_stream) == 0)
                << "failed to retrieve stream ID in " << __FILE__;
            return _data.at(_stream);
        },
        stream);
}

// Map rocprofiler_hip_stream_operation_t to respective name
template <size_t OpIdx>
struct hip_stream_operation_name;

#define HIP_STREAM_OPERATION_NAME(ENUM)                                                            \
    template <>                                                                                    \
    struct hip_stream_operation_name<ROCPROFILER_HIP_STREAM_##ENUM>                                \
    {                                                                                              \
        static constexpr auto name          = "HIP_STREAM_" #ENUM;                                 \
        static constexpr auto operation_idx = ROCPROFILER_HIP_STREAM_##ENUM;                       \
    };

HIP_STREAM_OPERATION_NAME(NONE)
HIP_STREAM_OPERATION_NAME(CREATE)
HIP_STREAM_OPERATION_NAME(DESTROY)
HIP_STREAM_OPERATION_NAME(SET)
#undef HIP_STREAM_OPERATION_NAME

template <size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return hip_stream_operation_name<OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

template <size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto _idx = hip_stream_operation_name<OpIdx>::operation_idx;
    if(_idx < ROCPROFILER_HIP_STREAM_LAST) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0) get_ids(_id_list, std::index_sequence<OpIdxTail...>{});
}

}  // namespace

template <size_t TableIdx,
          size_t OpIdx,
          typename RetT,
          typename... Args,
          typename FuncT = RetT (*)(Args...)>
FuncT create_write_functor(RetT (*func)(Args...))
{
    using function_type = FuncT;

    static function_type next_func = func;

    return [](Args... args) -> RetT {
        using function_args_type = common::mpl::type_list<Args...>;

        using callback_api_data_t = rocprofiler_callback_tracing_hip_stream_data_t;

        constexpr auto external_corr_id_domain_idx =
            hip_domain_info<TableIdx>::external_correlation_id_domain_idx;

        auto thr_id            = common::get_tid();
        auto callback_contexts = tracing::callback_context_data_vec_t{};
        auto buffered_contexts = tracing::buffered_context_data_vec_t{};
        auto external_corr_ids = tracing::external_correlation_id_map_t{};

        tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                   ROCPROFILER_BUFFER_TRACING_HIP_STREAM,
                                   callback_contexts,
                                   buffered_contexts,
                                   external_corr_ids);
        assert(buffered_contexts.empty() && "Stream tracing should not have any buffered contexts");

        auto internal_corr_id = 0;
        auto ancestor_corr_id = 0;
        auto tracer_data      = common::init_public_api_struct(callback_api_data_t{},
                                                          rocprofiler_stream_id_t{.handle = 0},
                                                          rocprofiler_address_t{.value = 0});

        constexpr auto stream_idx = common::mpl::index_of<hipStream_t*, function_args_type>::value;
        auto           stream = std::get<stream_idx>(std::make_tuple(std::forward<Args>(args)...));

        tracing::update_external_correlation_ids(
            external_corr_ids, thr_id, external_corr_id_domain_idx);

        auto _ret = next_func(std::forward<Args>(args)...);
        if(!callback_contexts.empty())
        {
            if(stream)
            {
                tracer_data.stream_id        = add_stream(*stream);
                tracer_data.stream_value.ptr = *stream;
            }
            tracing::execute_phase_none_callbacks(callback_contexts,
                                                  thr_id,
                                                  internal_corr_id,
                                                  external_corr_ids,
                                                  ancestor_corr_id,
                                                  ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                                  ROCPROFILER_HIP_STREAM_CREATE,
                                                  tracer_data);
        }

        if constexpr(!std::is_void<RetT>::value) return _ret;
    };
}

template <size_t TableIdx,
          size_t OpIdx,
          typename RetT,
          typename... Args,
          typename FuncT = RetT (*)(Args...)>
FuncT create_destroy_functor(RetT (*func)(Args...))
{
    using function_type = FuncT;

    static function_type next_func = func;

    return [](Args... args) -> RetT {
        using function_args_type  = common::mpl::type_list<Args...>;
        constexpr auto stream_idx = common::mpl::index_of<hipStream_t, function_args_type>::value;

        using callback_api_data_t = rocprofiler_callback_tracing_hip_stream_data_t;

        constexpr auto external_corr_id_domain_idx =
            hip_domain_info<TableIdx>::external_correlation_id_domain_idx;

        auto thr_id            = common::get_tid();
        auto callback_contexts = tracing::callback_context_data_vec_t{};
        auto buffered_contexts = tracing::buffered_context_data_vec_t{};
        auto external_corr_ids = tracing::external_correlation_id_map_t{};

        tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                   ROCPROFILER_BUFFER_TRACING_HIP_STREAM,
                                   callback_contexts,
                                   buffered_contexts,
                                   external_corr_ids);
        assert(buffered_contexts.empty() && "Stream tracing should not have any buffered contexts");

        auto internal_corr_id = 0;
        auto ancestor_corr_id = 0;
        auto tracer_data      = common::init_public_api_struct(callback_api_data_t{},
                                                          rocprofiler_stream_id_t{.handle = 0},
                                                          rocprofiler_address_t{.value = 0});

        auto stream = std::get<stream_idx>(std::make_tuple(std::forward<Args>(args)...));

        tracing::update_external_correlation_ids(
            external_corr_ids, thr_id, external_corr_id_domain_idx);

        auto _ret = next_func(std::forward<Args>(args)...);

        if(!callback_contexts.empty())
        {
            tracer_data.stream_id        = get_stream_id(stream);
            tracer_data.stream_value.ptr = stream;
            tracing::execute_phase_none_callbacks(callback_contexts,
                                                  thr_id,
                                                  internal_corr_id,
                                                  external_corr_ids,
                                                  ancestor_corr_id,
                                                  ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                                  ROCPROFILER_HIP_STREAM_DESTROY,
                                                  tracer_data);
        }

        if constexpr(!std::is_void<RetT>::value) return _ret;
    };
}

template <size_t TableIdx,
          size_t OpIdx,
          typename RetT,
          typename... Args,
          typename FuncT = RetT (*)(Args...)>
FuncT create_read_functor(RetT (*func)(Args...))
{
    using function_type = FuncT;

    static function_type next_func = func;

    return [](Args... args) -> RetT {
        using function_args_type  = common::mpl::type_list<Args...>;
        constexpr auto stream_idx = common::mpl::index_of<hipStream_t, function_args_type>::value;

        using callback_api_data_t = rocprofiler_callback_tracing_hip_stream_data_t;

        constexpr auto external_corr_id_domain_idx =
            hip_domain_info<TableIdx>::external_correlation_id_domain_idx;

        auto thr_id            = common::get_tid();
        auto callback_contexts = tracing::callback_context_data_vec_t{};
        auto buffered_contexts = tracing::buffered_context_data_vec_t{};
        auto external_corr_ids = tracing::external_correlation_id_map_t{};

        tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                   ROCPROFILER_BUFFER_TRACING_HIP_STREAM,
                                   callback_contexts,
                                   buffered_contexts,
                                   external_corr_ids);

        assert(buffered_contexts.empty() && "Stream tracing should not have any buffered contexts");

        auto internal_corr_id = 0;
        auto ancestor_corr_id = 0;
        auto tracer_data      = common::init_public_api_struct(callback_api_data_t{},
                                                          rocprofiler_stream_id_t{.handle = 0},
                                                          rocprofiler_address_t{.value = 0});

        auto stream = std::get<stream_idx>(std::make_tuple(std::forward<Args>(args)...));

        if(!callback_contexts.empty())
        {
            tracer_data.stream_id        = get_stream_id(stream);
            tracer_data.stream_value.ptr = stream;
            tracing::execute_phase_enter_callbacks(callback_contexts,
                                                   thr_id,
                                                   internal_corr_id,
                                                   external_corr_ids,
                                                   ancestor_corr_id,
                                                   ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                                   ROCPROFILER_HIP_STREAM_SET,
                                                   tracer_data);
        }

        tracing::update_external_correlation_ids(
            external_corr_ids, thr_id, external_corr_id_domain_idx);

        auto _ret = next_func(std::forward<Args>(args)...);

        if(!callback_contexts.empty())
        {
            tracing::execute_phase_exit_callbacks(callback_contexts,
                                                  external_corr_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                                  ROCPROFILER_HIP_STREAM_SET,
                                                  tracer_data);
        }

        if constexpr(!std::is_void<RetT>::value) return _ret;
    };
}
}  // namespace stream
}  // namespace hip
}  // namespace rocprofiler

namespace rocprofiler
{
namespace hip
{
namespace stream
{
namespace
{
bool
enable_stream_stack()
{
    if(hsa::enable_queue_intercept()) return true;

    for(const auto& itr : context::get_registered_contexts())
    {
        if(itr->is_tracing_one_of(ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                  ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                  ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                  ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                  ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                  ROCPROFILER_BUFFER_TRACING_HIP_STREAM,
                                  ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT))
            return true;
    }

    return false;
}

template <size_t TableIdx, typename Tp, size_t OpIdx>
void
update_table(Tp* _orig, std::integral_constant<size_t, OpIdx>)
{
    using table_type         = typename hip_table_lookup<TableIdx>::type;
    using info_type          = hip_api_info<TableIdx, OpIdx>;
    using function_args_type = decltype(info_type::get_args_type());

    static_assert(info_type::table_idx == ROCPROFILER_HIP_TABLE_ID_Runtime,
                  "This function should only be instantiated for HIP runtime API");

    if constexpr(std::is_same<table_type, Tp>::value &&
                 (common::mpl::is_one_of<hipStream_t, function_args_type>::value ||
                  common::mpl::is_one_of<hipStream_t*, function_args_type>::value))
    {
        auto _info = info_type{};

        // make sure we don't access a field that doesn't exist in input table
        if(_info.offset() >= _orig->size) return;

        ROCP_TRACE << fmt::format("[hip stream] checking table entry for {}", _info.name);

        constexpr auto num_args = function_args_type::size();

        if constexpr(common::mpl::is_one_of<hipStream_t, function_args_type>::value)
        {
            constexpr auto stream_idx =
                common::mpl::index_of<hipStream_t, function_args_type>::value;
            constexpr auto rstream_idx =
                common::mpl::index_of<hipStream_t, common::mpl::reverse<function_args_type>>::value;
            constexpr auto is_hip_stream_destroy_func =
                info_type::table_idx == ROCPROFILER_HIP_TABLE_ID_Runtime &&
                info_type::operation_idx == ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamDestroy;

            // index_of finds the first argument of that type. So find the first and last
            // arg of the given type and make sure it resolves to the same distance
            static_assert(stream_idx == (num_args - rstream_idx - 1),
                          "function has more than one stream argument");

            // 1. get the sub-table containing the function pointer in original table
            // 2. get reference to function pointer in sub-table in original table
            // 3. update function pointer with wrapper
            auto& _table = _info.get_table(_orig);
            auto& _func  = _info.get_table_func(_table);

            if constexpr(is_hip_stream_destroy_func)
            {
                ROCP_INFO << fmt::format(
                    "[hip stream] {} has been designated as a stream destroy function", _info.name);
                _func = create_destroy_functor<TableIdx, OpIdx>(_func);
            }
            else
            {
                ROCP_INFO << fmt::format(
                    "[hip stream] {} has been designated as a stream set function", _info.name);
                _func = create_read_functor<TableIdx, OpIdx>(_func);
            }
        }
        else if constexpr(common::mpl::is_one_of<hipStream_t*, function_args_type>::value)
        {
            constexpr auto stream_idx =
                common::mpl::index_of<hipStream_t*, function_args_type>::value;
            constexpr auto rstream_idx =
                common::mpl::index_of<hipStream_t*,
                                      common::mpl::reverse<function_args_type>>::value;

            // index_of finds the first argument of that type. So find the first and last
            // arg of the given type and make sure it resolves to the same distance
            static_assert(stream_idx == (num_args - rstream_idx - 1),
                          "function has more than one stream argument");

            ROCP_INFO << fmt::format(
                "[hip stream] {} has been designated as a stream create function", _info.name);

            // 1. get the sub-table containing the function pointer in original table
            // 2. get reference to function pointer in sub-table in original table
            // 3. update function pointer with wrapper
            auto& _table = _info.get_table(_orig);
            auto& _func  = _info.get_table_func(_table);
            _func        = create_write_functor<TableIdx, OpIdx>(_func);
        }
    }

    // suppress unused-but-set-parameter warning
    common::consume_args(_orig);
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

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_HIP_STREAM_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    constexpr auto last_id = ROCPROFILER_HIP_STREAM_LAST;
    auto           _data   = std::vector<uint32_t>{};
    _data.reserve(last_id);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_HIP_STREAM_LAST>{});
    return _data;
}

template <typename TableT>
void
update_table(TableT* _orig)
{
    add_stream(nullptr);

    if(!enable_stream_stack()) return;

    constexpr auto TableIdx = hip_table_id_lookup<TableT>::value;
    if(_orig)
        update_table<TableIdx>(_orig, std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
}

using hip_api_data_t   = rocprofiler_callback_tracing_hip_api_data_t;
using hip_op_args_cb_t = rocprofiler_callback_tracing_operation_args_cb_t;

#define INSTANTIATE_HIP_TABLE_FUNC(TABLE_TYPE, TABLE_IDX)                                          \
    template void update_table<TABLE_TYPE>(TABLE_TYPE * _tbl);

INSTANTIATE_HIP_TABLE_FUNC(hip_runtime_api_table_t, ROCPROFILER_HIP_TABLE_ID_Runtime)
}  // namespace stream
}  // namespace hip
}  // namespace rocprofiler
