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

#include "lib/rocprofiler-sdk/hsa/memory_allocation.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa/api_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>

#include <glog/logging.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <chrono>
#include <cstdlib>
#include <type_traits>

#define ROCPROFILER_LIB_ROCPROFILER_HSA_MEMORY_ALLOCATION_CPP_IMPL 1

// template specializations
#include "hsa.def.cpp"

namespace rocprofiler
{
namespace hsa
{
namespace memory_allocation
{
namespace
{
using context_t                = context::context;
using external_corr_id_map_t   = std::unordered_map<const context_t*, rocprofiler_user_data_t>;
using region_to_agent_map      = std::unordered_map<hsa_region_t, rocprofiler_agent_id_t>;
using memory_pool_to_agent_map = std::unordered_map<hsa_amd_memory_pool_t, rocprofiler_agent_id_t>;
using region_to_agent_pair     = std::pair<region_to_agent_map*, rocprofiler_agent_id_t>;
using map_pool_to_agent_pair   = std::pair<memory_pool_to_agent_map*, rocprofiler_agent_id_t>;

template <size_t TableIdx, size_t OpIdx, typename... Args>
hsa_status_t
memory_allocation_impl(Args... args);

template <size_t TableIdx, size_t OpIdx, typename... Args>
hsa_status_t
memory_free_impl(Args... args);

// Local enum to specify implementation of memory function wrappers
typedef enum
{
    HSA_NONE = 0,                  ///< Unknown memory allocation function
    HSA_MEMORY_ALLOCATE,           ///< Allocate memory function
    HSA_AMD_MEMORY_POOL_ALLOCATE,  ///< Allocate memory pool
    HSA_AMD_VMEM_ALLOCATE,         ///< Allocate vmem memory handle
    HSA_MEMORY_FREE,               ///< Free memory function
    HSA_AMD_MEMORY_POOL_FREE,      ///< Free memory pool
    HSA_AMD_VMEM_FREE,             ///< Release vmem memory handle
    HSA_LAST,
} hsa_memory_operation_functions_t;

// Set up information to identify agent from regions/pool
template <size_t OpIdx>
struct memory_allocation_info;

#define SPECIALIZE_MEMORY_ALLOCATION_INFO(                                                         \
    FUNCTION, ENUM, MAPTYPE, PAIRTYPE, SEARCHTYPE, ITERATEFUNC, IMPLEMENTATION)                    \
    template <>                                                                                    \
    struct memory_allocation_info<FUNCTION>                                                        \
    {                                                                                              \
        using maptype    = MAPTYPE;                                                                \
        using pairtype   = PAIRTYPE;                                                               \
        using searchtype = SEARCHTYPE;                                                             \
        auto&                 operator()() const { return ITERATEFUNC; }                           \
        static constexpr auto operation_idx = ROCPROFILER_MEMORY_ALLOCATION_##ENUM;                \
                                                                                                   \
        template <size_t TableIdx, size_t OpIdx, typename RetT, typename... Args>                  \
        static auto get_memory_allocation_impl(RetT (*)(Args...))                                  \
        {                                                                                          \
            return &IMPLEMENTATION<TableIdx, OpIdx, Args...>;                                      \
        }                                                                                          \
    };

SPECIALIZE_MEMORY_ALLOCATION_INFO(HSA_MEMORY_ALLOCATE,
                                  ALLOCATE,
                                  region_to_agent_map,
                                  region_to_agent_pair,
                                  hsa_region_t,
                                  get_core_table()->hsa_agent_iterate_regions_fn,
                                  memory_allocation_impl)
SPECIALIZE_MEMORY_ALLOCATION_INFO(HSA_AMD_MEMORY_POOL_ALLOCATE,
                                  ALLOCATE,
                                  memory_pool_to_agent_map,
                                  map_pool_to_agent_pair,
                                  hsa_amd_memory_pool_t,
                                  get_amd_ext_table()->hsa_amd_agent_iterate_memory_pools_fn,
                                  memory_allocation_impl)
SPECIALIZE_MEMORY_ALLOCATION_INFO(HSA_AMD_VMEM_ALLOCATE,
                                  VMEM_ALLOCATE,
                                  memory_pool_to_agent_map,
                                  map_pool_to_agent_pair,
                                  hsa_amd_memory_pool_t,
                                  get_amd_ext_table()->hsa_amd_agent_iterate_memory_pools_fn,
                                  memory_allocation_impl)
SPECIALIZE_MEMORY_ALLOCATION_INFO(HSA_MEMORY_FREE,
                                  FREE,
                                  region_to_agent_map,
                                  region_to_agent_pair,
                                  hsa_region_t,
                                  get_core_table()->hsa_agent_iterate_regions_fn,
                                  memory_free_impl)
SPECIALIZE_MEMORY_ALLOCATION_INFO(HSA_AMD_MEMORY_POOL_FREE,
                                  FREE,
                                  memory_pool_to_agent_map,
                                  map_pool_to_agent_pair,
                                  hsa_amd_memory_pool_t,
                                  get_amd_ext_table()->hsa_amd_agent_iterate_memory_pools_fn,
                                  memory_free_impl)
SPECIALIZE_MEMORY_ALLOCATION_INFO(HSA_AMD_VMEM_FREE,
                                  VMEM_FREE,
                                  memory_pool_to_agent_map,
                                  map_pool_to_agent_pair,
                                  hsa_amd_memory_pool_t,
                                  get_amd_ext_table()->hsa_amd_agent_iterate_memory_pools_fn,
                                  memory_free_impl)
#undef SPECIALIZE_MEMORY_ALLOCATION_INFO

// Map rocprofiler_memory_allocation_operation_t to respective name
template <size_t OpIdx>
struct memory_allocation_name;

#define MEMORY_ALLOCATION_NAME(ENUM)                                                               \
    template <>                                                                                    \
    struct memory_allocation_name<ROCPROFILER_MEMORY_ALLOCATION_##ENUM>                            \
    {                                                                                              \
        static constexpr auto name          = "MEMORY_ALLOCATION_" #ENUM;                          \
        static constexpr auto operation_idx = ROCPROFILER_MEMORY_ALLOCATION_##ENUM;                \
    };

MEMORY_ALLOCATION_NAME(NONE)
MEMORY_ALLOCATION_NAME(ALLOCATE)
MEMORY_ALLOCATION_NAME(VMEM_ALLOCATE)
MEMORY_ALLOCATION_NAME(FREE)
MEMORY_ALLOCATION_NAME(VMEM_FREE)
#undef MEMORY_ALLOCATION_NAME

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return memory_allocation_name<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Idx, size_t... IdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<Idx, IdxTail...>)
{
    if(std::string_view{memory_allocation_name<Idx>::name} == std::string_view{name})
        return memory_allocation_name<Idx>::operation_idx;
    if constexpr(sizeof...(IdxTail) > 0)
        return id_by_name(name, std::index_sequence<IdxTail...>{});
    else
        return ROCPROFILER_MEMORY_ALLOCATION_LAST;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_MEMORY_ALLOCATION_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, memory_allocation_name<Idx>::operation_idx), ...);
}

template <size_t... Idx>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, const char* _v) {
        if(_v != nullptr && strnlen(_v, 1) > 0) _vec.emplace_back(_v);
    };

    (_emplace(_name_list, memory_allocation_name<Idx>::name), ...);
}

bool
context_filter(const context::context* ctx)
{
    auto has_buffered =
        (ctx->buffered_tracer &&
         (ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)));

    auto has_callback =
        (ctx->callback_tracer &&
         (ctx->callback_tracer->domains(ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION)));

    return (has_buffered || has_callback);
}

// Sequence of HSA functions being tracked. Add to these to trace new commands
enum memory_allocation_core_id
{
    memory_allocation_core_allocate_id = ROCPROFILER_HSA_CORE_API_ID_hsa_memory_allocate,
    memory_allocation_core_free_id     = ROCPROFILER_HSA_CORE_API_ID_hsa_memory_free,
};
using memory_allocation_core_index_seq_t =
    std::index_sequence<memory_allocation_core_allocate_id, memory_allocation_core_free_id>;

enum memory_allocation_amd_ext_id
{
    memory_allocation_amd_ext_allocate_id =
        ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_allocate,
    memory_allocation_vmem_allocate_id = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_handle_create,
    memory_allocation_amd_ext_free_id  = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_free,
    memory_allocation_vmem_release_id  = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_handle_release,
};
using memory_allocation_amd_ext_index_seq_t =
    std::index_sequence<memory_allocation_amd_ext_allocate_id,
                        memory_allocation_vmem_allocate_id,
                        memory_allocation_amd_ext_free_id,
                        memory_allocation_vmem_release_id>;

template <size_t TableIdx>
struct memory_allocation_seq;

#define MEMORY_ALLOCATION_DEFINE_SEQ(TABLE_TYPE, SEQ)                                              \
    template <>                                                                                    \
    struct memory_allocation_seq<TABLE_TYPE>                                                       \
    {                                                                                              \
        static constexpr auto memory_allocation_index_seq_t = SEQ{};                               \
    };

MEMORY_ALLOCATION_DEFINE_SEQ(ROCPROFILER_HSA_TABLE_ID_Core, memory_allocation_core_index_seq_t)
MEMORY_ALLOCATION_DEFINE_SEQ(ROCPROFILER_HSA_TABLE_ID_AmdExt, memory_allocation_amd_ext_index_seq_t)

// Set argument indices for tracked functions
template <size_t Idx>
struct arg_indices;

#define HSA_MEMORY_ALLOCATE_DEFINE_ARG_INDICES(                                                    \
    ENUM_ID, STARTING_ADDRESS_IDX, SIZE_IDX, REGION_IDX)                                           \
    template <>                                                                                    \
    struct arg_indices<ENUM_ID>                                                                    \
    {                                                                                              \
        static constexpr auto address_idx = STARTING_ADDRESS_IDX;                                  \
        static constexpr auto size_idx    = SIZE_IDX;                                              \
        static constexpr auto region_idx  = REGION_IDX;                                            \
    };

HSA_MEMORY_ALLOCATE_DEFINE_ARG_INDICES(memory_allocation_core_allocate_id, 2, 1, 0)
HSA_MEMORY_ALLOCATE_DEFINE_ARG_INDICES(memory_allocation_amd_ext_allocate_id, 3, 1, 0)
HSA_MEMORY_ALLOCATE_DEFINE_ARG_INDICES(memory_allocation_vmem_allocate_id, 4, 1, 0)
HSA_MEMORY_ALLOCATE_DEFINE_ARG_INDICES(memory_allocation_core_free_id, 0, 0, 0)
HSA_MEMORY_ALLOCATE_DEFINE_ARG_INDICES(memory_allocation_amd_ext_free_id, 0, 0, 0)
HSA_MEMORY_ALLOCATE_DEFINE_ARG_INDICES(memory_allocation_vmem_release_id, 0, 0, 0)

// Define operation indices for each tracked functions
template <size_t Idx>
struct memory_allocation_op;

#define MEMORY_ALLOCATE_OPERATION_IDX(ENUM_ID, FUNCTION)                                           \
    template <>                                                                                    \
    struct memory_allocation_op<ENUM_ID>                                                           \
    {                                                                                              \
        static constexpr auto operation_idx = FUNCTION;                                            \
    };

MEMORY_ALLOCATE_OPERATION_IDX(memory_allocation_core_allocate_id, HSA_MEMORY_ALLOCATE);
MEMORY_ALLOCATE_OPERATION_IDX(memory_allocation_amd_ext_allocate_id, HSA_AMD_MEMORY_POOL_ALLOCATE);
MEMORY_ALLOCATE_OPERATION_IDX(memory_allocation_vmem_allocate_id, HSA_AMD_VMEM_ALLOCATE)
MEMORY_ALLOCATE_OPERATION_IDX(memory_allocation_core_free_id, HSA_MEMORY_FREE);
MEMORY_ALLOCATE_OPERATION_IDX(memory_allocation_amd_ext_free_id, HSA_AMD_MEMORY_POOL_FREE);
MEMORY_ALLOCATE_OPERATION_IDX(memory_allocation_vmem_release_id, HSA_AMD_VMEM_FREE);

template <typename FuncT, typename ArgsT, size_t... Idx>
decltype(auto)
invoke(FuncT&& _func, ArgsT&& _args, std::index_sequence<Idx...>)
{
    return std::forward<FuncT>(_func)(std::get<Idx>(_args)...);
}

template <size_t TableIdx, size_t OpIdx>
auto&
get_next_dispatch()
{
    using function_t     = typename hsa_api_meta<TableIdx, OpIdx>::function_type;
    static function_t _v = nullptr;
    return _v;
}

constexpr auto null_rocp_agent_id = rocprofiler_agent_id_t{.handle = 0};

struct memory_allocation_data
{
    using timestamp_t     = rocprofiler_timestamp_t;
    using callback_data_t = rocprofiler_callback_tracing_memory_allocation_data_t;
    using buffered_data_t = rocprofiler_buffer_tracing_memory_allocation_record_t;

    rocprofiler_thread_id_t                   tid            = common::get_tid();
    rocprofiler_agent_id_t                    agent          = null_rocp_agent_id;
    uint64_t                                  size_allocated = 0;
    rocprofiler_address_t                     address        = {.handle = 0};
    uint64_t                                  start_ts       = 0;
    context::correlation_id*                  correlation_id = nullptr;
    tracing::tracing_data                     tracing_data   = {};
    rocprofiler_memory_allocation_operation_t func           = ROCPROFILER_MEMORY_ALLOCATION_NONE;

    callback_data_t get_callback_data(timestamp_t _beg = 0, timestamp_t _end = 0) const;
    buffered_data_t get_buffered_record(const context_t* _ctx,
                                        timestamp_t      _beg = 0,
                                        timestamp_t      _end = 0) const;
};

memory_allocation_data::callback_data_t
memory_allocation_data::get_callback_data(timestamp_t _beg, timestamp_t _end) const
{
    return common::init_public_api_struct(
        callback_data_t{}, _beg, _end, agent, address, size_allocated);
}

memory_allocation_data::buffered_data_t
memory_allocation_data::get_buffered_record(const context_t* _ctx,
                                            timestamp_t      _beg,
                                            timestamp_t      _end) const
{
    auto _external_corr_id =
        (_ctx) ? tracing_data.external_correlation_ids.at(_ctx) : context::null_user_data;
    auto _corr_id = rocprofiler_correlation_id_t{
        correlation_id->internal, _external_corr_id, correlation_id->ancestor};

    return common::init_public_api_struct(buffered_data_t{},
                                          ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                                          func,
                                          _corr_id,
                                          correlation_id->thread_idx,
                                          _beg,
                                          _end,
                                          agent,
                                          address,
                                          size_allocated);
}

// Callback function to populate the mapping of agents to regions
template <size_t OpIdx, typename T>
hsa_status_t
callback_populate_map(T region_or_pool, void* data)
{
    auto _agent_map_pair = static_cast<typename memory_allocation_info<OpIdx>::pairtype*>(data);
    auto _rocprof_agent  = _agent_map_pair->second;
    auto existing_map    = _agent_map_pair->first;

    existing_map->insert({region_or_pool, _rocprof_agent});
    return HSA_STATUS_SUCCESS;
}

// Returns the rocprofiler agent when given the region/pool
template <size_t OpIdx, typename T, typename IterateFunc, typename CallbackFunc>
rocprofiler_agent_id_t
get_agent(T val, IterateFunc iterate_func, CallbackFunc callback)
{
    static auto existing = typename memory_allocation_info<OpIdx>::maptype();

    if(existing.count(val) == 0)
    {
        auto agents = rocprofiler::agent::get_agents();
        for(const auto* itr : agents)
        {
            auto hsa_agent = rocprofiler::agent::get_hsa_agent(itr);
            if(hsa_agent)
            {
                const auto* rocprof_agent = rocprofiler::agent::get_rocprofiler_agent(*hsa_agent);
                if(rocprof_agent)
                {
                    auto data = typename memory_allocation_info<OpIdx>::pairtype{&existing,
                                                                                 rocprof_agent->id};
                    iterate_func(*hsa_agent, callback, &data);
                }
            }
        }
    }
    return existing.count(val) == 0 ? null_rocp_agent_id : existing.at(val);
}

rocprofiler_address_t
handle_starting_addr(void** starting_addr_pointer)
{
    return rocprofiler_address_t{.ptr = (starting_addr_pointer) ? *starting_addr_pointer : nullptr};
}

// The handle field of hsa_amd_vmem_alloc_handle_t is the starting address
// cast as uint64_t, so returning the handle field after casting to void* suffices
rocprofiler_address_t
handle_starting_addr(hsa_amd_vmem_alloc_handle_t* vmem_alloc_handle)
{
    return rocprofiler_address_t{.handle = (vmem_alloc_handle) ? vmem_alloc_handle->handle : 0};
}

// Handling starting address for free memory operations
rocprofiler_address_t
handle_starting_addr(void* starting_addr_pointer)
{
    return rocprofiler_address_t{.ptr = starting_addr_pointer};
}

// Handles starting address for releasing handle
rocprofiler_address_t
handle_starting_addr(hsa_amd_vmem_alloc_handle_t vmem_alloc_handle)
{
    return rocprofiler_address_t{.handle = vmem_alloc_handle.handle};
}

// Wrapper implementation that stores memory allocation information
template <size_t TableIdx, size_t OpIdx, typename... Args>
hsa_status_t
memory_allocation_impl(Args... args)
{
    constexpr auto N                = sizeof...(Args);
    constexpr auto address_idx      = arg_indices<OpIdx>::address_idx;
    constexpr auto size_idx         = arg_indices<OpIdx>::size_idx;
    constexpr auto region_idx       = arg_indices<OpIdx>::region_idx;
    constexpr auto operation        = memory_allocation_op<OpIdx>::operation_idx;
    constexpr auto rocprofiler_enum = memory_allocation_info<operation>::operation_idx;

    auto&&                 _tied_args = std::tie(args...);
    memory_allocation_data _data{};

    {
        auto tracing_data = tracing::tracing_data{};

        tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                                   ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                                   OpIdx,
                                   tracing_data);
        // if no contexts are tracing memory copies for this direction, execute as usual
        if(tracing_data.empty())
        {
            return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                          std::move(_tied_args),
                          std::make_index_sequence<N>{});
        }
        _data.tracing_data = std::move(tracing_data);
    }

    auto& tracing_data          = _data.tracing_data;
    auto  starting_addr_pointer = std::get<address_idx>(_tied_args);
    auto  region_or_pool        = std::get<region_idx>(_tied_args);

    _data.tid   = common::get_tid();
    _data.agent = get_agent<operation>(
        region_or_pool,
        memory_allocation_info<operation>{}(),
        callback_populate_map<operation, typename memory_allocation_info<operation>::searchtype>);
    _data.size_allocated = std::get<size_idx>(_tied_args);
    _data.func           = rocprofiler_enum;
    _data.correlation_id = context::get_latest_correlation_id();

    bool _constructed_corr_id = false;
    if(!_data.correlation_id)
    {
        constexpr auto ref_count = 1;
        _data.correlation_id     = context::correlation_tracing_service::construct(ref_count);
        _constructed_corr_id     = true;
    }
    else
    {
        // increase the reference count to prevent this correlation ID from being retired by another
        // service
        _data.correlation_id->add_ref_count();
    }

    auto thr_id = _data.correlation_id->thread_idx;
    tracing::populate_external_correlation_ids(
        tracing_data.external_correlation_ids,
        thr_id,
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION,
        rocprofiler_enum,
        _data.correlation_id->internal);

    if(!tracing_data.callback_contexts.empty())
    {
        auto _tracer_data = _data.get_callback_data();

        tracing::execute_phase_enter_callbacks(tracing_data.callback_contexts,
                                               thr_id,
                                               _data.correlation_id->internal,
                                               tracing_data.external_correlation_ids,
                                               _data.correlation_id->ancestor,
                                               ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                                               rocprofiler_enum,
                                               _tracer_data);
        // enter callback may update the external correlation id field
        tracing::update_external_correlation_ids(
            tracing_data.external_correlation_ids,
            thr_id,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION);
    }
    auto start_ts = common::timestamp_ns();
    auto _ret     = invoke(
        get_next_dispatch<TableIdx, OpIdx>(), std::move(_tied_args), std::make_index_sequence<N>{});
    auto end_ts = common::timestamp_ns();
    // Starting address is set after memory_allocation function is run. May need additional safety
    // checks before retrieving starting address?
    if(starting_addr_pointer != nullptr)
    {
        _data.address = handle_starting_addr(starting_addr_pointer);
    }

    if(!tracing_data.empty())
    {
        if(!_data.tracing_data.callback_contexts.empty())
        {
            auto _tracer_data = _data.get_callback_data(start_ts, end_ts);

            tracing::execute_phase_exit_callbacks(_data.tracing_data.callback_contexts,
                                                  _data.tracing_data.external_correlation_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                                                  rocprofiler_enum,
                                                  _tracer_data);
        }

        if(!_data.tracing_data.buffered_contexts.empty())
        {
            auto record = _data.get_buffered_record(nullptr, start_ts, end_ts);

            tracing::execute_buffer_record_emplace(_data.tracing_data.buffered_contexts,
                                                   _data.tid,
                                                   _data.correlation_id->internal,
                                                   _data.tracing_data.external_correlation_ids,
                                                   _data.correlation_id->ancestor,
                                                   ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                                                   rocprofiler_enum,
                                                   record);
        }
    }

    // decrement the reference count after usage in the callback/buffers
    _data.correlation_id->sub_ref_count();

    if(_constructed_corr_id) context::pop_latest_correlation_id(_data.correlation_id);

    return _ret;
}

// Wrapper implementation that stores memory free operation information
template <size_t TableIdx, size_t OpIdx, typename... Args>
hsa_status_t
memory_free_impl(Args... args)
{
    constexpr auto N                = sizeof...(Args);
    constexpr auto address_idx      = arg_indices<OpIdx>::address_idx;
    constexpr auto operation        = memory_allocation_op<OpIdx>::operation_idx;
    constexpr auto rocprofiler_enum = memory_allocation_info<operation>::operation_idx;

    common::consume_args(arg_indices<OpIdx>::size_idx, arg_indices<OpIdx>::region_idx);

    auto&&                 _tied_args = std::tie(args...);
    memory_allocation_data _data{};

    {
        auto tracing_data = tracing::tracing_data{};

        tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                                   ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                                   OpIdx,
                                   tracing_data);
        // if no contexts are tracing memory copies for this direction, execute as usual
        if(tracing_data.empty())
        {
            return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                          std::move(_tied_args),
                          std::make_index_sequence<N>{});
        }
        _data.tracing_data = std::move(tracing_data);
    }

    auto& tracing_data = _data.tracing_data;

    _data.tid            = common::get_tid();
    _data.func           = rocprofiler_enum;
    _data.correlation_id = context::get_latest_correlation_id();
    _data.address        = handle_starting_addr(std::get<address_idx>(_tied_args));

    bool _constructed_corr_id = false;
    if(!_data.correlation_id)
    {
        constexpr auto ref_count = 1;
        _data.correlation_id     = context::correlation_tracing_service::construct(ref_count);
        _constructed_corr_id     = true;
    }
    else
    {
        // increase the reference count to prevent this correlation ID from being retired by another
        // service
        _data.correlation_id->add_ref_count();
    }

    auto thr_id = _data.correlation_id->thread_idx;
    tracing::populate_external_correlation_ids(
        tracing_data.external_correlation_ids,
        thr_id,
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION,
        rocprofiler_enum,
        _data.correlation_id->internal);

    if(!tracing_data.callback_contexts.empty())
    {
        auto _tracer_data = _data.get_callback_data();

        tracing::execute_phase_enter_callbacks(tracing_data.callback_contexts,
                                               thr_id,
                                               _data.correlation_id->internal,
                                               tracing_data.external_correlation_ids,
                                               _data.correlation_id->ancestor,
                                               ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                                               rocprofiler_enum,
                                               _tracer_data);
        // enter callback may update the external correlation id field
        tracing::update_external_correlation_ids(
            tracing_data.external_correlation_ids,
            thr_id,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION);
    }
    auto start_ts = common::timestamp_ns();
    auto _ret     = invoke(
        get_next_dispatch<TableIdx, OpIdx>(), std::move(_tied_args), std::make_index_sequence<N>{});
    auto end_ts = common::timestamp_ns();

    if(!tracing_data.empty())
    {
        if(!_data.tracing_data.callback_contexts.empty())
        {
            auto _tracer_data = _data.get_callback_data(start_ts, end_ts);

            tracing::execute_phase_exit_callbacks(_data.tracing_data.callback_contexts,
                                                  _data.tracing_data.external_correlation_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                                                  rocprofiler_enum,
                                                  _tracer_data);
        }

        if(!_data.tracing_data.buffered_contexts.empty())
        {
            auto record = _data.get_buffered_record(nullptr, start_ts, end_ts);

            tracing::execute_buffer_record_emplace(_data.tracing_data.buffered_contexts,
                                                   _data.tid,
                                                   _data.correlation_id->internal,
                                                   _data.tracing_data.external_correlation_ids,
                                                   _data.correlation_id->ancestor,
                                                   ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                                                   rocprofiler_enum,
                                                   record);
        }
    }

    // decrement the reference count after usage in the callback/buffers
    _data.correlation_id->sub_ref_count();

    if(_constructed_corr_id) context::pop_latest_correlation_id(_data.correlation_id);

    return _ret;
}

}  // namespace
// check out the assembly here... this compiles to a switch statement
const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_MEMORY_ALLOCATION_LAST>{});
}

uint32_t
id_by_name(const char* name)
{
    return id_by_name(name, std::make_index_sequence<ROCPROFILER_MEMORY_ALLOCATION_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_MEMORY_ALLOCATION_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_MEMORY_ALLOCATION_LAST>{});
    return _data;
}

std::vector<const char*>
get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_MEMORY_ALLOCATION_LAST);
    get_names(_data, std::make_index_sequence<ROCPROFILER_MEMORY_ALLOCATION_LAST>{});
    return _data;
}

template <size_t TableIdx, typename LookupT = internal_table, typename Tp, size_t OpIdx>
void
memory_allocation_save(Tp* _orig, uint64_t _tbl_instance, std::integral_constant<size_t, OpIdx>)
{
    using table_type = typename hsa_table_lookup<TableIdx>::type;

    if constexpr(std::is_same<table_type, Tp>::value)
    {
        auto _meta = hsa_api_meta<TableIdx, OpIdx>{};

        // original table and function
        auto& _orig_table = _meta.get_table(_orig);
        auto& _orig_func  = _meta.get_table_func(_orig_table);

        // table with copy function
        auto& _allocate_func = get_next_dispatch<TableIdx, OpIdx>();

        ROCP_FATAL_IF(_allocate_func && _tbl_instance == 0)
            << _meta.name << " has non-null function pointer " << _allocate_func
            << " despite this being the first instance of the library being copies";

        if(!_allocate_func)
        {
            ROCP_TRACE << "copying table entry for " << _meta.name;
            _allocate_func = _orig_func;
        }
        else
        {
            ROCP_TRACE << "skipping copying table entry for " << _meta.name
                       << " from table instance " << _tbl_instance;
        }
    }
}
template <size_t TableIdx,
          typename LookupT = internal_table,
          typename Tp,
          size_t OpIdx,
          size_t... OpIdxTail>
void
memory_allocation_save(Tp* _orig, uint64_t _tbl_instance, std::index_sequence<OpIdx, OpIdxTail...>)
{
    memory_allocation_save<TableIdx, LookupT>(
        _orig, _tbl_instance, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0)
        memory_allocation_save<TableIdx, LookupT>(
            _orig, _tbl_instance, std::index_sequence<OpIdxTail...>{});
}

template <typename TableT>
void
memory_allocation_save(TableT* _orig, uint64_t _tbl_instance)
{
    constexpr auto TableIdx = hsa_table_id_lookup<TableT>::value;
    if(_orig)
        memory_allocation_save<TableIdx, internal_table>(
            _orig, _tbl_instance, memory_allocation_seq<TableIdx>::memory_allocation_index_seq_t);
}

template <size_t TableIdx, typename Tp, size_t OpIdx>
void
memory_allocation_wrap(Tp* _orig, std::integral_constant<size_t, OpIdx>)
{
    auto  _meta  = hsa_api_meta<TableIdx, OpIdx>{};
    auto& _table = _meta.get_table(_orig);
    auto& _func  = _meta.get_table_func(_table);

    auto& _dispatch = get_next_dispatch<TableIdx, OpIdx>();
    CHECK_NOTNULL(_dispatch);
    constexpr auto LocalIdx = memory_allocation_op<OpIdx>::operation_idx;
    _func = memory_allocation_info<LocalIdx>::template get_memory_allocation_impl<TableIdx, OpIdx>(
        _func);
}

template <size_t TableIdx, typename Tp, size_t OpIdx, size_t... OpIdxTail>
void
memory_allocation_wrap(Tp* _orig, std::index_sequence<OpIdx, OpIdxTail...>)
{
    memory_allocation_wrap<TableIdx>(_orig, std::integral_constant<size_t, OpIdx>{});
    if constexpr(sizeof...(OpIdxTail) > 0)
        memory_allocation_wrap<TableIdx>(_orig, std::index_sequence<OpIdxTail...>{});
}

template <typename TableT>
void
memory_allocation_wrap(TableT* _orig)
{
    constexpr auto TableIdx = hsa_table_id_lookup<TableT>::value;
    if(_orig)
    {
        memory_allocation_wrap<TableIdx>(
            _orig, memory_allocation_seq<TableIdx>::memory_allocation_index_seq_t);
    }
}

}  // namespace memory_allocation

template <typename TableT>
void
memory_allocation_init(TableT* _orig, uint64_t _tbl_instance)
{
    constexpr auto TableIdx = hsa_table_id_lookup<TableT>::value;
    if(_orig)
    {
        memory_allocation::memory_allocation_save<TableIdx>(
            _orig,
            _tbl_instance,
            memory_allocation::memory_allocation_seq<TableIdx>::memory_allocation_index_seq_t);

        auto ctxs = context::get_registered_contexts(memory_allocation::context_filter);
        if(!ctxs.empty())
        {
            memory_allocation::memory_allocation_wrap<TableIdx>(
                _orig,
                memory_allocation::memory_allocation_seq<TableIdx>::memory_allocation_index_seq_t);
        }
    }
}

#define INSTANTIATE_MEMORY_ALLOC_FUNC(TABLE_TYPE, TABLE_IDX)                                       \
    template void memory_allocation_init<TABLE_TYPE>(TABLE_TYPE * _tbl, uint64_t _instv);          \
    template void memory_allocation::memory_allocation_save<TABLE_TYPE>(TABLE_TYPE * _tbl,         \
                                                                        uint64_t _instv);          \
    template void memory_allocation::memory_allocation_wrap<TABLE_TYPE>(TABLE_TYPE * _tbl);

INSTANTIATE_MEMORY_ALLOC_FUNC(hsa_core_table_t, ROCPROFILER_HSA_TABLE_ID_Core)
INSTANTIATE_MEMORY_ALLOC_FUNC(hsa_amd_ext_table_t, ROCPROFILER_HSA_TABLE_ID_AmdExt)
#undef INSTANTIATE_MEMORY_ALLOC_FUNC

}  // namespace hsa
}  // namespace rocprofiler
