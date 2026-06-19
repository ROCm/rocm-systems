// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/causal/data.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"
#include <cstdint>

#include <map>
#include <thread>
#include <timemory/components/gotcha/backends.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/utility/types.hpp>
#include <tuple>
#include <vector>

#include "logger/debug.hpp"

#include <spdlog/fmt/ranges.h>

#include <string_view>
#include <utility>

struct entry_key
{
    std::string name;
    std::string category;

    friend bool operator<(const entry_key& lhs, const entry_key& rhs)
    {
        if(lhs.name != rhs.name)
        {
            return lhs.name < rhs.name;
        }

        return lhs.category < rhs.category;
    }
};

using timestamp_t = std::uint64_t;

struct pending_cache_entry
{
    timestamp_t start_ts = 0;
    std::string args     = {};
};

inline thread_local std::map<entry_key, std::vector<pending_cache_entry>>
    map_name_to_args;

namespace
{

void
cache_region(std::uint64_t thread_id, const std::string& name, std::uint64_t start_ts,
             std::uint64_t end_ts, const std::string& category,
             const std::string& args_str = {})
{
    constexpr size_t      NO_CORRELATION_ID = 0;
    constexpr const char* CALLSTACK         = "{}";
    rocprofsys::trace_cache::get_buffer_storage().store(
        rocprofsys::trace_cache::region_sample{
            thread_id, name.c_str(), NO_CORRELATION_ID, NO_CORRELATION_ID, start_ts,
            end_ts, CALLSTACK, args_str.c_str(), category.c_str() });
}

template <typename CategoryT, typename... Args>
void
cache_start(const char* name, std::string args_str = {})
{
    const auto start_ts =
        static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
    map_name_to_args[{ name, rocprofsys::trait::name<CategoryT>::value }].push_back(
        pending_cache_entry{ start_ts, std::move(args_str) });
}

template <typename CategoryT>
void
cache_stop(const char* name)
{
    entry_key key{ name, rocprofsys::trait::name<CategoryT>::value };
    auto      x = map_name_to_args.find(key);
    if(x != map_name_to_args.end() && !x->second.empty())
    {
        auto entry = std::move(x->second.back());
        x->second.pop_back();
        if(x->second.empty()) map_name_to_args.erase(x);

        const auto end_ts =
            static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
        std::uint64_t thread_id = 0;

        const auto& extended_info =
            rocprofsys::thread_info::get(std::this_thread::get_id());
        if(extended_info.has_value() && extended_info->index_data.has_value())
        {
            constexpr size_t UNKNOWN_TIME = 0;
            thread_id                     = extended_info->index_data->system_value;
            rocprofsys::trace_cache::get_metadata_registry().add_thread_info(
                { getppid(), getpid(), thread_id, UNKNOWN_TIME, UNKNOWN_TIME, "{}" });
        }

        cache_region(thread_id, name, entry.start_ts, end_ts,
                     rocprofsys::trait::name<CategoryT>::value, entry.args);
    }
}

/// Flush all pending cached entries for this thread.
/// Called during finalization to ensure entries that were started but not stopped
/// (e.g., main entry point) are written to the trace cache. Every pending frame
/// in each per-key stack is emitted, so recursive/self-nested regions that were
/// never popped still produce one region per outstanding push.
inline void
flush_pending_cached_entries()
{
    const auto end_ts = static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
    std::uint64_t thread_id = 0;

    const auto& extended_info = rocprofsys::thread_info::get(std::this_thread::get_id());
    if(extended_info.has_value() && extended_info->index_data.has_value())
    {
        constexpr size_t UNKNOWN_TIME = 0;
        thread_id                     = extended_info->index_data->system_value;
        rocprofsys::trace_cache::get_metadata_registry().add_thread_info(
            { getppid(), getpid(), thread_id, UNKNOWN_TIME, UNKNOWN_TIME, "{}" });
    }

    for(const auto& [key, entry_stack] : map_name_to_args)
    {
        for(const auto& entry : entry_stack)
        {
            cache_region(thread_id, key.name, entry.start_ts, end_ts, key.category,
                         entry.args);
        }
    }
    map_name_to_args.clear();
}
}  // namespace

namespace tim
{
namespace quirk
{
struct causal : concepts::quirk_type
{};

struct perfetto : concepts::quirk_type
{};

struct timemory : concepts::quirk_type
{};
}  // namespace quirk
}  // namespace tim

namespace rocprofsys
{
namespace component
{
using tim::is_one_of;
using tim::type_list;

// these categories increment push/pop counts, which are used for sanity checks since
// they should ALWAYS be popped if they were pushed
// Note: There is a known imbalance in the push/pop counts for category::host when using
//       OpenMP Tools (OMPT).
using tracing_count_categories_t =
    type_list<category::host, category::mpi, category::pthread, category::rocm_hip_api,
              category::rocm_hsa_api, category::rocm_rccl>;

// convert these categories to throughput points
using causal_throughput_categories_t =
    type_list<category::host, category::kokkos, category::rocm_ompt_api,
              category::rocm_hip_api, category::rocm_hsa_api, category::rocm_rccl,
              category::rocm_marker_api>;

// define this outside of category region functions so that the
// static thread_local is global instead of per-template instantiation
inline ThreadState
get_thread_status()
{
    static thread_local auto _thread_init_once = std::once_flag{};
    std::call_once(_thread_init_once, tracing::thread_init);

    return get_thread_state();
}

// timemory component which calls rocprof-sys functions
// (used in gotcha wrappers)
template <typename CategoryT>
struct category_region : comp::base<category_region<CategoryT>, void>
{
    using gotcha_data_t = tim::component::gotcha_data;

    static constexpr auto category_name = trait::name<CategoryT>::value;

    static std::string label()
    {
        return fmt::format("rocprofsys_{}_region", category_name);
    }

    template <typename... OptsT, typename... Args>
    static void start(std::string_view name, Args&&...);

    template <typename... OptsT, typename... Args>
    static void stop(std::string_view name, Args&&...);

    template <typename... OptsT, typename... Args>
    static void mark(std::string_view name, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(const gotcha_data_t&, audit::incoming, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(const gotcha_data_t&, audit::outgoing, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(std::string_view, audit::incoming, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(std::string_view, audit::outgoing, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(quirk::config<OptsT...>, Args&&...);

    static void start_with_args(std::string_view name, std::string serialized_args);

private:
    // Shared implementation for start() / start_with_args()
    template <typename... OptsT, typename... Args>
    static void start_impl(std::string_view name, std::string cache_args, Args&&...);
};

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::start(std::string_view name, Args&&... args)
{
    start_impl<OptsT...>(name, std::string{}, std::forward<Args>(args)...);
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::start_impl(std::string_view name, std::string cache_args,
                                       Args&&... args)
{
    // skip if category is disabled
    if(tracing::category_push_disabled<CategoryT>()) return;

    // unconditionally return if thread is disabled or finalized
    if(get_thread_state() == ThreadState::Disabled) return;
    if(get_state() >= State::Finalized) return;

    if(name.empty()) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    // the expectation here is that if the state is not active then the call
    // to rocprofsys_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !rocprofsys_init_tooling_hidden()) return;

    if(get_thread_status() == ThreadState::Disabled) return;

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::timemory, type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::perfetto, type_list<OptsT...>>::value);

    constexpr bool _ct_use_causal =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::causal, type_list<OptsT...>>::value);

    if(tracing::debug_push)
    {
        LOG_DEBUG("[{}][PID={}][state={}][thread_state={}] rocprofsys_push_region({})",
                  category_name, process::get_id(), std::to_string(get_state()),
                  std::to_string(get_thread_state()), name.data());
    }

    if constexpr(is_one_of<CategoryT, tracing_count_categories_t>::value)
    {
        ++tracing::push_count();
    }

    auto _hash = tim::add_hash_id(name);
    name       = tim::get_hash_identifier_fast(_hash);

    if constexpr(_ct_use_causal)
    {
        if constexpr(!is_one_of<CategoryT, causal_throughput_categories_t>::value)
        {
            if(get_use_causal()) causal::push_progress_point(name);
        }
    }

    if constexpr(_ct_use_timemory)
    {
        if(get_use_timemory())
        {
            tracing::push_timemory(CategoryT{}, name, std::forward<Args>(args)...);
        }
    }

    if constexpr(_ct_use_perfetto)
    {
        if(get_use_perfetto())
        {
            tracing::push_perfetto(CategoryT{}, name.data(), std::forward<Args>(args)...);
        }
    }

    cache_start<CategoryT>(name.data(), std::move(cache_args));
}

// Starts a region and attaches the pre-serialized args to it in a single push.
// The args ride through start_impl() into the lone cache_start() call, so the
// name is hashed and the per-thread map is touched exactly once. Because the
// args are a plain function argument (not a thread-local), there is no
// re-entrancy/misattribution window and no cross-call ordering contract.
template <typename CategoryT>
void
category_region<CategoryT>::start_with_args(std::string_view name,
                                            std::string      serialized_args)
{
    start_impl(name, std::move(serialized_args));
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::stop(std::string_view name, Args&&... args)
{
    // skip if category is disabled
    if(tracing::category_pop_disabled<CategoryT>()) return;

    if(get_thread_state() == ThreadState::Disabled) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::timemory, type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::perfetto, type_list<OptsT...>>::value);

    constexpr bool _ct_use_causal =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::causal, type_list<OptsT...>>::value);

    if(tracing::debug_pop)
    {
        LOG_DEBUG("[{}][PID={}][state={}][thread_state={}] rocprofsys_pop_region({})",
                  category_name, process::get_id(), std::to_string(get_state()),
                  std::to_string(get_thread_state()), name.data());
    }

    // only execute when active
    if(get_state() == State::Active)
    {
        if constexpr(is_one_of<CategoryT, tracing_count_categories_t>::value)
        {
            ++tracing::pop_count();
        }

        if constexpr(_ct_use_perfetto)
        {
            if(get_use_perfetto())
            {
                tracing::pop_perfetto(CategoryT{}, name.data(),
                                      std::forward<Args>(args)...);
            }
        }

        if constexpr(_ct_use_timemory)
        {
            if(get_use_timemory())
            {
                tracing::pop_timemory(CategoryT{}, name, std::forward<Args>(args)...);
            }
        }

        if constexpr(_ct_use_causal)
        {
            if constexpr(is_one_of<CategoryT, causal_throughput_categories_t>::value)
            {
                if(get_use_causal()) causal::mark_progress_point(name);
            }
            else
            {
                if(get_use_causal()) causal::pop_progress_point(name);
            }
        }

        cache_stop<CategoryT>(name.data());
    }
    else
    {
        LOG_DEBUG("[{}] rocprofsys_pop_region({}) ignored :: state = {}", category_name,
                  name.data(), std::to_string(get_state()));
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::mark(std::string_view name, Args&&...)
{
    constexpr bool _ct_use_causal =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::causal, type_list<OptsT...>>::value);

    if constexpr(!_ct_use_causal) return;

    // skip if category is disabled
    if(tracing::category_mark_disabled<CategoryT>()) return;

    // the expectation here is that if the state is not active then the call
    // to rocprofsys_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !rocprofsys_init_tooling_hidden()) return;

    // unconditionally return if thread is disabled or finalized
    if(get_thread_state() >= ThreadState::Completed) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    if(get_use_causal())
    {
        if(tracing::debug_mark)
        {
            LOG_DEBUG("[{}][PID={}][state={}][thread_state={}] rocprofsys_progress({})",
                      category_name, process::get_id(), std::to_string(get_state()),
                      std::to_string(get_thread_state()), name.data());
        }

        causal::mark_progress_point(name);
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::incoming,
                                  Args&&... _args)
{
    start<OptsT...>(_data.tool_id.c_str(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
        {
            std::int64_t _n = 0;
            ROCPROFSYS_FOLD_EXPRESSION(tracing::add_perfetto_annotation(
                ctx, rocprofsys::utility::demangle<std::remove_reference_t<Args>>(),
                _args, _n++));
        }
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::outgoing,
                                  Args&&... _args)
{
    stop<OptsT...>(_data.tool_id.c_str(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
            tracing::add_perfetto_annotation(
                ctx, "return",
                fmt::format("{}", fmt::join(std::forward_as_tuple(_args...), ", ")));
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(std::string_view _name, audit::incoming,
                                  Args&&... _args)
{
    start<OptsT...>(_name.data(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
        {
            std::int64_t _n = 0;
            ROCPROFSYS_FOLD_EXPRESSION(tracing::add_perfetto_annotation(
                ctx, rocprofsys::utility::demangle<std::remove_reference_t<Args>>(),
                _args, _n++));
        }
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(std::string_view _name, audit::outgoing,
                                  Args&&... _args)
{
    stop<OptsT...>(_name.data(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
            tracing::add_perfetto_annotation(
                ctx, "return",
                fmt::format("{}", fmt::join(std::forward_as_tuple(_args...), ", ")));
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(quirk::config<OptsT...>, Args&&... _args)
{
    audit<OptsT...>(std::forward<Args>(_args)...);
}

template <typename CategoryT>
struct local_category_region : comp::base<local_category_region<CategoryT>, void>
{
    using impl_type = category_region<CategoryT>;

    static constexpr auto category_name = impl_type::category_name;
    static std::string    label() { return impl_type::label(); }

    template <typename... OptsT, typename... Args>
    auto start(Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template start<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto stop(Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template stop<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto mark(Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template mark<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto audit(Args&&... args)
        -> decltype(impl_type::template audit<OptsT...>(std::declval<std::string_view>(),
                                                        std::forward<Args>(args)...))
    {
        if(m_prefix.empty()) return;
        return impl_type::template audit<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto audit(quirk::config<OptsT...>, Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template audit<OptsT...>(quirk::config<OptsT...>{}, m_prefix,
                                                   std::forward<Args>(args)...);
    }

    void set_prefix(std::string_view _v) { m_prefix = _v; }

private:
    std::string_view m_prefix = {};
};
}  // namespace component
}  // namespace rocprofsys
