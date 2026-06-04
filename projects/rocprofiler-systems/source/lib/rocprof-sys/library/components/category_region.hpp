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

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
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

/// Per-thread stack of region begin timestamps
struct pending_region_storage_t
{
    std::uint64_t                                 thread_id        = 0;
    std::map<entry_key, std::vector<timestamp_t>> map_name_to_args = {};
    // Only contended during flush_all_pending_cached_entries(), which may drain this
    // storage while its owning thread is still in cache_start/cache_stop
    std::mutex mutex = {};
};

/// Global directory holding every thread's pending_region_storage_t. A thread adds its
/// own entry via acquire() and removes it via release() when it exits.
/// Leaked so thread-local destructors calling release() at process exit can't touch a
/// freed registry
struct pending_region_registry_t
{
    std::vector<std::unique_ptr<pending_region_storage_t>> thread_storages      = {};
    std::mutex                                             thread_storage_mutex = {};

    static pending_region_registry_t& instance()
    {
        static pending_region_registry_t* _instance = new pending_region_registry_t{};
        return *_instance;
    }

    static pending_region_storage_t* acquire()
    {
        auto&                       _self = instance();
        std::lock_guard<std::mutex> _lk{ _self.thread_storage_mutex };
        _self.thread_storages.emplace_back(std::make_unique<pending_region_storage_t>());
        return _self.thread_storages.back().get();
    }

    static void release(pending_region_storage_t* storage)
    {
        if(!storage) return;
        auto&                       _self = instance();
        std::lock_guard<std::mutex> _lk{ _self.thread_storage_mutex };
        auto                        itr =
            std::find_if(_self.thread_storages.begin(), _self.thread_storages.end(),
                         [storage](const auto& _s) { return _s.get() == storage; });
        if(itr != _self.thread_storages.end()) _self.thread_storages.erase(itr);
    }
};

namespace
{

/// Owns a thread's slot in the registry: registers a storage on first use and removes it
/// when the owning thread exits
struct pending_region_storage_handle_t
{
    pending_region_storage_handle_t()
    : storage{ pending_region_registry_t::acquire() }
    {}
    ~pending_region_storage_handle_t() { pending_region_registry_t::release(storage); }

    pending_region_storage_handle_t(const pending_region_storage_handle_t&) = delete;
    pending_region_storage_handle_t& operator=(const pending_region_storage_handle_t&) =
        delete;

    pending_region_storage_t* storage = nullptr;
};

/// Per-thread handle into the global registry. The first access on a thread allocates
/// and registers a dedicated storage; subsequent accesses reuse the cached pointer
inline pending_region_storage_t&
get_pending_region_storage()
{
    static thread_local pending_region_storage_handle_t _v = {};
    return *_v.storage;
}

/// Resolve the calling thread's system tid and make sure its metadata is registered
inline std::uint64_t
current_region_thread_id()
{
    const auto& extended_info = rocprofsys::thread_info::get(std::this_thread::get_id());
    if(extended_info.has_value() && extended_info->index_data.has_value())
    {
        constexpr size_t    UNKNOWN_TIME = 0;
        const std::uint64_t thread_id    = extended_info->index_data->system_value;
        rocprofsys::trace_cache::get_metadata_registry().add_thread_info(
            { getppid(), getpid(), thread_id, UNKNOWN_TIME, UNKNOWN_TIME, "{}" });
        return thread_id;
    }
    return 0;
}

void
cache_region(std::uint64_t thread_id, const std::string& name, std::uint64_t start_ts,
             std::uint64_t end_ts, const std::string& category)
{
    constexpr size_t      NO_CORRELATION_ID = 0;
    constexpr const char* CALLSTACK         = "{}";
    constexpr const char* ARGUMENTS         = "";
    rocprofsys::trace_cache::get_buffer_storage().store(
        rocprofsys::trace_cache::region_sample{
            thread_id, name.c_str(), NO_CORRELATION_ID, NO_CORRELATION_ID, start_ts,
            end_ts, CALLSTACK, ARGUMENTS, category.c_str() });
}

template <typename CategoryT, typename... Args>
void
cache_start(const char* name)
{
    const auto start_ts =
        static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
    auto&                       _storage = get_pending_region_storage();
    std::lock_guard<std::mutex> _lk{ _storage.mutex };
    if(_storage.thread_id == 0) _storage.thread_id = current_region_thread_id();
    _storage.map_name_to_args[{ name, rocprofsys::trait::name<CategoryT>::value }]
        .push_back(start_ts);
}

template <typename CategoryT>
void
cache_stop(const char* name)
{
    auto&                       _storage = get_pending_region_storage();
    std::lock_guard<std::mutex> _lk{ _storage.mutex };
    entry_key                   key{ name, rocprofsys::trait::name<CategoryT>::value };
    auto                        x = _storage.map_name_to_args.find(key);
    if(x != _storage.map_name_to_args.end() && !x->second.empty())
    {
        auto timestamp = x->second.back();
        x->second.pop_back();
        if(x->second.empty()) _storage.map_name_to_args.erase(x);

        const auto end_ts =
            static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
        const auto thread_id = current_region_thread_id();
        if(_storage.thread_id == 0) _storage.thread_id = thread_id;

        cache_region(thread_id, name, timestamp, end_ts,
                     rocprofsys::trait::name<CategoryT>::value);
    }
}

/// Drain a single thread's pending storage, emitting one region per outstanding
/// push with a synthetic end timestamp. The frame name is also suffixed with
/// "[incomplete]" (excluding the main thread's program-entry frame)
inline void
flush_pending_storage(pending_region_storage_t& _storage, timestamp_t end_ts)
{
    static constexpr std::string_view incomplete_suffix{ "incomplete" };
    // The main thread's program-entry frame (named after the executable) is
    // always open at finalization by design. It should not be tagged as incomplete
    const auto&                 _exe_name = rocprofsys::config::get_exe_name();
    std::lock_guard<std::mutex> _lk{ _storage.mutex };
    for(const auto& [key, start_ts_stack] : _storage.map_name_to_args)
    {
        const auto _name = (key.name == _exe_name)
                               ? key.name
                               : fmt::format("{} [{}]", key.name, incomplete_suffix);
        for(const auto& start_ts : start_ts_stack)
        {
            cache_region(_storage.thread_id, _name, start_ts, end_ts, key.category);
        }
    }
    _storage.map_name_to_args.clear();
}

/// Flush all pending cached entries for this thread
inline void
flush_pending_cached_entries()
{
    const auto end_ts = static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
    flush_pending_storage(get_pending_region_storage(), end_ts);
}

/// Flush pending cached entries across every thread.
/// Identical to flush_pending_cached_entries() but drains the storages of all threads
/// registered in the global registry
inline void
flush_all_pending_cached_entries()
{
    const auto end_ts = static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
    auto&      _registry = pending_region_registry_t::instance();
    std::lock_guard<std::mutex> _lk{ _registry.thread_storage_mutex };
    for(auto& _storage : _registry.thread_storages)
    {
        if(_storage) flush_pending_storage(*_storage, end_ts);
    }
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
//       In general, for known imbalances, add ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK=ON to the
//       ctest environment to avoid the CI_THROW check.
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
};

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::start(std::string_view name, Args&&... args)
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

    cache_start<CategoryT>(name.data());
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
