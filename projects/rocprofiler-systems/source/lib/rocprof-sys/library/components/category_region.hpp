// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common_types.hpp"
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

#include <concepts>
#include <map>
#include <span>
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

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
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
    timestamp_t   start_ts  = 0;
    std::string   args      = {};
    std::uint32_t arg_count = 0;
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

template <typename Tp>
std::string
get_serialized_arg_type()
{
    using value_type = std::decay_t<Tp>;
    if constexpr(std::is_convertible<value_type, std::string_view>::value)
    {
        return "string";
    }
    else
    {
        return rocprofsys::utility::demangle<value_type>();
    }
}

template <typename Tp>
std::string
get_serialized_arg_value(Tp&& value)
{
    std::stringstream ss;
    ss << std::forward<Tp>(value);
    return ss.str();
}

// Append one trace-cache record "idx;;type;;name;;value;;" to args_str
template <typename KeyT, typename ValueT>
void
append_serialized_arg(std::string& args_str, std::uint32_t idx, KeyT&& key,
                      ValueT&& value)
{
    args_str += rocprofsys::get_args_string({ rocprofsys::argument_info{
        .arg_number = idx,
        .arg_type   = get_serialized_arg_type<ValueT>(),
        .arg_name   = std::string{ std::string_view{ std::forward<KeyT>(key) } },
        .arg_value  = get_serialized_arg_value(std::forward<ValueT>(value)) } });
}

// Same record format, but type/value are pre-stringified by the caller
inline void
append_serialized_arg(std::string& args_str, std::uint32_t idx, std::string_view arg_type,
                      std::string_view key, std::string_view value)
{
    args_str += rocprofsys::get_args_string(
        { rocprofsys::argument_info{ .arg_number = idx,
                                     .arg_type   = std::string{ arg_type },
                                     .arg_name   = std::string{ key },
                                     .arg_value  = std::string{ value } } });
}

// A type qualifies as a trace-cache argument "name" slot when it is string-like.
template <typename Tp>
concept trace_cache_arg_name = std::convertible_to<std::decay_t<Tp>, std::string_view>;

// Args passed to start()/stop() qualify as cacheable ("arg-name", value) pairs only when
// they are non-empty, grouped two-by-two, and every "name" slot is string-like
template <typename TupleT, size_t... Idx>
constexpr bool
has_trace_cache_arg_pairs(std::index_sequence<Idx...>)
{
    if constexpr(sizeof...(Idx) == 0 || sizeof...(Idx) % 2 != 0)
    {
        return false;
    }
    else
    {
        return (
            (Idx % 2 != 0 || trace_cache_arg_name<std::tuple_element_t<Idx, TupleT>>) &&
            ...);
    }
}

template <size_t Idx = 0, typename TupleT>
void
append_serialized_args(std::string& args_str, TupleT&& args)
{
    if constexpr(Idx < std::tuple_size<std::remove_reference_t<TupleT>>::value)
    {
        append_serialized_arg(args_str, Idx / 2, std::get<Idx>(args),
                              std::get<Idx + 1>(args));
        append_serialized_args<Idx + 2>(args_str, std::forward<TupleT>(args));
    }
}

// Serialize explicit ("arg-name", value) argument pairs passed to start() into the
// trace-cache wire format. Returns an empty string when Args are not name/value pairs.
template <typename... Args>
std::string
serialize_name_value_pairs(Args&&... args)
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    using tuple_type = std::tuple<Args...>;
    if constexpr(has_trace_cache_arg_pairs<tuple_type>(
                     std::make_index_sequence<sizeof...(Args)>{}))
    {
        auto        args_tuple = std::forward_as_tuple(args...);
        std::string args_str;
        append_serialized_args(args_str, args_tuple);
        return args_str;
    }
    else
    {
        return {};
    }
}

// Renumber the arg_number field of every record in args_str so that the first
// record becomes next_idx, the second next_idx+1, and so on. Returns the number
// of records that were renumbered.
inline std::uint32_t
renumber_serialized_args(std::string& args_str, std::uint32_t next_idx)
{
    auto args = rocprofsys::process_arguments_string(args_str);
    for(auto& arg : args)
        arg.arg_number = next_idx++;
    args_str = rocprofsys::get_args_string(args);
    return static_cast<std::uint32_t>(args.size());
}

// Counts the number of records (each "idx;;type;;name;;value;;") in args_str.
inline std::uint32_t
count_serialized_args(const std::string& args_str)
{
    return static_cast<std::uint32_t>(
        rocprofsys::process_arguments_string(args_str).size());
}

// Decode a single rocprofsys_annotation_t into a trace-cache argument record.
// Returns false (leaving out untouched) when the annotation is empty/short-circuited.
// or its type is out of range.
template <size_t Idx, size_t... Tail>
bool
annotation_to_argument(const rocprofsys_annotation_t& annotation,
                       rocprofsys::argument_info& out, std::index_sequence<Idx, Tail...>)
{
    if(annotation.name == nullptr || annotation.type == ROCPROFSYS_VALUE_NONE ||
       annotation.value == nullptr)
        return false;

    if(annotation.type == Idx)
    {
        using type = rocprofsys::tracing::annotation_value_type_t<Idx>;

        out.arg_name = annotation.name;
        if constexpr(rocprofsys::concepts::is_string_type<type>::value)
        {
            // ROCPROFSYS_VALUE_CSTR: value is the C-string itself
            out.arg_type  = "string";
            out.arg_value = reinterpret_cast<const char*>(annotation.value);
        }
        else
        {
            out.arg_type = rocprofsys::utility::demangle<type>();
            // pointers are passed by value; everything else is dereferenced
            if constexpr(std::is_pointer<type>::value)
                out.arg_value =
                    fmt::format("{}", reinterpret_cast<type>(annotation.value));
            else
                out.arg_value =
                    fmt::format("{}", *reinterpret_cast<const type*>(annotation.value));
        }
        return true;
    }

    if constexpr(sizeof...(Tail) > 0)
        return annotation_to_argument(annotation, out, std::index_sequence<Tail...>{});
    else
        return false;
}

// Transforms rocprofsys_annotation_t records into the trace-cache wire format,
// skipping entries that are empty or have an out-of-range type.
inline std::string
serialize_annotation_args(std::span<const rocprofsys_annotation_t> annotations)
{
    if(annotations.empty()) return {};

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    rocprofsys::function_args_t args;
    args.reserve(annotations.size());

    std::uint32_t idx = 0;
    for(const auto& annotation : annotations)
    {
        rocprofsys::argument_info info{};
        if(annotation_to_argument(
               annotation, info,
               rocprofsys::utility::make_index_sequence_range<1,
                                                              ROCPROFSYS_VALUE_LAST>{}))
        {
            info.arg_number = idx++;
            args.push_back(std::move(info));
        }
    }

    return rocprofsys::get_args_string(args);
}

// Serializes gotcha audit arguments into the trace-cache format.
// Names are synthesized as "arg{N}-{demangled-type}" to mirror add_perfetto_annotation
template <typename... Args>
std::string
serialize_annotation_args(Args&&... args)
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    std::string   args_str = {};
    std::uint32_t idx      = 0;
    ROCPROFSYS_FOLD_EXPRESSION(
        (append_serialized_arg(
             args_str, idx,
             fmt::format("arg{}-{}", idx,
                         rocprofsys::utility::demangle<std::remove_reference_t<Args>>()),
             std::forward<Args>(args)),
         ++idx));
    return args_str;
}

// Outgoing audits pass at most one return value. This matches perfetto's single "return"
// annotation.
template <typename T>
std::string
serialize_return_arg(T&& value)
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    std::string args_str = {};
    append_serialized_arg(args_str, 0, "return", std::forward<T>(value));
    return args_str;
}

template <typename CategoryT, typename... Args>
void
cache_start(const char* name, std::string args_str = {})
{
    const auto start_ts =
        static_cast<timestamp_t>(rocprofsys::comp::wall_clock::record());
    const auto arg_count = count_serialized_args(args_str);
    map_name_to_args[{ name, rocprofsys::trait::name<CategoryT>::value }].push_back(
        pending_cache_entry{ start_ts, std::move(args_str), arg_count });
}

// Appends already-serialized args to the most recent (top-of-stack) pending entry
// for {name, category} on this thread. Any existing args keep their numbering and the
// appended args are renumbered to continue the sequence
template <typename CategoryT>
void
append_cache_args(const char* name, std::string args_str)
{
    if(args_str.empty()) return;

    auto key = entry_key{ name, rocprofsys::trait::name<CategoryT>::value };
    auto itr = map_name_to_args.find(key);
    if(itr != map_name_to_args.end() && !itr->second.empty())
    {
        auto& entry = itr->second.back();
        // If args were previously stored, the new ones must be renumbered to accommodate
        auto appended_arg_count = renumber_serialized_args(args_str, entry.arg_count);

        // Update the entry with the new args
        entry.arg_count += appended_arg_count;
        entry.args += std::move(args_str);
    }
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

    // Appends pre-serialized args to the currently-open region with this name.
    // Used by the gotcha audit and perfetto-annotation paths
    static void append_cache_args(std::string_view name, std::string serialized_args);

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

    // Gotcha starts pass the region name followed by ("arg-name", value) pairs.
    // Serialize those pairs into the trace-cache wire format
    // (Perfetto annotations are handled separately)
    constexpr bool _has_cache_args = has_trace_cache_arg_pairs<std::tuple<Args...>>(
        std::make_index_sequence<sizeof...(Args)>{});
    if constexpr(_has_cache_args)
    {
        cache_args = serialize_name_value_pairs(args...);
    }

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
void
category_region<CategoryT>::append_cache_args(std::string_view name,
                                              std::string      serialized_args)
{
    if(name.empty() || serialized_args.empty()) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto _hash = tim::add_hash_id(name);
    name       = tim::get_hash_identifier_fast(_hash);
    ::append_cache_args<CategoryT>(name.data(), std::move(serialized_args));
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

    if constexpr(sizeof...(Args) > 0)
    {
        append_cache_args(_data.tool_id.c_str(), serialize_annotation_args(_args...));
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::outgoing,
                                  Args&&... _args)
{
    if constexpr(sizeof...(Args) > 0)
    {
        append_cache_args(_data.tool_id.c_str(), serialize_return_arg(_args...));
    }

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

    if constexpr(sizeof...(Args) > 0)
    {
        append_cache_args(_name.data(), serialize_annotation_args(_args...));
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(std::string_view _name, audit::outgoing,
                                  Args&&... _args)
{
    if constexpr(sizeof...(Args) > 0)
    {
        append_cache_args(_name.data(), serialize_return_arg(_args...));
    }

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
