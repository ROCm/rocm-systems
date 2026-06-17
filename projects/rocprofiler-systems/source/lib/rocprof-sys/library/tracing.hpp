// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/perfetto.hpp"
#include "core/perfetto/emitter.hpp"
#include "core/perfetto/engine.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/causal/sampling.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/tracing/annotation.hpp"
#include <cstdint>

#include <timemory/components/io/components.hpp>
#include <timemory/components/network/types.hpp>
#include <timemory/components/papi/types.hpp>
#include <timemory/components/rusage/components.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/components/timing/components.hpp>
#include <timemory/enum.h>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/types.hpp>

#include "logger/debug.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <ratio>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace tracing
{
using interval_data_instances = thread_data<std::vector<bool>>;
using hash_value_t            = core::perfetto::hash_value_t;

using core::perfetto::ensure_synthetic_process_track_emitted;
using core::perfetto::get_active_process_track;
using core::perfetto::get_perfetto_category_uuid;
using core::perfetto::get_perfetto_string;
using core::perfetto::get_perfetto_track;
using core::perfetto::get_perfetto_track_uuids;
using core::perfetto::get_perfetto_track_uuids_mutex;
using core::perfetto::hash_combine;
using core::perfetto::hash_combine_all;

using perfetto_annotate_component_types = tim::mpl::available_t<type_list<
    comp::cpu_clock, comp::cpu_util, comp::kernel_mode_time, comp::num_major_page_faults,
    comp::num_minor_page_faults, comp::page_rss, comp::peak_rss, comp::papi_array_t,
    comp::papi_vector, comp::priority_context_switch, comp::voluntary_context_switch,
    comp::process_cpu_clock, comp::process_cpu_util, comp::system_clock,
    comp::thread_cpu_clock, comp::thread_cpu_util, comp::user_clock, comp::user_mode_time,
    comp::virtual_memory>>;

//
//  declarations
//
extern ROCPROFSYS_HIDDEN_API bool debug_push;
extern ROCPROFSYS_HIDDEN_API bool debug_pop;
extern ROCPROFSYS_HIDDEN_API bool debug_user;
extern ROCPROFSYS_HIDDEN_API bool debug_mark;

void
copy_timemory_hash_ids();

std::vector<std::function<void()>>&
get_finalization_functions();

void
record_thread_start_time();

void
thread_init();

template <typename CategoryT>
auto&
get_category_stack();

template <typename CategoryT, typename... Args>
inline void
push_perfetto(CategoryT, const char*, Args&&...);

template <typename CategoryT, typename... Args>
inline void
pop_perfetto(CategoryT, const char*, Args&&...);

template <typename CategoryT, typename... Args>
inline void
push_perfetto_ts(CategoryT, const char*, std::uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_ts(CategoryT, const char*, std::uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
push_perfetto_track(CategoryT, const char*, ::perfetto::Track, std::uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_track(CategoryT, const char*, ::perfetto::Track, std::uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
mark_perfetto(CategoryT, const char*, Args&&...);

template <typename CategoryT, typename... Args>
inline void
mark_perfetto_ts(CategoryT, const char*, std::uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
mark_perfetto_track(CategoryT, const char*, ::perfetto::Track, std::uint64_t, Args&&...);

template <typename Tp = std::uint64_t>
ROCPROFSYS_INLINE auto
now()
{
    return ::tim::get_clock_real_now<Tp, std::nano>();
}

inline auto&
get_instrumentation_bundles(std::int64_t _tid = threading::get_id())
{
    return instrumentation_bundles::instance(construct_on_thread{ _tid });
}

inline auto&
push_count()
{
    static std::atomic<size_t> _v{ 0 };
    return _v;
}

inline auto&
pop_count()
{
    static std::atomic<size_t> _v{ 0 };
    return _v;
}

struct category_stack
{
    std::int32_t profile = 0;  // use signed so compiler doesn't have to
    std::int32_t tracing = 0;  // account for underflow/overflow
};

template <typename CategoryT>
auto&
get_category_stack()
{
    static thread_local auto _v = category_stack{};
    return _v;
}

template <typename CategoryT>
auto&
get_tracing_stack()
{
    return get_category_stack<CategoryT>().tracing;
}

template <typename CategoryT>
auto&
get_profile_stack()
{
    return get_category_stack<CategoryT>().profile;
}

template <typename CategoryT>
auto
category_push_disabled()
{
    return !trait::runtime_enabled<CategoryT>::get();
}

template <typename CategoryT>
auto
category_mark_disabled()
{
    return !trait::runtime_enabled<CategoryT>::get();
}

template <typename CategoryT>
auto
category_pop_disabled()
{
    return !trait::runtime_enabled<CategoryT>::get() &&
           (get_profile_stack<CategoryT>() + get_tracing_stack<CategoryT>()) <= 0;
}

template <typename CategoryT>
auto
tracing_pop_disabled()
{
    return !trait::runtime_enabled<CategoryT>::get() &&
           get_tracing_stack<CategoryT>() <= 0;
}

template <typename CategoryT>
auto
profile_pop_disabled()
{
    return !trait::runtime_enabled<CategoryT>::get() &&
           get_profile_stack<CategoryT>() <= 0;
}

template <typename CategoryT, typename... Args>
inline void
push_timemory(CategoryT, std::string_view name, Args&&... args)
{
    // skip if category is disabled
    if(category_push_disabled<CategoryT>()) return;

    auto& _data = tracing::get_instrumentation_bundles();
    if(ROCPROFSYS_LIKELY(_data != nullptr))
    {
        // this generates a hash for the raw string array
        auto _hash = tim::add_hash_id(name);
        _data->construct(_hash)->start(std::forward<Args>(args)...);
        // increment the profile stack
        ++get_profile_stack<CategoryT>();
    }
}

template <typename CategoryT>
inline std::pair<instrumentation_bundle_t*, size_t>
get_timemory(CategoryT, std::string_view name)
{
    using return_type = std::pair<instrumentation_bundle_t*, size_t>;
    // skip if category is disabled and not pushed on this thread
    if(profile_pop_disabled<CategoryT>()) return return_type{ nullptr, -1 };

    auto  _hash = tim::hash::get_hash_id(name);
    auto& _data = tracing::get_instrumentation_bundles();
    if(ROCPROFSYS_UNLIKELY(_data == nullptr || _data->empty()))
    {
        LOG_DEBUG("[rocprofsys_pop_trace] skipped {} :: empty bundle stack", name);
        return return_type{ nullptr, -1 };
    }

    auto*& _v_back = _data->back();
    if(ROCPROFSYS_LIKELY(_v_back->get_hash() == _hash))
    {
        return std::make_pair(_v_back, _data->size() - 1);
    }
    else if(_data->size() > 1)
    {
        for(size_t i = _data->size() - 1; i > 0; --i)
        {
            auto*& _v = _data->at(i - 1);
            if(_v->get_hash() == _hash)
            {
                return std::make_pair(_v, i - 1);
            }
        }
    }

    return return_type{ nullptr, -1 };
}

template <typename CategoryT, typename... Args>
inline auto
stop_timemory(CategoryT, std::string_view name, Args&&... args)
{
    using return_type = std::pair<instrumentation_bundle_t*, size_t>;

    // skip if category is disabled and not pushed on this thread
    if(profile_pop_disabled<CategoryT>()) return return_type{ nullptr, -1 };

    auto&& _data = get_timemory(CategoryT{}, name);
    if(_data.first)
    {
        _data.first->stop(std::forward<Args>(args)...);
    }
    return _data;
}

inline void
destroy_timemory(std::pair<instrumentation_bundle_t*, size_t> _data)
{
    if(_data.first)
    {
        auto& _bundles = tracing::get_instrumentation_bundles();
        if(ROCPROFSYS_LIKELY(_bundles != nullptr))
            _bundles->destroy(_data.first, _data.second);
    }
}

template <typename CategoryT, typename... Args>
inline void
pop_timemory(CategoryT, std::string_view name, Args&&... args)
{
    // skip if category is disabled and not pushed on this thread
    if(profile_pop_disabled<CategoryT>()) return;

    auto _data = stop_timemory(CategoryT{}, name, std::forward<Args>(args)...);
    if(_data.first) destroy_timemory(std::move(_data));
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto(CategoryT, const char* name, Args&&... args)
{
    // skip if category is disabled
    if(category_push_disabled<CategoryT>()) return;

    if constexpr(sizeof...(Args) == 1 &&
                 std::is_invocable<Args..., ::perfetto::EventContext>::value)
    {
        ++get_tracing_stack<CategoryT>();
        std::uint64_t _ts = now();
        if(config::get_perfetto_annotations())
        {
            TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, get_perfetto_string(name),
                              _ts, "begin_ns", _ts, std::forward<Args>(args)...);
        }
        else
        {
            TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, get_perfetto_string(name),
                              _ts, std::forward<Args>(args)...);
        }
    }
    else
    {
        using tuple_type = std::tuple<concepts::unqualified_type_t<Args>...>;
        using arg0_type  = concepts::tuple_element_t<0, tuple_type>;
        using arg1_type  = concepts::tuple_element_t<1, tuple_type>;

        if constexpr(std::is_same<arg0_type, ::perfetto::Track>::value &&
                     std::is_same<arg1_type, std::uint64_t>::value)
        {
            push_perfetto_track(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else if constexpr(std::is_same<arg0_type, std::uint64_t>::value)
        {
            push_perfetto_ts(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else
        {
            ++get_tracing_stack<CategoryT>();
            std::uint64_t _ts = now();
            TRACE_EVENT_BEGIN(
                trait::name<CategoryT>::value, get_perfetto_string(name), _ts,
                std::forward<Args>(args)..., [&](::perfetto::EventContext ctx) {
                    if(config::get_perfetto_annotations())
                    {
                        tracing::add_perfetto_annotation(ctx, "begin_ns", _ts);
                    }
                });
        }
    }
}

/// \brief This function is used to take an existing lambda accepting a
/// perfetto::EventContext and append the timemory annotations. Examples
/// are seen in the pop_perfetto* functions
template <typename CategoryT, typename Arg>
inline decltype(auto)
perfetto_annotate_timemory_data(CategoryT, const char* name, Arg&& arg)
{
    if constexpr(std::is_invocable<Arg, ::perfetto::EventContext>::value)
    {
        return [&arg, name](::perfetto::EventContext _ctx) {
            if(config::get_perfetto_annotations())
            {
                auto _timemory_data = get_timemory(CategoryT{}, name);
                if(_timemory_data.first)
                {
                    _timemory_data.first->stop();
                    _timemory_data.first
                        ->template invoke_with<tim::operation::perfetto_annotate>(
                            perfetto_annotate_component_types{}, _ctx);
                }
            }
            std::forward<Arg>(arg)(std::move(_ctx));
        };
    }
    else
    {
        return std::move(arg);
    }
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto(CategoryT, const char* name, Args&&... args)
{
    // skip if category is disabled and not pushed on this thread
    if(tracing_pop_disabled<CategoryT>()) return;

    if constexpr(sizeof...(Args) == 1 &&
                 std::is_invocable<Args..., ::perfetto::EventContext>::value)
    {
        // decrement tracing stack
        --get_tracing_stack<CategoryT>();
        std::uint64_t _ts = now();
        if(config::get_perfetto_annotations())
        {
            TRACE_EVENT_END(trait::name<CategoryT>::value, _ts, "end_ns", _ts,
                            perfetto_annotate_timemory_data(CategoryT{}, name,
                                                            std::forward<Args>(args))...);
        }
        else
        {
            TRACE_EVENT_END(trait::name<CategoryT>::value, _ts,
                            std::forward<Args>(args)...);
        }
    }
    else
    {
        using tuple_type = std::tuple<concepts::unqualified_type_t<Args>...>;
        using arg0_type  = concepts::tuple_element_t<0, tuple_type>;
        using arg1_type  = concepts::tuple_element_t<1, tuple_type>;

        if constexpr(std::is_same<arg0_type, ::perfetto::Track>::value &&
                     std::is_same<arg1_type, std::uint64_t>::value)
        {
            pop_perfetto_track(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else if constexpr(std::is_same<arg0_type, std::uint64_t>::value)
        {
            pop_perfetto_ts(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else
        {
            // decrement tracing stack
            --get_tracing_stack<CategoryT>();
            std::uint64_t _ts = now();
            TRACE_EVENT_END(
                trait::name<CategoryT>::value, _ts, std::forward<Args>(args)...,
                perfetto_annotate_timemory_data(
                    CategoryT{}, name, [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
                        {
                            tracing::add_perfetto_annotation(ctx, "end_ns", _ts);
                        }
                    }));
        }
    }

    (void) name;
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto_ts(CategoryT, const char* name, std::uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(category_push_disabled<CategoryT>()) return;

    ++get_tracing_stack<CategoryT>();
    TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, get_perfetto_string(name), _ts,
                      std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_ts(CategoryT, const char* name, std::uint64_t _ts, Args&&... args)
{
    // skip if category is disabled and not pushed on this thread
    if(tracing_pop_disabled<CategoryT>()) return;

    // decrement tracing stack
    --get_tracing_stack<CategoryT>();

    TRACE_EVENT_END(
        trait::name<CategoryT>::value, _ts,
        perfetto_annotate_timemory_data(CategoryT{}, name, std::forward<Args>(args))...);
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto_track(CategoryT, const char* name, ::perfetto::Track _track,
                    std::uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(category_push_disabled<CategoryT>()) return;

    ++get_tracing_stack<CategoryT>();
    core::perfetto::push_perfetto_track(CategoryT{}, name, _track, _ts,
                                        std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_track(CategoryT, const char* name, ::perfetto::Track _track,
                   std::uint64_t _ts, Args&&... args)
{
    // skip if category is disabled and not pushed on this thread
    if(tracing_pop_disabled<CategoryT>()) return;

    // decrement tracing stack
    --get_tracing_stack<CategoryT>();

    core::perfetto::pop_perfetto_track(
        CategoryT{}, name, _track, _ts,
        perfetto_annotate_timemory_data(CategoryT{}, name, std::forward<Args>(args))...);
}

template <typename CategoryT, typename... Args>
inline void
mark_perfetto(CategoryT, const char* name, Args&&... args)
{
    // skip if category is disabled
    if(category_mark_disabled<CategoryT>()) return;

    if constexpr(sizeof...(Args) == 1 &&
                 std::is_invocable<Args..., ::perfetto::EventContext>::value)
    {
        std::uint64_t _ts = now();
        if(config::get_perfetto_annotations())
        {
            TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, get_perfetto_string(name),
                                _ts, "ns", _ts, std::forward<Args>(args)...);
        }
        else
        {
            TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, get_perfetto_string(name),
                                _ts, std::forward<Args>(args)...);
        }
    }
    else
    {
        using tuple_type = std::tuple<concepts::unqualified_type_t<Args>...>;
        using arg0_type  = concepts::tuple_element_t<0, tuple_type>;
        using arg1_type  = concepts::tuple_element_t<1, tuple_type>;

        if constexpr(std::is_same<arg0_type, ::perfetto::Track>::value &&
                     std::is_same<arg1_type, std::uint64_t>::value)
        {
            mark_perfetto_track(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else if constexpr(std::is_same<arg0_type, std::uint64_t>::value)
        {
            mark_perfetto_ts(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else
        {
            std::uint64_t _ts = now();
            TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, get_perfetto_string(name),
                                _ts, std::forward<Args>(args)...,
                                [&](::perfetto::EventContext ctx) {
                                    if(config::get_perfetto_annotations())
                                    {
                                        tracing::add_perfetto_annotation(ctx, "ns", _ts);
                                    }
                                });
        }
    }
}

template <typename CategoryT, typename... Args>
inline void
mark_perfetto_ts(CategoryT, const char* name, std::uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(category_mark_disabled<CategoryT>()) return;

    TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, get_perfetto_string(name), _ts,
                        std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
mark_perfetto_track(CategoryT, const char* name, ::perfetto::Track _track,
                    std::uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(category_mark_disabled<CategoryT>()) return;

    TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, get_perfetto_string(name), _track,
                        _ts, std::forward<Args>(args)...);
}

template <typename FuncT>
std::int64_t
get_clock_skew(FuncT&& _timestamp_func, std::int64_t _n = 1)
{
    namespace cpu = tim::cpu;
    // synchronize timestamps
    // We'll take a CPU timestamp before and after taking a GPU timestmp, then
    // take the average of those two, hoping that it's roughly at the same time
    // as the GPU timestamp.
    auto _cpu_now = []() {
        cpu::fence();
        return now();
    };

    auto _gpu_now = [&_timestamp_func]() {
        cpu::fence();
        return std::forward<FuncT>(_timestamp_func)();
    };

    auto _compute = [&_cpu_now, &_gpu_now]() {
        volatile std::uint64_t _cpu_ts = 0;
        volatile std::uint64_t _gpu_ts = 0;
        _cpu_ts += _cpu_now();
        _gpu_ts += _gpu_now();
        _cpu_ts += _cpu_now();
        return static_cast<std::int64_t>(_cpu_ts / 2) -
               static_cast<std::int64_t>(_gpu_ts);
    };

    std::int64_t _diff = 0;
    for(std::int64_t i = 0; i < _n; ++i)
    {
        _diff += _compute();
    }
    return (_diff / _n);
}
}  // namespace tracing
}  // namespace rocprofsys
