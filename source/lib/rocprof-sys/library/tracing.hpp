// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/causal/sampling.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/tracing/annotation.hpp"

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

#include <atomic>
#include <functional>
#include <memory>
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
using interval_data_instances           = thread_data<std::vector<bool>>;
using hash_value_t                      = tim::hash_value_t;
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

std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids();

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

template <typename T>
auto
get_perfetto_string(T& name)
{
    if constexpr(std::is_const_v<T>)
    {
        return ::perfetto::StaticString{ name };
    }
    else
    {
        return ::perfetto::DynamicString{ name };
    }
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto(CategoryT, const char*, Args&&...);

template <typename CategoryT, typename... Args>
inline void
pop_perfetto(CategoryT, const char*, Args&&...);

template <typename CategoryT, typename... Args>
inline void
push_perfetto_ts(CategoryT, const char*, uint64_t _ts, Args&&...);

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_ts(CategoryT, const char*, uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
push_perfetto_track(CategoryT, const char*, ::perfetto::Track, uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_track(CategoryT, const char*, ::perfetto::Track, uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
mark_perfetto(CategoryT, const char*, Args&&...);

template <typename CategoryT, typename... Args>
inline void
mark_perfetto_ts(CategoryT, const char*, uint64_t, Args&&...);

template <typename CategoryT, typename... Args>
inline void
mark_perfetto_track(CategoryT, const char*, ::perfetto::Track, uint64_t, Args&&...);

//
//  definitions
//

template <typename CategoryT, typename... Args>
auto
get_perfetto_category_uuid(Args&&... _args)
{
    return tim::hash::get_hash_id(
        tim::hash::get_hash_id(JOIN('_', "rocprofsys", trait::name<CategoryT>::value)),
        std::forward<Args>(_args)...);
}

template <typename CategoryT, typename TrackT = ::perfetto::Track, typename FuncT,
          typename... Args>
auto
get_perfetto_track(CategoryT, FuncT&& _desc_generator, Args&&... _args)
{
    auto  _uuid = get_perfetto_category_uuid<CategoryT>(std::forward<Args>(_args)...);
    auto& _track_uuids = get_perfetto_track_uuids();
    if(_track_uuids.find(_uuid) == _track_uuids.end())
    {
        const auto _track = TrackT(_uuid, ::perfetto::ProcessTrack::Current());
        auto       _desc  = _track.Serialize();

        auto _name = std::forward<FuncT>(_desc_generator)(std::forward<Args>(_args)...);
        _desc.set_name(_name);
        ::perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

        ROCPROFSYS_VERBOSE_F(4, "[%s] Created %s(%zu) with description: \"%s\"\n",
                             trait::name<CategoryT>::value, demangle<TrackT>().c_str(),
                             _uuid, _name.c_str());

        _track_uuids.emplace(_uuid, _name);
    }

    // guard this with ppdefs in addition to runtime check to avoid
    // overhead of generating string during releases
#if defined(ROCPROFSYS_CI) && ROCPROFSYS_CI > 0
    auto _name = std::forward<FuncT>(_desc_generator)(std::forward<Args>(_args)...);
    ROCPROFSYS_CI_THROW(_track_uuids.at(_uuid) != _name,
                        "Error! Multiple invocations of UUID %zu produced different "
                        "descriptions: \"%s\" and \"%s\"\n",
                        _uuid, _track_uuids.at(_uuid).c_str(), _name.c_str());
#endif

    return TrackT(_uuid, ::perfetto::ProcessTrack::Current());
}

template <typename Tp = uint64_t>
ROCPROFSYS_INLINE auto
now()
{
    return ::tim::get_clock_real_now<Tp, std::nano>();
}

inline auto&
get_instrumentation_bundles(int64_t _tid = threading::get_id())
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
    int32_t profile = 0;  // use signed so compiler doesn't have to
    int32_t tracing = 0;  // account for underflow/overflow
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
        ROCPROFSYS_DEBUG("[%s] skipped %s :: empty bundle stack\n",
                         "rocprofsys_pop_trace", name.data());
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
        uint64_t _ts = now();
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
                     std::is_same<arg1_type, uint64_t>::value)
        {
            push_perfetto_track(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else if constexpr(std::is_same<arg0_type, uint64_t>::value)
        {
            push_perfetto_ts(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else
        {
            ++get_tracing_stack<CategoryT>();
            uint64_t _ts = now();
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
        uint64_t _ts = now();
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
                     std::is_same<arg1_type, uint64_t>::value)
        {
            pop_perfetto_track(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else if constexpr(std::is_same<arg0_type, uint64_t>::value)
        {
            pop_perfetto_ts(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else
        {
            // decrement tracing stack
            --get_tracing_stack<CategoryT>();
            uint64_t _ts = now();
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
push_perfetto_ts(CategoryT, const char* name, uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(category_push_disabled<CategoryT>()) return;

    ++get_tracing_stack<CategoryT>();
    TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, get_perfetto_string(name), _ts,
                      std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_ts(CategoryT, const char* name, uint64_t _ts, Args&&... args)
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
push_perfetto_track(CategoryT, const char* name, ::perfetto::Track _track, uint64_t _ts,
                    Args&&... args)
{
    // skip if category is disabled
    if(category_push_disabled<CategoryT>()) return;

    ++get_tracing_stack<CategoryT>();
    TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, get_perfetto_string(name), _track,
                      _ts, std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_track(CategoryT, const char* name, ::perfetto::Track _track, uint64_t _ts,
                   Args&&... args)
{
    // skip if category is disabled and not pushed on this thread
    if(tracing_pop_disabled<CategoryT>()) return;

    // decrement tracing stack
    --get_tracing_stack<CategoryT>();

    TRACE_EVENT_END(
        trait::name<CategoryT>::value, _track, _ts,
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
        uint64_t _ts = now();
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
                     std::is_same<arg1_type, uint64_t>::value)
        {
            mark_perfetto_track(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else if constexpr(std::is_same<arg0_type, uint64_t>::value)
        {
            mark_perfetto_ts(CategoryT{}, name, std::forward<Args>(args)...);
        }
        else
        {
            uint64_t _ts = now();
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
mark_perfetto_ts(CategoryT, const char* name, uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(category_mark_disabled<CategoryT>()) return;

    TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, get_perfetto_string(name), _ts,
                        std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
mark_perfetto_track(CategoryT, const char* name, ::perfetto::Track _track, uint64_t _ts,
                    Args&&... args)
{
    // skip if category is disabled
    if(category_mark_disabled<CategoryT>()) return;

    TRACE_EVENT_INSTANT(trait::name<CategoryT>::value, get_perfetto_string(name), _track,
                        _ts, std::forward<Args>(args)...);
}

template <typename FuncT>
int64_t
get_clock_skew(FuncT&& _timestamp_func, int64_t _n = 1)
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
        volatile uint64_t _cpu_ts = 0;
        volatile uint64_t _gpu_ts = 0;
        _cpu_ts += _cpu_now();
        _gpu_ts += _gpu_now();
        _cpu_ts += _cpu_now();
        return static_cast<int64_t>(_cpu_ts / 2) - static_cast<int64_t>(_gpu_ts);
    };

    int64_t _diff = 0;
    for(int64_t i = 0; i < _n; ++i)
    {
        _diff += _compute();
    }
    return (_diff / _n);
}
}  // namespace tracing
}  // namespace rocprofsys
