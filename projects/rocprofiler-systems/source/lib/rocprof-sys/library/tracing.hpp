// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/control/clocks/timeline.hpp"
#include "core/demangler.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/causal/sampling.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
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

namespace rocprofsys::tracing
{
using interval_data_instances = thread_data<std::vector<bool>>;

//
//  declarations
//
extern ROCPROFSYS_HIDDEN_API bool debug_push;
extern ROCPROFSYS_HIDDEN_API bool debug_pop;
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

template <typename Tp = std::uint64_t>
ROCPROFSYS_INLINE auto
now()
{
    return control::clocks::timeline_ns<Tp>();
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
    if(category_push_disabled<CategoryT>())
    {
        return;
    }

    auto& _data = tracing::get_instrumentation_bundles();
    if(ROCPROFSYS_LIKELY(_data != nullptr))
    {
        // this generates a hash for the raw string array
        auto const _hash = tim::add_hash_id(name);
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
    if(profile_pop_disabled<CategoryT>())
    {
        return return_type{ nullptr, -1 };
    }

    auto const _hash = tim::hash::get_hash_id(name);
    auto&      _data = tracing::get_instrumentation_bundles();
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
    if(_data->size() > 1)
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
    if(profile_pop_disabled<CategoryT>())
    {
        return return_type{ nullptr, -1 };
    }

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
        {
            _bundles->destroy(_data.first, _data.second);
        }
    }
}

template <typename CategoryT, typename... Args>
inline void
pop_timemory(CategoryT, std::string_view name, Args&&... args)
{
    // skip if category is disabled and not pushed on this thread
    if(profile_pop_disabled<CategoryT>())
    {
        return;
    }

    auto _data = stop_timemory(CategoryT{}, name, std::forward<Args>(args)...);
    if(_data.first)
    {
        destroy_timemory(std::move(_data));
    }
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
}  // namespace rocprofsys::tracing
