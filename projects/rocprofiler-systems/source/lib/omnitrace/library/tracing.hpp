// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/perfetto.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/timemory.hpp"
#include "library/utility.hpp"

namespace omnitrace
{
namespace tracing
{
using interval_data_instances = thread_data<std::vector<bool>>;

std::unique_ptr<perfetto::TracingSession>&
get_trace_session();

std::vector<std::function<void()>>&
get_finalization_functions();

tim::hash_map_ptr_t&
get_timemory_hash_ids(int64_t _tid = threading::get_id());

tim::hash_alias_ptr_t&
get_timemory_hash_aliases(int64_t _tid = threading::get_id());

namespace
{
bool debug_push =  // NOLINT
    tim::get_env("OMNITRACE_DEBUG_PUSH", false) || get_debug_env();
bool debug_pop =  // NOLINT
    tim::get_env("OMNITRACE_DEBUG_POP", false) || get_debug_env();
bool debug_user =  // NOLINT
    tim::get_env("OMNITRACE_DEBUG_USER_REGIONS", false) || get_debug_env();
}  // namespace

inline auto&
get_interval_data(int64_t _tid = threading::get_id())
{
    static auto& _v =
        interval_data_instances::instances(interval_data_instances::construct_on_init{});
    return _v.at(_tid);
}

inline auto&
get_instrumentation_bundles(int64_t _tid = threading::get_id())
{
    static thread_local auto& _v = instrumentation_bundles::instances().at(_tid);
    return _v;
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

inline void
thread_init()
{
    static thread_local auto _thread_setup = []() {
        if(threading::get_id() > 0)
            threading::set_thread_name(JOIN(" ", "Thread", threading::get_id()).c_str());
        thread_data<omnitrace_thread_bundle_t>::construct(
            JOIN('/', "omnitrace/process", process::get_id(), "thread",
                 threading::get_id()),
            quirk::config<quirk::auto_start>{});
        get_interval_data()->reserve(512);
        // save the hash maps
        get_timemory_hash_ids()     = tim::get_hash_ids();
        get_timemory_hash_aliases() = tim::get_hash_aliases();
        return true;
    }();
    static thread_local auto _dtor = scope::destructor{ []() {
        if(get_state() != State::Finalized)
        {
            if(get_use_sampling()) sampling::shutdown();
            auto& _thr_bundle = thread_data<omnitrace_thread_bundle_t>::instance();
            if(_thr_bundle && _thr_bundle->get<comp::wall_clock>() &&
               _thr_bundle->get<comp::wall_clock>()->get_is_running())
                _thr_bundle->stop();
        }
    } };
    (void) _thread_setup;
    (void) _dtor;
}

inline void
thread_init_sampling()
{
    static thread_local auto _v = []() {
        auto _idx = utility::get_thread_index();
        // the main thread will initialize sampling when it initializes the tooling
        if(_idx > 0)
        {
            auto _use_sampling = get_use_sampling();
            if(_use_sampling) sampling::setup();
            return _use_sampling;
        }
        return false;
    }();
    (void) _v;
}

template <typename... Args>
inline void
push_timemory(const char* name, Args&&... args)
{
    auto& _data = tracing::get_instrumentation_bundles();
    // this generates a hash for the raw string array
    auto  _hash   = tim::add_hash_id(tim::string_view_t{ name });
    auto* _bundle = _data.allocator.allocate(1);
    _data.bundles.emplace_back(_bundle);
    _data.allocator.construct(_bundle, _hash);
    _bundle->start(std::forward<Args>(args)...);
}

template <typename... Args>
inline void
pop_timemory(const char* name, Args&&... args)
{
    auto  _hash = tim::hash::get_hash_id(tim::string_view_t{ name });
    auto& _data = tracing::get_instrumentation_bundles();
    if(_data.bundles.empty())
    {
        OMNITRACE_DEBUG("[%s] skipped %s :: empty bundle stack\n", "omnitrace_pop_trace",
                        name);
        return;
    }
    for(size_t i = _data.bundles.size(); i > 0; --i)
    {
        auto*& _v = _data.bundles.at(i - 1);
        if(_v->get_hash() == _hash)
        {
            _v->stop(std::forward<Args>(args)...);
            _data.allocator.destroy(_v);
            _data.allocator.deallocate(_v, 1);
            _data.bundles.erase(_data.bundles.begin() + (i - 1));
            break;
        }
    }
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto(CategoryT, const char* name, Args&&... args)
{
    uint64_t _ts = comp::wall_clock::record();
    TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, perfetto::StaticString(name), _ts,
                      std::forward<Args>(args)..., "begin_ns", _ts);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto(CategoryT, const char*, Args&&... args)
{
    uint64_t _ts = comp::wall_clock::record();
    TRACE_EVENT_END(trait::name<CategoryT>::value, _ts, std::forward<Args>(args)...,
                    "end_ns", _ts);
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto_ts(CategoryT, const char* name, uint64_t _ts, Args&&... args)
{
    TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, perfetto::StaticString(name), _ts,
                      std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_ts(CategoryT, const char*, uint64_t _ts, Args&&... args)
{
    TRACE_EVENT_END(trait::name<CategoryT>::value, _ts, std::forward<Args>(args)...);
}
}  // namespace tracing
}  // namespace omnitrace
