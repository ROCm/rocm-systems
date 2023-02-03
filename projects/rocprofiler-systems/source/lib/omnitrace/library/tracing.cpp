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

#include "library/tracing.hpp"
#include "library/config.hpp"
#include "library/state.hpp"
#include "library/thread_info.hpp"

namespace omnitrace
{
namespace tracing
{
bool debug_push = tim::get_env("OMNITRACE_DEBUG_PUSH", false) || get_debug_env();
bool debug_pop  = tim::get_env("OMNITRACE_DEBUG_POP", false) || get_debug_env();
bool debug_mark = tim::get_env("OMNITRACE_DEBUG_MARK", false) || get_debug_env();
bool debug_user = tim::get_env("OMNITRACE_DEBUG_USER_REGIONS", false) || get_debug_env();

std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids()
{
    static thread_local auto _v = std::unordered_map<hash_value_t, std::string>{};
    return _v;
}

tim::hash_map_ptr_t&
get_timemory_hash_ids(int64_t _tid)
{
    static auto _v = std::array<tim::hash_map_ptr_t, omnitrace::max_supported_threads>{};
    return _v.at(_tid);
}

tim::hash_alias_ptr_t&
get_timemory_hash_aliases(int64_t _tid)
{
    static auto _v =
        std::array<tim::hash_alias_ptr_t, omnitrace::max_supported_threads>{};
    return _v.at(_tid);
}

std::vector<std::function<void()>>&
get_finalization_functions()
{
    static auto _v = std::vector<std::function<void()>>{};
    return _v;
}

void
record_thread_start_time()
{
    static thread_local std::once_flag _once{};
    std::call_once(_once, []() {
        thread_info::set_start(comp::wall_clock::record(), get_mode() != Mode::Sampling);
    });
}

void
thread_init()
{
    if(get_thread_state() == ThreadState::Disabled) return;

    static thread_local auto _thread_dtor = scope::destructor{ []() {
        if(get_state() != State::Finalized)
        {
            if(get_use_causal())
                causal::sampling::shutdown();
            else if(get_use_sampling())
                sampling::shutdown();
            auto& _thr_bundle = thread_data<thread_bundle_t>::instance();
            if(_thr_bundle && _thr_bundle->get<comp::wall_clock>() &&
               _thr_bundle->get<comp::wall_clock>()->get_is_running())
                _thr_bundle->stop();
        }
    } };

    if(get_thread_state() == ThreadState::Disabled) return;

    static thread_local auto _thread_setup = []() {
        if(threading::get_id() > 0)
            threading::set_thread_name(JOIN(" ", "Thread", threading::get_id()).c_str());
        thread_data<thread_bundle_t>::construct(JOIN('/', "omnitrace/process",
                                                     process::get_id(), "thread",
                                                     threading::get_id()),
                                                quirk::config<quirk::auto_start>{});
        // save the hash maps
        get_timemory_hash_ids()     = tim::get_hash_ids();
        get_timemory_hash_aliases() = tim::get_hash_aliases();
        record_thread_start_time();
        return true;
    }();

    if(get_thread_state() == ThreadState::Disabled) return;

    static thread_local auto _sample_setup = []() {
        auto _idx = utility::get_thread_index();
        // the main thread will initialize sampling when it initializes the tooling
        if(_idx > 0)
        {
            auto _use_causal   = get_use_causal();
            auto _use_sampling = get_use_sampling();
            if(_use_causal || _use_sampling)
            {
                OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
                if(_use_causal)
                    causal::sampling::setup();
                else if(_use_sampling)
                    sampling::setup();
            }
            return (_use_causal || _use_sampling);
        }
        return false;
    }();

    (void) _thread_dtor;
    (void) _thread_setup;
    (void) _sample_setup;
}
}  // namespace tracing
}  // namespace omnitrace
