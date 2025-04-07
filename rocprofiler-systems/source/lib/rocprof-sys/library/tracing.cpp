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

#include "library/tracing.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/process/threading.hpp>

namespace rocprofsys
{
namespace tracing
{
namespace
{
tim::hash_map_ptr_t&
get_timemory_hash_ids(int64_t _tid = threading::get_id());

tim::hash_alias_ptr_t&
get_timemory_hash_aliases(int64_t _tid = threading::get_id());

tim::hash_map_ptr_t&
get_timemory_hash_ids(int64_t _tid)
{
    return thread_data<identity<tim::hash_map_ptr_t>>::instance(
        construct_on_thread{ _tid });
}

tim::hash_alias_ptr_t&
get_timemory_hash_aliases(int64_t _tid)
{
    return thread_data<identity<tim::hash_alias_ptr_t>>::instance(
        construct_on_thread{ _tid });
}
}  // namespace

bool debug_push = tim::get_env("ROCPROFSYS_DEBUG_PUSH", false) || get_debug_env();
bool debug_pop  = tim::get_env("ROCPROFSYS_DEBUG_POP", false) || get_debug_env();
bool debug_mark = tim::get_env("ROCPROFSYS_DEBUG_MARK", false) || get_debug_env();
bool debug_user = tim::get_env("ROCPROFSYS_DEBUG_USER_REGIONS", false) || get_debug_env();

std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids()
{
    static thread_local auto _v = std::unordered_map<hash_value_t, std::string>{};
    return _v;
}

void
copy_timemory_hash_ids()
{
    auto_lock_t _ilk{ type_mutex<tim::hash_map_t>(), std::defer_lock };
    auto_lock_t _alk{ type_mutex<tim::hash_alias_map_t>(), std::defer_lock };

    if(!_ilk.owns_lock()) _ilk.lock();
    if(!_alk.owns_lock()) _alk.lock();

    // copy these over so that all hashes are known
    auto& _hmain = tim::hash::get_main_hash_ids();
    auto& _amain = tim::hash::get_main_hash_aliases();
    ROCPROFSYS_REQUIRE(_hmain != nullptr) << "no main timemory hash ids";
    ROCPROFSYS_REQUIRE(_amain != nullptr) << "no main timemory hash aliases";

    // combine all the hash and alias info into one container
    for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
    {
        auto& _hitr = get_timemory_hash_ids(i);
        auto& _aitr = get_timemory_hash_aliases(i);

        if(_hitr)
        {
            for(const auto& itr : *_hitr)
                _hmain->emplace(itr.first, itr.second);
        }
        if(_aitr)
        {
            for(auto itr : *_aitr)
                _amain->emplace(itr.first, itr.second);
        }
    }

    // distribute the contents of that combined container to each thread-specific
    // container before finalizing
    if(get_state() == State::Finalized)
    {
        for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
        {
            auto& _hitr = get_timemory_hash_ids(i);
            auto& _aitr = get_timemory_hash_aliases(i);

            if(_hitr) *_hitr = *_hmain;
            if(_aitr) *_aitr = *_amain;
        }
    }
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
        const auto& _tinfo = thread_info::init();
        auto _tidx = (_tinfo && _tinfo->index_data) ? _tinfo->index_data->sequent_value
                                                    : threading::get_id();

        ROCPROFSYS_REQUIRE(_tidx >= 0)
            << "thread setup failed. thread info not initialized: " << [&_tinfo]() {
                   if(_tinfo) return JOIN("", *_tinfo);
                   return std::string{ "no thread_info" };
               }();

        if(_tidx > 0) threading::set_thread_name(JOIN(" ", "Thread", _tidx).c_str());
        thread_data<thread_bundle_t>::construct(
            JOIN('/', "rocprofsys/process", process::get_id(), "thread", _tidx),
            quirk::config<quirk::auto_start>{});
        // save the hash maps
        get_timemory_hash_ids(_tidx)     = tim::get_hash_ids();
        get_timemory_hash_aliases(_tidx) = tim::get_hash_aliases();

        ROCPROFSYS_REQUIRE(get_timemory_hash_ids(_tidx) != nullptr)
            << "no timemory hash ids pointer for thread " << _tidx;
        ROCPROFSYS_REQUIRE(get_timemory_hash_aliases(_tidx) != nullptr)
            << "no timemory hash aliases pointer for thread " << _tidx;

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
                ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
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
}  // namespace rocprofsys
