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

#include "library/components/pthread_mutex_gotcha.hpp"
#include "library.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_sampler.hpp"
#include "library/utility.hpp"
#include "timemory/backends/threading.hpp"

#include <timemory/utility/signals.hpp>
#include <timemory/utility/types.hpp>

#include <pthread.h>

namespace omnitrace
{
using Device = critical_trace::Device;
using Phase  = critical_trace::Phase;

pthread_mutex_gotcha::hash_array_t&
pthread_mutex_gotcha::get_hashes()
{
    // theoretically, this private function will NEVER be called until it
    // is called by a gotcha wrapper, which means the tool ids should be
    // fully populated. If that fails to be the case for some reason,
    // we could see weird results.
    static auto _v = []() {
        const auto&  _data = pthread_mutex_gotcha_t::get_gotcha_data();
        hash_array_t _init{};
        for(size_t i = 0; i < gotcha_capacity; ++i)
        {
            auto&& _id = _data.at(i).tool_id;
            if(!_id.empty())
                _init.at(i) = critical_trace::add_hash_id(_id.c_str());
            else
            {
                OMNITRACE_VERBOSE(
                    1,
                    "WARNING!!! pthread_mutex_gotcha tool id at index %zu was empty!\n",
                    i);
            }
            OMNITRACE_CI_FAIL(
                _id.empty() || _init.at(i) == 0,
                "pthread_mutex_gotcha tool id at index %zu has no hash value\n", i);
        }
        return _init;
    }();
    return _v;
}

void
pthread_mutex_gotcha::configure()
{
    pthread_mutex_gotcha_t::get_initializer() = []() {
        if(config::get_trace_thread_locks())
        {
            pthread_mutex_gotcha::validate();

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<0, int, pthread_mutex_t*>{ "pthread_mutex_lock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<1, int, pthread_mutex_t*>{ "pthread_mutex_unlock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<2, int, pthread_mutex_t*>{ "pthread_mutex_trylock" });
        }
    };
}

void
pthread_mutex_gotcha::shutdown()
{}

void
pthread_mutex_gotcha::validate()
{
    if(config::get_trace_thread_locks() && config::get_use_perfetto())
    {
        OMNITRACE_PRINT_F("\n");
        OMNITRACE_PRINT_F("\n");
        OMNITRACE_PRINT_F("\n");
        OMNITRACE_PRINT_F(
            "The overhead of all the mutex locking internally by perfetto is\n")
        OMNITRACE_PRINT_F(
            "so significant that all timing data is rendered meaningless.\n");
        OMNITRACE_PRINT_F(
            "However, mutex locking is effectively non-existant in timemory.\n");
        OMNITRACE_PRINT_F("If you want to trace the mutex locking:\n")
        OMNITRACE_PRINT_F("   OMNITRACE_USE_TIMEMORY=ON\n");
        OMNITRACE_PRINT_F("   OMNITRACE_USE_PERFETTO=OFF\n");
        OMNITRACE_PRINT_F("\n");
        OMNITRACE_PRINT_F("\n");
        OMNITRACE_PRINT_F("\n");
        OMNITRACE_FAIL_F("OMNITRACE_USE_PERFETTO and OMNITRACE_TRACE_THREAD_LOCKS cannot "
                         "both be enabled.\n");
    }
}

int
pthread_mutex_gotcha::operator()(const gotcha_data_t& _data,
                                 int (*_callee)(pthread_mutex_t*),
                                 pthread_mutex_t* _mutex)
{
    if(is_disabled())
    {
        if(!_callee)
        {
            OMNITRACE_PRINT("Warning! nullptr to %s\n", _data.tool_id.c_str());
            return EINVAL;
        }
        return (*_callee)(_mutex);
    }

    uint64_t _cid        = 0;
    uint64_t _parent_cid = 0;
    uint16_t _depth      = 0;
    int64_t  _ts         = 0;

    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    if(get_use_critical_trace())
    {
        std::tie(_cid, _parent_cid, _depth) = create_cpu_cid_entry();
        _ts                                 = comp::wall_clock::record();
    }

    omnitrace_push_region(_data.tool_id.c_str());
    auto _ret = (*_callee)(_mutex);
    omnitrace_pop_region(_data.tool_id.c_str());

    if(get_use_critical_trace())
    {
        add_critical_trace<Device::CPU, Phase::DELTA>(
            threading::get_id(), _cid, 0, _parent_cid, _ts, comp::wall_clock::record(),
            reinterpret_cast<uintptr_t>(_mutex), get_hashes().at(_data.index), _depth);
    }

    return _ret;
}

bool
pthread_mutex_gotcha::is_disabled()
{
    return (omnitrace::get_state() != omnitrace::State::Active ||
            omnitrace::get_thread_state() != omnitrace::ThreadState::Enabled ||
            (get_use_sampling() && !pthread_gotcha::sampling_enabled_on_child_threads()));
}
}  // namespace omnitrace
