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

#include <cstdint>
#include <pthread.h>
#include <stdexcept>

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
        hash_array_t _init = {};
        size_t       i0    = (config::get_trace_thread_locks()) ? 0 : 3;
        for(size_t i = i0; i < gotcha_capacity; ++i)
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

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<3, int, pthread_barrier_t*>{ "pthread_barrier_wait" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<4, int, pthread_rwlock_t*>{ "pthread_rwlock_rdlock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<5, int, pthread_rwlock_t*>{ "pthread_rwlock_tryrdlock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<6, int, pthread_rwlock_t*>{ "pthread_rwlock_trywrlock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<7, int, pthread_rwlock_t*>{ "pthread_rwlock_unlock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<8, int, pthread_rwlock_t*>{ "pthread_rwlock_wrlock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<9, int, pthread_spinlock_t*>{ "pthread_spin_lock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<10, int, pthread_spinlock_t*>{ "pthread_spin_trylock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<11, int, pthread_spinlock_t*>{ "pthread_spin_unlock" });

        pthread_mutex_gotcha_t::configure(
            comp::gotcha_config<12, int, pthread_t, void**>{ "pthread_join" });
    };
}

void
pthread_mutex_gotcha::shutdown()
{
    pthread_mutex_gotcha_t::disable();
}

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

template <typename... Args>
auto
pthread_mutex_gotcha::operator()(uintptr_t&& _id, const comp::gotcha_data& _data,
                                 int (*_callee)(Args...), Args... _args) const
{
    if(is_disabled())
    {
        if(!_callee)
        {
            OMNITRACE_PRINT("Warning! nullptr to %s\n", _data.tool_id.c_str());
            return EINVAL;
        }
        return (*_callee)(_args...);
    }

    uint64_t _cid        = 0;
    uint64_t _parent_cid = 0;
    uint32_t _depth      = 0;
    int64_t  _ts         = 0;

    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    if(_id < std::numeric_limits<uintptr_t>::max() && get_use_critical_trace())
    {
        std::tie(_cid, _parent_cid, _depth) = create_cpu_cid_entry();
        _ts                                 = comp::wall_clock::record();
    }

    omnitrace_push_region(_data.tool_id.c_str());
    auto _ret = (*_callee)(_args...);
    omnitrace_pop_region(_data.tool_id.c_str());

    if(_id < std::numeric_limits<uintptr_t>::max() && get_use_critical_trace())
    {
        add_critical_trace<Device::CPU, Phase::DELTA>(
            threading::get_id(), _cid, 0, _parent_cid, _ts, comp::wall_clock::record(), 0,
            _id, get_hashes().at(_data.index), _depth);
    }

    return _ret;
    tim::consume_parameters(_id, _cid, _parent_cid, _depth, _ts);
}

int
pthread_mutex_gotcha::operator()(const gotcha_data_t& _data,
                                 int (*_callee)(pthread_mutex_t*),
                                 pthread_mutex_t* _mutex) const
{
    return (*this)(reinterpret_cast<uintptr_t>(_mutex), _data, _callee, _mutex);
}

int
pthread_mutex_gotcha::operator()(const gotcha_data_t& _data,
                                 int (*_callee)(pthread_spinlock_t*),
                                 pthread_spinlock_t* _lock) const
{
    return (*this)(reinterpret_cast<uintptr_t>(_lock), _data, _callee, _lock);
}

int
pthread_mutex_gotcha::operator()(const gotcha_data_t& _data,
                                 int (*_callee)(pthread_rwlock_t*),
                                 pthread_rwlock_t* _lock) const
{
    return (*this)(reinterpret_cast<uintptr_t>(_lock), _data, _callee, _lock);
}

int
pthread_mutex_gotcha::operator()(const gotcha_data_t& _data,
                                 int (*_callee)(pthread_barrier_t*),
                                 pthread_barrier_t* _barrier) const
{
    return (*this)(reinterpret_cast<uintptr_t>(_barrier), _data, _callee, _barrier);
}

int
pthread_mutex_gotcha::operator()(const gotcha_data_t& _data,
                                 int (*_callee)(pthread_t, void**), pthread_t _thr,
                                 void** _tinfo) const
{
    return (*this)(static_cast<uintptr_t>(threading::get_id()), _data, _callee, _thr,
                   _tinfo);
}

bool
pthread_mutex_gotcha::is_disabled()
{
    return (omnitrace::get_state() != omnitrace::State::Active ||
            omnitrace::get_thread_state() != omnitrace::ThreadState::Enabled ||
            (get_use_sampling() && !pthread_gotcha::sampling_enabled_on_child_threads()));
}
}  // namespace omnitrace
