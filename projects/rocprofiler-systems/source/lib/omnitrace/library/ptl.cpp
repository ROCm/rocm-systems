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

#include "library/ptl.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <PTL/ThreadPool.hh>

#include <timemory/backends/threading.hpp>
#include <timemory/utility/declaration.hpp>

namespace omnitrace
{
namespace tasking
{
namespace
{
auto _thread_pool_cfg = []() {
    int64_t _nthreads = 0;
    if(config::settings_are_configured())
    {
        _nthreads = config::get_thread_pool_size();
    }
    else
    {
        const int64_t _max_threads = std::thread::hardware_concurrency() / 2;
        const int64_t _min_threads = 1;

        _nthreads = get_env<int64_t>("OMNITRACE_THREAD_POOL_SIZE", -1, false);
        if(_nthreads == -1)
        {
            _nthreads = 4;
            if(_nthreads > _max_threads) _nthreads = _max_threads;
            if(_nthreads < _min_threads) _nthreads = _min_threads;

            tim::set_env("OMNITRACE_THREAD_POOL_SIZE", _nthreads, 0);
        }
    }

    PTL::ThreadPool::Config _v{};
    _v.init         = true;
    _v.use_affinity = false;
    _v.use_tbb      = false;
    _v.verbose      = -1;
    _v.initializer  = []() {
        thread_info::init(true);
        threading::set_thread_name(
            JOIN('.', "ptl", PTL::Threading::GetThreadId()).c_str());
        sampling::block_signals();
    };
    _v.finalizer = []() {};
    _v.priority  = 5;
    _v.pool_size = _nthreads;
    return _v;
};

auto&
get_thread_pool_state()
{
    static auto _v = State::PreInit;
    return _v;
}

PTL::ThreadPool&
get_thread_pool()
{
    static auto _v =
        (get_thread_pool_state() = State::Active, PTL::ThreadPool{ _thread_pool_cfg() });
    return _v;
}
}  // namespace

namespace roctracer
{
namespace
{
auto&
get_thread_pool_state()
{
    static auto _v = State::PreInit;
    return _v;
}
}  // namespace
}  // namespace roctracer

namespace critical_trace
{
namespace
{
auto&
get_thread_pool_state()
{
    static auto _v = State::PreInit;
    return _v;
}
}  // namespace
}  // namespace critical_trace

void
setup()
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    (void) get_thread_pool();
}

void
join()
{
    if(roctracer::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("waiting for all roctracer tasks to complete...\n");
        for(size_t i = 0; i < max_supported_threads; ++i)
            roctracer::get_task_group(i).join();
    }

    if(critical_trace::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("waiting for all critical tasks to complete...\n");
        for(size_t i = 0; i < max_supported_threads; ++i)
            critical_trace::get_task_group(i).join();
    }
}

void
shutdown()
{
    if(roctracer::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("Waiting on completion of roctracer tasks...\n");
        for(size_t i = 0; i < max_supported_threads; ++i)
        {
            roctracer::get_task_group(i).join();
            roctracer::get_task_group(i).clear();
            roctracer::get_task_group(i).set_pool(nullptr);
        }
        roctracer::get_thread_pool_state() = State::Finalized;
    }

    if(critical_trace::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("Waiting on completion of critical trace tasks...\n");
        for(size_t i = 0; i < max_supported_threads; ++i)
        {
            critical_trace::get_task_group(i).join();
            critical_trace::get_task_group(i).clear();
            critical_trace::get_task_group(i).set_pool(nullptr);
        }
        critical_trace::get_thread_pool_state() = State::Finalized;
    }

    if(get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("Destroying the omnitrace thread pool...\n");
        get_thread_pool().destroy_threadpool();
        get_thread_pool_state() = State::Finalized;
    }
}

size_t
initialize_threadpool(size_t _v)
{
    return get_thread_pool().initialize_threadpool(_v);
}

PTL::TaskGroup<void>&
roctracer::get_task_group(int64_t _tid)
{
    struct local
    {};
    using thread_data_t = thread_data<PTL::TaskGroup<void>, local>;
    static auto& _v =
        thread_data_t::instances(construct_on_init{}, &tasking::get_thread_pool());
    return *_v.at(_tid);
}

PTL::TaskGroup<void>&
critical_trace::get_task_group(int64_t _tid)
{
    struct local
    {};
    using thread_data_t = thread_data<PTL::TaskGroup<void>, local>;
    static auto& _v =
        thread_data_t::instances(construct_on_init{}, &tasking::get_thread_pool());
    return *_v.at(_tid);
}
}  // namespace tasking
}  // namespace omnitrace
