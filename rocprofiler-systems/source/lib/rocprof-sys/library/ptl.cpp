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

#include "library/ptl.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/state.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <PTL/ThreadPool.hh>
#include <PTL/UserTaskQueue.hh>

#include <timemory/backends/threading.hpp>
#include <timemory/utility/declaration.hpp>

namespace rocprofsys
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

        _nthreads = get_env<int64_t>("ROCPROFSYS_THREAD_POOL_SIZE", -1, false);
        if(_nthreads == -1)
        {
            _nthreads = 4;
            if(_nthreads > _max_threads) _nthreads = _max_threads;
            if(_nthreads < _min_threads) _nthreads = _min_threads;

            tim::set_env("ROCPROFSYS_THREAD_POOL_SIZE", _nthreads, 0);
        }
    }

    static char  buffer[sizeof(PTL::UserTaskQueue)];
    static auto* _task_queue = new((void*) buffer) PTL::UserTaskQueue(_nthreads);

    PTL::ThreadPool::Config _v{};
    _v.init         = true;
    _v.use_affinity = false;
    _v.use_tbb      = false;
    _v.verbose      = -1;
    _v.initializer  = []() {
        thread_info::init(true);
        threading::set_thread_name(
            JOIN('.', "ptl", PTL::Threading::GetThreadId()).c_str());
        set_thread_state(ThreadState::Disabled);
        sampling::block_signals();
    };
    _v.finalizer  = []() {};
    _v.priority   = 5;
    _v.pool_size  = _nthreads;
    _v.task_queue = _task_queue;
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
    static auto  _cfg = _thread_pool_cfg();
    static auto* _v =
        (get_thread_pool_state() = State::Active, new PTL::ThreadPool{ _cfg });
    return *_v;
}
}  // namespace

namespace general
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
}  // namespace general

void
setup()
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
    (void) get_thread_pool();
}

void
join()
{
    if(general::get_thread_pool_state() == State::Active)
    {
        ROCPROFSYS_DEBUG_F("waiting for all general tasks to complete...\n");
        for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
            general::get_task_group(i).join();
    }
}

void
shutdown()
{
    if(general::get_thread_pool_state() == State::Active)
    {
        ROCPROFSYS_DEBUG_F("Waiting on completion of general tasks...\n");
        for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
        {
            general::get_task_group(i).join();
            general::get_task_group(i).clear();
            general::get_task_group(i).set_pool(nullptr);
        }
        general::get_thread_pool_state() = State::Finalized;
    }

    if(get_thread_pool_state() == State::Active)
    {
        ROCPROFSYS_DEBUG_F("Destroying the rocprof-sys thread pool...\n");
        get_thread_pool().destroy_threadpool();
        get_thread_pool_state() = State::Finalized;
    }
    else
    {
        ROCPROFSYS_DEBUG_F("thread-pool is not active...\n");
    }
}

size_t
initialize_threadpool(size_t _v)
{
    return get_thread_pool().initialize_threadpool(_v);
}

PTL::TaskGroup<void>&
general::get_task_group(int64_t _tid)
{
    struct local
    {};
    using thread_data_t = thread_data<PTL::TaskGroup<void>, local>;
    static thread_local auto& _v =
        thread_data_t::instance(construct_on_thread{ _tid }, &tasking::get_thread_pool());
    return *_v;
}

}  // namespace tasking
}  // namespace rocprofsys
