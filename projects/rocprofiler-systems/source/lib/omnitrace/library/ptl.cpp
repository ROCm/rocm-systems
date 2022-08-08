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
    PTL::ThreadPool::Config _v{};
    _v.init         = true;
    _v.use_affinity = false;
    _v.use_tbb      = false;
    _v.verbose      = -1;
    _v.initializer  = []() {
        threading::offset_this_id(true);
        set_thread_state(ThreadState::Internal);
        sampling::block_signals();
        threading::set_thread_name(
            JOIN('.', "ptl", PTL::Threading::GetThreadId()).c_str());
    };
    _v.finalizer = []() {};
    _v.priority  = 5;
    _v.pool_size = 1;
    return _v;
}();
}

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
{}

void
join()
{
    if(roctracer::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("waiting for all roctracer tasks to complete...\n");
        tasking::roctracer::get_task_group().join();
    }

    if(critical_trace::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("waiting for all critical tasks to complete...\n");
        tasking::critical_trace::get_task_group().join();
    }
}

void
shutdown()
{
    if(roctracer::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("Destroying the roctracer thread pool...\n");
        std::unique_lock<std::mutex> _lk{ roctracer::get_mutex() };
        roctracer::get_task_group().join();
        roctracer::get_task_group().clear();
        roctracer::get_task_group().set_pool(nullptr);
        roctracer::get_thread_pool().destroy_threadpool();
        roctracer::get_thread_pool_state() = State::Finalized;
    }

    if(critical_trace::get_thread_pool_state() == State::Active)
    {
        OMNITRACE_DEBUG_F("Destroying the critical trace thread pool...\n");
        std::unique_lock<std::mutex> _lk{ critical_trace::get_mutex() };
        critical_trace::get_task_group().join();
        critical_trace::get_task_group().clear();
        critical_trace::get_task_group().set_pool(nullptr);
        critical_trace::get_thread_pool().destroy_threadpool();
        critical_trace::get_thread_pool_state() = State::Finalized;
    }
}

std::mutex&
roctracer::get_mutex()
{
    static std::mutex _v{};
    return _v;
}

PTL::ThreadPool&
roctracer::get_thread_pool()
{
    static auto _v = (roctracer::get_thread_pool_state() = State::Active,
                      PTL::ThreadPool{ _thread_pool_cfg });
    return _v;
}

PTL::TaskGroup<void>&
roctracer::get_task_group()
{
    static PTL::TaskGroup<void> _v{ &roctracer::get_thread_pool() };
    return _v;
}

std::mutex&
critical_trace::get_mutex()
{
    static std::mutex _v{};
    return _v;
}

PTL::ThreadPool&
critical_trace::get_thread_pool()
{
    static auto _v = (critical_trace::get_thread_pool_state() = State::Active,
                      PTL::ThreadPool{ _thread_pool_cfg });
    return _v;
}

PTL::TaskGroup<void>&
critical_trace::get_task_group()
{
    static PTL::TaskGroup<void> _v{ &critical_trace::get_thread_pool() };
    return _v;
}

}  // namespace tasking
}  // namespace omnitrace
