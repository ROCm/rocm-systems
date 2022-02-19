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
#include "library/sampling.hpp"

#include <PTL/ThreadPool.hh>
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
    _v.initializer  = []() {
        sampling::block_signals();
        threading::set_thread_name(
            TIMEMORY_JOIN('.', "ptl", PTL::Threading::GetThreadId()).c_str());
    };
    _v.finalizer = []() {};
    _v.priority  = 5;
    _v.pool_size = 1;
    return _v;
}();
}

std::mutex&
get_roctracer_mutex()
{
    static std::mutex _v{};
    return _v;
}

PTL::ThreadPool&
get_roctracer_thread_pool()
{
    static auto _v = PTL::ThreadPool{ _thread_pool_cfg };
    return _v;
}

PTL::TaskGroup<void>&
get_roctracer_task_group()
{
    static PTL::TaskGroup<void> _v{ &get_roctracer_thread_pool() };
    return _v;
}

std::mutex&
get_critical_trace_mutex()
{
    static std::mutex _v{};
    return _v;
}

PTL::ThreadPool&
get_critical_trace_thread_pool()
{
    static auto _v = PTL::ThreadPool{ _thread_pool_cfg };
    return _v;
}

PTL::TaskGroup<void>&
get_critical_trace_task_group()
{
    static PTL::TaskGroup<void> _v{ &get_critical_trace_thread_pool() };
    return _v;
}

}  // namespace tasking
}  // namespace omnitrace
