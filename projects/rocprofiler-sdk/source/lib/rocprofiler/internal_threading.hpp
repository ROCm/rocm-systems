// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler/internal_threading.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/allocator.hpp"

#include <PTL/TaskGroup.hh>
#include <PTL/ThreadPool.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace internal_threading
{
class ThreadPool : public PTL::ThreadPool
{
public:
    using parent_type = PTL::ThreadPool;

    ThreadPool(const parent_type::Config&);
    ~ThreadPool();
};

class TaskGroup : public PTL::TaskGroup<void>
{
public:
    using parent_type = PTL::TaskGroup<void>;

    TaskGroup(std::shared_ptr<ThreadPool>);

private:
    std::shared_ptr<ThreadPool> m_pool = {};
};

using thread_pool_t = ThreadPool;
using task_group_t  = TaskGroup;

void notify_pre_internal_thread_create(rocprofiler_runtime_library_t);
void notify_post_internal_thread_create(rocprofiler_runtime_library_t);

// initialize the default thread pool
void
initialize();

// destroy all the thread pools
void
finalize();

// creates a new thread
rocprofiler_callback_thread_t
create_callback_thread();

// returns the task group for the given callback thread identifier
task_group_t* get_task_group(rocprofiler_callback_thread_t);
}  // namespace internal_threading
}  // namespace rocprofiler
