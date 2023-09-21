// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <PTL/TaskGroup.hh>
#include <PTL/ThreadPool.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace internal_threading
{
using thread_pool_t        = PTL::ThreadPool;
using task_group_t         = PTL::TaskGroup<void>;
using unique_thread_pool_t = std::unique_ptr<thread_pool_t, void (*)(thread_pool_t*)>;
using unique_task_group_t  = std::unique_ptr<task_group_t>;
using thread_pool_vec_t    = std::vector<unique_thread_pool_t>;
using task_group_vec_t     = std::vector<unique_task_group_t>;

void notify_pre_internal_thread_create(rocprofiler_internal_thread_library_t);
void notify_post_internal_thread_create(rocprofiler_internal_thread_library_t);

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
